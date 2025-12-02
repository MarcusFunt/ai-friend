import asyncio
import os
import tempfile
from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

MODEL_ID = os.getenv("MODEL_ID", "nvidia/parakeet-tdt-0.6b-v3")


def load_asr_pipeline():
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    device = 0 if torch.cuda.is_available() else "cpu"

    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        max_new_tokens=128,
        chunk_length_s=30,
    )

app = FastAPI(title="Parakeet TDT Demo", description="Push-to-talk speech-to-text demo")
pipeline_lock = asyncio.Lock()
asr_pipeline = None


async def get_asr_pipeline():
    global asr_pipeline
    if asr_pipeline is not None:
        return asr_pipeline

    async with pipeline_lock:
        if asr_pipeline is None:
            loop = asyncio.get_running_loop()
            asr_pipeline = await loop.run_in_executor(None, load_asr_pipeline)

    return asr_pipeline

app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = Path(__file__).parent / "static" / "index.html"
    return FileResponse(index_path)


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        suffix = Path(file.filename).suffix or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        # Lazily load the pipeline to keep startup responsive.
        loop = asyncio.get_running_loop()
        asr_pipeline = await get_asr_pipeline()

        # Run inference in a thread to avoid blocking the event loop.
        result = await loop.run_in_executor(None, asr_pipeline, tmp_path)
        transcription = result["text"]

        return {"text": transcription}
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=500)
    finally:
        if "tmp_path" in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
