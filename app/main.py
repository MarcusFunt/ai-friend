import asyncio
import logging
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import librosa

MODEL_ID = os.getenv("MODEL_ID", "openai/whisper-large-v3-turbo")
# Use float16 for faster inference, and chunking for long-form audio.
# See: https://huggingface.co/openai/whisper-large-v3-turbo#long-form-transcription
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

logger = logging.getLogger(__name__)


def load_asr_model():
    """Loads the ASR model."""
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID, dtype=TORCH_DTYPE, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(DEVICE)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        dtype=TORCH_DTYPE,
        device=DEVICE,
    )

app = FastAPI(title="Whisper Demo", description="Push-to-talk speech-to-text demo")
model_lock = asyncio.Lock()
asr_pipeline = None


async def get_asr_pipeline():
    """Gets the ASR pipeline, loading it if necessary."""
    global asr_pipeline
    if asr_pipeline is not None:
        return asr_pipeline

    async with model_lock:
        if asr_pipeline is None:
            loop = asyncio.get_running_loop()
            asr_pipeline = await loop.run_in_executor(None, load_asr_model)

    return asr_pipeline

app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = Path(__file__).parent / "static" / "index.html"
    return FileResponse(index_path)


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Transcribes an audio file."""
    tmp_path = ""
    try:
        suffix = Path(file.filename).suffix or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        # Lazily load the model to keep startup responsive.
        asr_pipeline_instance = await get_asr_pipeline()

        # Load the audio file
        audio, sr = librosa.load(tmp_path, sr=16000)

        # Run inference in a thread to avoid blocking the event loop.
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, asr_pipeline_instance, audio)

        if not result or not result.get("text"):
            message = "Transcription returned no results."
            logger.error(message)
            return JSONResponse({"error": message}, status_code=500)

        return {"text": result["text"].strip()}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Transcription failed: %s", exc)
        return JSONResponse({"error": "Transcription failed."}, status_code=500)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
