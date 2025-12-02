import asyncio
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import nemo.collections.asr as nemo_asr

MODEL_ID = os.getenv("MODEL_ID", "nvidia/parakeet-tdt-0.6b-v3")


def load_asr_model():
    """Loads the ASR model."""
    return nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_ID)

app = FastAPI(title="Parakeet TDT Demo", description="Push-to-talk speech-to-text demo")
model_lock = asyncio.Lock()
asr_model = None


async def get_asr_model():
    """Gets the ASR model, loading it if necessary."""
    global asr_model
    if asr_model is not None:
        return asr_model

    async with model_lock:
        if asr_model is None:
            loop = asyncio.get_running_loop()
            asr_model = await loop.run_in_executor(None, load_asr_model)

    return asr_model

app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = Path(__file__).parent / "static" / "index.html"
    return FileResponse(index_path)


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Transcribes an audio file."""
    try:
        suffix = Path(file.filename).suffix or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        # Lazily load the model to keep startup responsive.
        asr_model_instance = await get_asr_model()

        # Run inference in a thread to avoid blocking the event loop.
        loop = asyncio.get_running_loop()
        # The transcribe method expects a list of file paths.
        result = await loop.run_in_executor(None, asr_model_instance.transcribe, [tmp_path])

        # The result is a list of transcription strings.
        transcription = result[0] if result else ""

        return {"text": transcription}
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=500)
    finally:
        if "tmp_path" in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
