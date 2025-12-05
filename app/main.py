import asyncio
import logging
import os
import subprocess
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

MODEL_ID = os.getenv("MODEL_ID", "openai/whisper-large-v3-turbo")
TARGET_LANGUAGE = os.getenv("TARGET_LANGUAGE", "en")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "96"))
CHUNK_LENGTH_S = int(os.getenv("CHUNK_LENGTH_S", "20"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))

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
        max_new_tokens=MAX_NEW_TOKENS,
        chunk_length_s=CHUNK_LENGTH_S,
        batch_size=BATCH_SIZE,
        return_timestamps=False,
        dtype=TORCH_DTYPE,
        device=DEVICE,
        language=TARGET_LANGUAGE,
        generate_kwargs={"task": "transcribe", "language": TARGET_LANGUAGE},
        ignore_warning=True,
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
    converted_path = ""
    try:
        suffix = Path(file.filename).suffix or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        loop = asyncio.get_running_loop()
        try:
            converted_path = await loop.run_in_executor(None, convert_to_wav, tmp_path)
            audio_input = converted_path
        except subprocess.CalledProcessError as exc:
            logger.warning("Falling back to original audio after ffmpeg failure: %s", exc)
            audio_input = tmp_path

        # Lazily load the model to keep startup responsive.
        asr_pipeline_instance = await get_asr_pipeline()

        # Run inference in a thread to avoid blocking the event loop.
        result = await loop.run_in_executor(None, asr_pipeline_instance, audio_input)

        if not result or not result.get("text"):
            message = "Transcription returned no results."
            logger.error(message)
            return JSONResponse({"error": message}, status_code=500)

        return {"text": result["text"].strip()}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Transcription failed: %s", exc)
        return JSONResponse({"error": "Transcription failed."}, status_code=500)
    finally:
        for path in (tmp_path, converted_path):
            if path and os.path.exists(path):
                os.remove(path)


def convert_to_wav(source_path: str) -> str:
    """Converts an input audio file to a mono 16kHz WAV file using ffmpeg."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        output_path = tmp.name

    command = [
        "ffmpeg",
        "-y",
        "-i",
        source_path,
        "-ac",
        "1",
        "-ar",
        "16000",
        output_path,
    ]

    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return output_path
