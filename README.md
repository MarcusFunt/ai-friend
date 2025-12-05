# Whisper Large V3 Turbo Push-to-Talk Demo

This repository provides a minimal web UI and Python backend to try the [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) automatic speech recognition model. The interface runs in a browser and uses a press-and-hold workflow with the space bar.

## Running locally with Docker

```bash
docker build -t whisper-large-v3-turbo-demo .
docker run -p 8000:8000 whisper-large-v3-turbo-demo
```

Then open http://localhost:8000/ and hold the space bar to record. The audio clip is sent to the backend, which streams it through the model and returns the transcription.

### Performance tips

- For the fastest latency, run the container on a machine with a CUDA GPU. The service will automatically switch to `float16` on GPU and fall back to CPU otherwise.
- To trade a bit of quality for speed on CPU, try a smaller model by setting `MODEL_ID=openai/whisper-small` (or `openai/whisper-medium`).
- You can tune decoding and chunking without code changes:

  ```bash
  docker run -p 8000:8000 \
    -e MODEL_ID=openai/whisper-small \
    -e MAX_NEW_TOKENS=96 \
    -e CHUNK_LENGTH_S=20 \
    -e BATCH_SIZE=8 \
    whisper-large-v3-turbo-demo
  ```

  Reducing `MAX_NEW_TOKENS`, `CHUNK_LENGTH_S`, and `BATCH_SIZE` shortens inference time and memory use at the cost of some accuracy on long clips.

## How it works

- **Backend:** FastAPI loads the Whisper Large V3 Turbo model with the Transformers speech-recognition pipeline and exposes a `/api/transcribe` endpoint. Audio uploads are processed in a worker thread to keep the event loop responsive.
- **Frontend:** A simple HTML/CSS/JS page sets up MediaRecorder, toggled by spacebar presses (or mouse clicks). When recording stops, the captured audio is posted to the backend and the transcript is displayed.
