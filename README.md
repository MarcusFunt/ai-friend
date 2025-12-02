# Parakeet TDT Push-to-Talk Demo

This repository provides a minimal web UI and Python backend to try the [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) automatic speech recognition model. The interface runs in a browser and uses a press-and-hold workflow with the space bar.

## Running locally with Docker

```bash
docker build -t parakeet-tdt-demo .
docker run -p 8000:8000 parakeet-tdt-demo
```

Then open http://localhost:8000/ and hold the space bar to record. The audio clip is sent to the backend, which streams it through the model and returns the transcription.

## How it works

- **Backend:** FastAPI loads the Parakeet TDT model with the Transformers speech-recognition pipeline and exposes a `/api/transcribe` endpoint. Audio uploads are processed in a worker thread to keep the event loop responsive.
- **Frontend:** A simple HTML/CSS/JS page sets up MediaRecorder, toggled by spacebar presses (or mouse clicks). When recording stops, the captured audio is posted to the backend and the transcript is displayed.
