#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME=${IMAGE_NAME:-parakeet-tdt-demo}
CACHE_VOLUME=${CACHE_VOLUME:-parakeet-tdt-cache}
PORT=${PORT:-8000}

log() {
  echo "[installer] $*"
}

if ! command -v docker >/dev/null 2>&1; then
  log "Docker is required but was not found in PATH. Please install Docker and try again."
  exit 1
fi

log "Ensuring Hugging Face cache volume '${CACHE_VOLUME}' exists..."
if ! docker volume inspect "${CACHE_VOLUME}" >/dev/null 2>&1; then
  docker volume create "${CACHE_VOLUME}" >/dev/null
  log "Created cache volume '${CACHE_VOLUME}'."
fi

log "Building Docker image '${IMAGE_NAME}'..."
docker build -t "${IMAGE_NAME}" .

log "Downloading model artifacts into the cache volume..."
docker run --rm \
  --name "${IMAGE_NAME}-setup" \
  -v "${CACHE_VOLUME}:/root/.cache/huggingface" \
  "${IMAGE_NAME}" \
  python - <<'PY'
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"

print("Downloading model and processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
)

print("Warming pipeline cache...")
_ = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch.float32,
    device="cpu",
    max_new_tokens=2,
    chunk_length_s=1,
)

print("Model cache ready.")
PY

cat <<EOF

Installer finished.
To launch the demo server with the warmed cache run:
  docker run -p ${PORT}:8000 -v ${CACHE_VOLUME}:/root/.cache/huggingface ${IMAGE_NAME}
EOF
