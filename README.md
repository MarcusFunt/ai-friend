# ai-friend

This repository contains a small live ASR demo for the [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) model using a Dear PyGui interface.

## Setup
Install dependencies (they will download the model on first run):

```bash
pip install -r requirements.txt
```

## Running the demo
Launch the Dear PyGui app and start transcribing from your default microphone:

```bash
python parakeet_demo.py
```

* Click **Start Listening** to begin capturing microphone audio and streaming it to the ASR model.
* Click **Stop Listening** to halt capture and inference.
* The transcript panel shows the most recent inference window (6 seconds by default) and updates while listening.

The script uses CPU by default, but will automatically use GPU if CUDA is available. Adjust the microphone device in `parakeet_demo.py` if your default input is not correct.
