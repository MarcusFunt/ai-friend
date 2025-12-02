"""
Live ASR demo for the NVIDIA parakeet-tdt-0.6b-v3 model with a Dear PyGui interface.

Controls:
- Use "Start Listening" to begin streaming microphone audio to the ASR model.
- Use "Stop Listening" to halt recording and model inference.

The interface shows the most recent transcript chunk and a status indicator. Audio is
captured from the default microphone at 16 kHz; adjust the device settings in
`sounddevice` if needed.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import sounddevice as sd
import torch
from dearpygui import dearpygui as dpg
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
SAMPLE_RATE = 16_000


@dataclass
class LiveASR:
    """Simple live ASR helper using a Hugging Face pipeline."""

    window_seconds: float = 6.0
    stride_seconds: float = 3.0
    max_buffer_seconds: float = 12.0
    pipeline: Optional[object] = None
    stream: Optional[sd.InputStream] = None
    transcript: str = ""
    _audio_queue: queue.Queue[np.ndarray] = field(default_factory=queue.Queue, init=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _worker: Optional[threading.Thread] = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.pipeline is None:
            self.pipeline = self._build_pipeline()

    def _build_pipeline(self):
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device_index = 0 if torch.cuda.is_available() else -1
        model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID, torch_dtype=torch_dtype)
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        return pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device_index,
        )

    def start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._stop_event.clear()
        self._audio_queue = queue.Queue()
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=self._audio_callback,
            dtype="float32",
        )
        self.stream.start()
        self._worker = threading.Thread(target=self._run_inference, daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None

    def _audio_callback(self, indata, frames, _time_info, _status) -> None:
        self._audio_queue.put(indata.copy().flatten())

    def _run_inference(self) -> None:
        buffer: np.ndarray = np.zeros(0, dtype=np.float32)
        last_run = 0.0
        max_samples = int(SAMPLE_RATE * self.max_buffer_seconds)
        window_samples = int(SAMPLE_RATE * self.window_seconds)

        while not self._stop_event.is_set() or not self._audio_queue.empty():
            try:
                chunk = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                chunk = None

            if chunk is not None:
                buffer = np.concatenate([buffer, chunk])
                if buffer.size > max_samples:
                    buffer = buffer[-max_samples:]

            now = time.time()
            if now - last_run < self.stride_seconds:
                continue
            if buffer.size < SAMPLE_RATE:
                continue

            last_run = now
            segment = buffer[-window_samples:]
            result = self.pipeline(segment, chunk_length_s=self.window_seconds)
            text = result.get("text", "").strip()
            self.transcript = text
            dpg.set_value("transcript_text", text)
            dpg.set_value("status_text", f"Status: Listening (updated {time.strftime('%H:%M:%S')})")

        dpg.set_value("status_text", "Status: Idle")


class ParakeetApp:
    def __init__(self) -> None:
        self.asr = LiveASR()
        self._build_ui()

    def _build_ui(self) -> None:
        dpg.create_context()
        with dpg.window(label="NVIDIA Parakeet Live ASR", width=720, height=480):
            dpg.add_text(f"Model: {MODEL_ID}")
            dpg.add_text(
                "Click 'Start Listening' and speak into your default microphone."
            )
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_button(label="Start Listening", callback=self.start_listening)
                dpg.add_button(label="Stop Listening", callback=self.stop_listening)
            dpg.add_text("Status: Idle", tag="status_text")
            dpg.add_spacing(count=1)
            dpg.add_text("Transcript (most recent window):")
            dpg.add_input_text(
                tag="transcript_text",
                multiline=True,
                readonly=True,
                height=320,
                width=680,
                default_value="",
            )

        dpg.create_viewport(title="Parakeet Live ASR", width=760, height=560)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_exit_callback(self.stop_listening)

    def start(self) -> None:
        dpg.start_dearpygui()
        dpg.destroy_context()

    def start_listening(self, _sender=None, _app_data=None, _user_data=None) -> None:
        dpg.set_value("status_text", "Status: Loading model...")
        if not (self.asr._worker and self.asr._worker.is_alive()):
            self.asr.start()
            dpg.set_value("status_text", "Status: Listening")

    def stop_listening(self, _sender=None, _app_data=None, _user_data=None) -> None:
        self.asr.stop()
        dpg.set_value("status_text", "Status: Idle")


def main() -> None:
    app = ParakeetApp()
    app.start()


if __name__ == "__main__":
    main()
