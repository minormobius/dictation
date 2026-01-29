"""
Voice-to-Text Dictation Tool
Press Right Ctrl to start recording, press again to stop and transcribe.
Result is copied to clipboard and printed to stdout.
"""

import sys
import io
import time
import argparse
import threading
import queue
import numpy as np
import sounddevice as sd
import keyboard
import pyperclip
from faster_whisper import WhisperModel

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SAMPLE_RATE = 16000
CHANNELS = 1


def beep(freq=800, duration=0.1):
    """Play a short beep for audio feedback."""
    try:
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), dtype=np.float32)
        tone = 0.3 * np.sin(2 * np.pi * freq * t)
        sd.play(tone, samplerate=SAMPLE_RATE)
        sd.wait()
    except Exception:
        pass


def record_while_held(key="right ctrl"):
    """Record audio while key is held down. Returns numpy array."""
    audio_queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"  [audio status: {status}]", flush=True)
        audio_queue.put(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        callback=callback,
        blocksize=1024,
    )

    stream.start()
    print("  * REC", flush=True)

    # Record while key is held
    while keyboard.is_pressed(key):
        time.sleep(0.02)

    stream.stop()
    stream.close()

    # Collect all audio chunks
    chunks = []
    while not audio_queue.empty():
        chunks.append(audio_queue.get())

    if not chunks:
        print("  * No audio chunks captured.", flush=True)
        return None

    audio = np.concatenate(chunks, axis=0).flatten()
    duration = len(audio) / SAMPLE_RATE
    peak = np.max(np.abs(audio))
    print(f"  * Stopped. {duration:.1f}s, peak: {peak:.4f}", flush=True)

    return audio


def transcribe(model, audio):
    """Transcribe audio array using faster-whisper."""
    segments, info = model.transcribe(
        audio,
        beam_size=1,
        language="en",
        vad_filter=False,
    )

    text_parts = []
    for segment in segments:
        text_parts.append(segment.text.strip())

    return " ".join(text_parts)


def main():
    parser = argparse.ArgumentParser(description="Voice-to-text dictation tool")
    parser.add_argument(
        "--model",
        default="base.en",
        choices=["tiny.en", "base.en", "small.en", "medium.en"],
        help="Whisper model size (default: base.en)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cuda", "cpu"],
        help="Compute device (default: cpu)",
    )
    args = parser.parse_args()

    print(f"Loading model '{args.model}' on {args.device}...", flush=True)
    model = None
    if args.device == "cuda":
        for ct in ("float16", "int8"):
            try:
                model = WhisperModel(args.model, device="cuda", compute_type=ct)
                print(f"  Loaded on CUDA with {ct}.", flush=True)
                break
            except Exception as e:
                print(f"  CUDA {ct} failed ({e}), trying next...", flush=True)
    if model is None:
        print("  Loading on CPU with int8...", flush=True)
        model = WhisperModel(args.model, device="cpu", compute_type="int8")

    print(f"Model loaded. Ready!", flush=True)

    # Show default input device
    try:
        dev = sd.query_devices(kind="input")
        print(f"  Mic:    {dev['name']} ({dev['max_input_channels']}ch)", flush=True)
    except Exception:
        print("  Mic:    (could not detect default input)", flush=True)

    print(f"  Hotkey: Hold Right Ctrl = record, release = transcribe", flush=True)
    print(f"  Quit:   Ctrl+C", flush=True)
    print("", flush=True)

    try:
        while True:
            print("Waiting... hold Right Ctrl to record.", flush=True)
            keyboard.wait("right ctrl")

            audio = record_while_held("right ctrl")

            if audio is None or len(audio) < SAMPLE_RATE * 0.3:
                print("  (too short, skipped)\n", flush=True)
                continue

            print("  Transcribing...", flush=True)
            text = transcribe(model, audio)

            if text.strip():
                pyperclip.copy(text)
                print(f"  >> {text}\n", flush=True)
            else:
                print("  (no speech detected)\n", flush=True)

    except KeyboardInterrupt:
        print("\nExiting.", flush=True)


if __name__ == "__main__":
    main()
