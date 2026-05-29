"""
TTS learning tutorial:

1. Use SpeechT5 as a small, easy-to-inspect model.
2. Use VoxCPM2 as a larger modern multilingual TTS model.

The script runs SpeechT5 by default. Set RUN_VOXCPM2=1 to run VoxCPM2.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import soundfile as sf
import torch


OUTPUT_DIR = Path("speech/outputs")
SMALL_TEXT = "Speech synthesis turns written text into a waveform that we can play."
VOXCPM_TEXT = "(年轻女性，温柔自然，语速适中) 你好，这是 VoxCPM2 的语音合成示例。"


def pick_torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def save_wav(path: Path, waveform, sample_rate: int, elapsed_seconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, waveform, sample_rate)
    duration_seconds = len(waveform) / sample_rate
    rtf = elapsed_seconds / duration_seconds if duration_seconds else float("inf")
    print(f"saved: {path}")
    print(f"sample_rate={sample_rate}, duration={duration_seconds:.2f}s, elapsed={elapsed_seconds:.2f}s, rtf={rtf:.2f}")


def run_speecht5() -> None:
    from datasets import load_dataset
    from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

    device = pick_torch_device()
    print(f"SpeechT5 device: {device}")

    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(device)

    inputs = processor(text=SMALL_TEXT, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    print(f"input_ids shape: {tuple(input_ids.shape)}")
    print(f"speaker_embeddings shape: {tuple(speaker_embeddings.shape)}")
    print(f"acoustic model parameters: {count_parameters(model) / 1e6:.1f}M")
    print(f"vocoder parameters: {count_parameters(vocoder) / 1e6:.1f}M")

    start = time.perf_counter()
    with torch.inference_mode():
        speech = model.generate_speech(input_ids, speaker_embeddings, vocoder=vocoder)
    elapsed = time.perf_counter() - start

    save_wav(OUTPUT_DIR / "speecht5_demo.wav", speech.detach().cpu().numpy(), 16_000, elapsed)


def run_voxcpm2() -> None:
    from voxcpm import VoxCPM

    print("Loading VoxCPM2. First run downloads large model weights.")
    model = VoxCPM.from_pretrained(
        "openbmb/VoxCPM2",
        device="auto",
        load_denoiser=False,
        optimize=False,
    )

    start = time.perf_counter()
    wav = model.generate(
        text=VOXCPM_TEXT,
        cfg_value=2.0,
        inference_timesteps=10,
        normalize=True,
    )
    elapsed = time.perf_counter() - start

    save_wav(OUTPUT_DIR / "voxcpm2_voice_design.wav", wav, model.tts_model.sample_rate, elapsed)


if __name__ == "__main__":
    run_speecht5()

    if os.getenv("RUN_VOXCPM2") == "1":
        run_voxcpm2()
    else:
        print("Skip VoxCPM2. Set RUN_VOXCPM2=1 to run the larger model section.")
