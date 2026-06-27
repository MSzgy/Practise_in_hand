"""
TTS learning tutorial:

1. Use a ModelScope Sambert + HiFiGAN model as a small runnable TTS model.
2. Use VoxCPM2 as a larger modern multilingual TTS model.

The script runs the small ModelScope TTS model by default.
Set RUN_VOXCPM2=1 to run VoxCPM2.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

import soundfile as sf
import torch


OUTPUT_DIR = Path("speech/outputs")
SMALL_TTS_MODEL_ID = os.getenv("SMALL_TTS_MODEL_ID", "damo/speech_sambert-hifigan_tts_zh-cn_16k")
VOXCPM2_MODEL_ID = os.getenv("VOXCPM2_MODEL_ID", "OpenBMB/VoxCPM2")
SMALL_TEXT = "语音合成会把文本转换成可以播放的波形。"
VOXCPM_TEXT = "(年轻女性，温柔自然，语速适中) 你好，这是 VoxCPM2 的语音合成示例。"
SAMBERTHIFIGAN_MODEL_KEYWORD = "sambert-hifigan"


def sambert_hifigan_python_issue(model_id: str) -> str | None:
    if SAMBERTHIFIGAN_MODEL_KEYWORD not in model_id.lower():
        return None
    if sys.version_info < (3, 11):
        return None
    return (
        f"{model_id} uses ModelScope SambertHifigan, which requires Python <= 3.10. "
        f"Current Python is {sys.version_info.major}.{sys.version_info.minor}. "
        "Run the small-model section in a Python 3.10 environment, or set "
        "RUN_SMALL_TTS=0 and use VoxCPM2 / another Python 3.11-compatible TTS model."
    )


def pick_torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def download_modelscope_model(model_id: str) -> str:
    from modelscope import snapshot_download

    print(f"Loading ModelScope model: {model_id}")
    model_dir = snapshot_download(model_id)
    print(f"ModelScope cache: {model_dir}")
    return model_dir


def report_wav(path: Path, elapsed_seconds: float) -> dict[str, float | str]:
    info = sf.info(path)
    duration_seconds = info.frames / info.samplerate if info.samplerate else 0.0
    rtf = elapsed_seconds / duration_seconds if duration_seconds else float("inf")
    print(f"saved: {path}")
    print(
        f"sample_rate={info.samplerate}, duration={duration_seconds:.2f}s, "
        f"elapsed={elapsed_seconds:.2f}s, rtf={rtf:.2f}"
    )
    return {
        "path": str(path),
        "sample_rate": info.samplerate,
        "duration_seconds": duration_seconds,
        "elapsed_seconds": elapsed_seconds,
        "rtf": rtf,
    }


def save_wav(path: Path, waveform, sample_rate: int, elapsed_seconds: float) -> dict[str, float | str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, waveform, sample_rate)
    return report_wav(path, elapsed_seconds)


def save_wav_bytes(path: Path, wav_bytes: bytes | bytearray, elapsed_seconds: float) -> dict[str, float | str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(wav_bytes))
    return report_wav(path, elapsed_seconds)


def save_modelscope_tts_output(path: Path, output, elapsed_seconds: float) -> dict[str, float | str]:
    from modelscope.outputs import OutputKeys

    wav = output
    if isinstance(output, dict):
        wav = output.get(OutputKeys.OUTPUT_WAV) or output.get("output_wav") or output.get("wav")

    if wav is None:
        details = output.keys() if isinstance(output, dict) else type(output)
        raise ValueError(f"ModelScope TTS output does not contain WAV data: {details}")

    if isinstance(wav, (bytes, bytearray)):
        return save_wav_bytes(path, wav, elapsed_seconds)

    if isinstance(wav, (str, os.PathLike)):
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(wav, path)
        return report_wav(path, elapsed_seconds)

    sample_rate = int(output.get("sample_rate", 16_000)) if isinstance(output, dict) else 16_000
    return save_wav(path, wav, sample_rate, elapsed_seconds)


def run_small_modelscope_tts() -> None:
    compatibility_issue = sambert_hifigan_python_issue(SMALL_TTS_MODEL_ID)
    if compatibility_issue:
        print(f"Skip small ModelScope TTS: {compatibility_issue}")
        return

    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    print(f"Small TTS device hint: {pick_torch_device()}")
    model_dir = download_modelscope_model(SMALL_TTS_MODEL_ID)
    tts = pipeline(task=Tasks.text_to_speech, model=model_dir)

    print(f"text: {SMALL_TEXT}")

    start = time.perf_counter()
    output = tts(input=SMALL_TEXT)
    elapsed = time.perf_counter() - start

    save_modelscope_tts_output(OUTPUT_DIR / "modelscope_sambert_demo.wav", output, elapsed)


def run_voxcpm2() -> None:
    from voxcpm import VoxCPM

    model_dir = download_modelscope_model(VOXCPM2_MODEL_ID)
    print("Loading VoxCPM2 from local ModelScope cache.")
    model = VoxCPM.from_pretrained(
        model_dir,
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
    if os.getenv("RUN_SMALL_TTS", "1") == "1":
        run_small_modelscope_tts()
    else:
        print("Skip small ModelScope TTS. RUN_SMALL_TTS is not 1.")

    if os.getenv("RUN_VOXCPM2") == "1":
        run_voxcpm2()
    else:
        print("Skip VoxCPM2. Set RUN_VOXCPM2=1 to run the larger model section.")
