from __future__ import annotations

from pathlib import Path

import torch
import torchaudio


def audio_id_from_path(audio_path: str | Path) -> str:
    return Path(audio_path).stem


def load_waveform(
    audio_path: str | Path,
    target_sample_rate: int = 16000,
    mono: bool = True,
) -> tuple[torch.Tensor, int]:
    """Load audio as a tensor shaped [channels, samples]."""
    waveform, sample_rate = torchaudio.load(str(audio_path))
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        waveform = resampler(waveform)
        sample_rate = target_sample_rate
    return waveform, sample_rate


def duration_seconds(waveform: torch.Tensor, sample_rate: int) -> float:
    return waveform.shape[-1] / sample_rate


def waveform_to_batch(waveform: torch.Tensor) -> torch.Tensor:
    """Convert [channels, samples] waveform to SpeechBrain-friendly [batch, samples]."""
    if waveform.ndim == 1:
        return waveform.unsqueeze(0)
    if waveform.shape[0] == 1:
        return waveform
    return waveform.mean(dim=0, keepdim=True)

