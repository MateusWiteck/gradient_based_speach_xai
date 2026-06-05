from __future__ import annotations

from pathlib import Path

import pandas as pd
import soundfile as sf
import torch


def audio_id_from_path(audio_path: str | Path) -> str:
    return Path(audio_path).stem


def load_waveform(
    audio_path: str | Path,
    target_sample_rate: int = 16000,
    mono: bool = True,
) -> tuple[torch.Tensor, int]:
    """Load audio as a tensor shaped [channels, samples]."""
    import torchaudio

    try:
        waveform, sample_rate = torchaudio.load(str(audio_path))
    except ImportError:
        audio_samples, sample_rate = sf.read(str(audio_path), always_2d=True)
        waveform = torch.from_numpy(audio_samples.T).float()
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


def discover_audio_metadata(iemocap_root: str | Path, limit: int | None = None) -> pd.DataFrame:
    """Build a simple metadata table from local audio files."""
    root = Path(iemocap_root)
    audio_paths = []
    for extension in ("*.wav", "*.flac", "*.mp3"):
        audio_paths.extend(root.rglob(extension))

    audio_paths = sorted(audio_paths)
    if limit is not None:
        audio_paths = audio_paths[:limit]

    return pd.DataFrame(
        {
            "audio_id": [path.stem for path in audio_paths],
            "audio_path": [str(path) for path in audio_paths],
            "true_label": [None for _ in audio_paths],
        }
    )


def load_metadata_or_discover(
    metadata_csv: str | Path,
    iemocap_root: str | Path,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load configured metadata or discover audio files from the dataset root."""
    metadata_path = Path(metadata_csv)
    if metadata_path.exists():
        metadata_table = pd.read_csv(metadata_path)
        if limit is not None:
            metadata_table = metadata_table.head(limit)
        return metadata_table

    return discover_audio_metadata(iemocap_root, limit=limit)
