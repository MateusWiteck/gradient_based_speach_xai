from __future__ import annotations

import pandas as pd
import torch


def silence_intervals(
    waveform: torch.Tensor,
    sample_rate: int,
    intervals_table: pd.DataFrame,
) -> torch.Tensor:
    """Replace selected audio regions with silence while preserving waveform length."""
    masked_waveform = waveform.clone()
    total_samples = masked_waveform.shape[-1]

    for row in intervals_table.itertuples(index=False):
        start_sample = max(0, int(round(float(row.start) * sample_rate)))
        end_sample = min(total_samples, int(round(float(row.end) * sample_rate)))
        if end_sample > start_sample:
            masked_waveform[..., start_sample:end_sample] = 0.0

    return masked_waveform


def total_interval_duration(intervals_table: pd.DataFrame) -> float:
    if intervals_table.empty:
        return 0.0
    durations = intervals_table["end"].astype(float) - intervals_table["start"].astype(float)
    return float(durations.clip(lower=0).sum())

