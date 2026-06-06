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
    """Return the duration covered by the union of all intervals."""
    if intervals_table.empty:
        return 0.0

    intervals = sorted(
        (
            float(row.start),
            float(row.end),
        )
        for row in intervals_table.itertuples(index=False)
        if float(row.end) > float(row.start)
    )
    if not intervals:
        return 0.0

    covered_seconds = 0.0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        covered_seconds += current_end - current_start
        current_start, current_end = start, end

    return covered_seconds + current_end - current_start
