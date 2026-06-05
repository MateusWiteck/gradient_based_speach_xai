from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def plot_waveform(waveform: torch.Tensor, sample_rate: int, ax=None, title: str | None = None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))
    samples = waveform.detach().cpu().squeeze().numpy()
    times = np.arange(samples.shape[-1]) / sample_rate
    ax.plot(times, samples, linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    if title:
        ax.set_title(title)
    return ax


def shade_intervals(ax, intervals_table: pd.DataFrame, color: str = "tab:red", alpha: float = 0.25):
    for row in intervals_table.itertuples(index=False):
        ax.axvspan(float(row.start), float(row.end), color=color, alpha=alpha)
    return ax

