from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


SelectionMode = Literal["top", "bottom"]


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    score: float | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return scores
    min_score = np.nanmin(scores)
    max_score = np.nanmax(scores)
    if np.isclose(max_score, min_score):
        return np.zeros_like(scores)
    return (scores - min_score) / (max_score - min_score)


def token_relevance_to_intervals(
    audio_id: str,
    token_relevance: np.ndarray,
    audio_duration: float,
) -> pd.DataFrame:
    """Map token relevance values to fixed token intervals over the full audio duration."""
    token_relevance = normalize_scores(np.asarray(token_relevance, dtype=float))
    token_count = len(token_relevance)
    if token_count == 0:
        return pd.DataFrame(columns=["audio_id", "start", "end", "score"])

    edges = np.linspace(0.0, audio_duration, token_count + 1)
    return pd.DataFrame(
        {
            "audio_id": audio_id,
            "start": edges[:-1],
            "end": edges[1:],
            "score": token_relevance,
        }
    )


def select_intervals_by_duration(
    intervals_table: pd.DataFrame,
    duration_seconds: float,
    mode: SelectionMode = "top",
) -> pd.DataFrame:
    """Select ranked intervals until the requested duration in seconds is reached.

    This is the main helper for the corrected evaluation: SpeechXAI top-k words
    determine a duration X, then LeGrad and random baselines receive the same X.
    """
    if intervals_table.empty:
        return intervals_table.copy()

    ascending = mode == "bottom"
    ranked = intervals_table.sort_values("score", ascending=ascending).copy()
    selected_rows = []
    used_seconds = 0.0

    for row in ranked.itertuples(index=False):
        start = float(row.start)
        end = float(row.end)
        duration = max(0.0, end - start)
        if duration == 0.0:
            continue
        remaining = duration_seconds - used_seconds
        if remaining <= 1e-9:
            break
        clipped_end = start + min(duration, remaining)
        row_dict = row._asdict()
        row_dict["end"] = clipped_end
        selected_rows.append(row_dict)
        used_seconds += clipped_end - start

    return pd.DataFrame(selected_rows, columns=intervals_table.columns)


def select_top_k_intervals(
    intervals_table: pd.DataFrame,
    k: int,
    mode: SelectionMode = "top",
) -> pd.DataFrame:
    """Select top or bottom k scored intervals without changing their durations."""
    if intervals_table.empty:
        return intervals_table.copy()
    ascending = mode == "bottom"
    return intervals_table.sort_values("score", ascending=ascending).head(k).copy()


def random_intervals_by_duration(
    audio_id: str,
    audio_duration: float,
    duration_seconds: float,
    bin_seconds: float = 0.05,
    seed: int | None = None,
) -> pd.DataFrame:
    """Sample random fixed-size bins until a duration in seconds is reached."""
    rng = np.random.default_rng(seed)
    bin_count = max(1, int(np.ceil(audio_duration / bin_seconds)))
    starts = np.arange(bin_count) * bin_seconds
    ends = np.minimum(starts + bin_seconds, audio_duration)
    candidates = pd.DataFrame(
        {
            "audio_id": audio_id,
            "start": starts,
            "end": ends,
            "score": rng.random(bin_count),
        }
    )
    return select_intervals_by_duration(candidates, duration_seconds, mode="top")
