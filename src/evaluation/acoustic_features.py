"""Lightweight acoustic descriptors for relevance-selected audio regions.

The functions in this module intentionally avoid heavyweight dependencies such
as Praat/Parselmouth or ASR systems. Pitch is estimated with a simple
frame-wise autocorrelation method, and speaking rate is represented by an
energy-peak proxy rather than true words-per-second.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


EPS = 1e-10


@dataclass(frozen=True)
class AcousticConfig:
    """Configuration for frame-level acoustic feature extraction."""

    frame_ms: float = 40.0
    hop_ms: float = 10.0
    pitch_min_hz: float = 50.0
    pitch_max_hz: float = 500.0
    voicing_threshold: float = 0.3
    silence_db_below_reference: float = 35.0
    absolute_silence_floor: float = 1e-4
    min_pause_ms: float = 100.0
    min_peak_distance_ms: float = 120.0


def waveform_to_numpy(waveform: torch.Tensor | np.ndarray) -> np.ndarray:
    """Return a mono float numpy array from a torch or numpy waveform."""
    if isinstance(waveform, torch.Tensor):
        values = waveform.detach().cpu().float().numpy()
    else:
        values = np.asarray(waveform, dtype=np.float32)
    values = np.squeeze(values)
    if values.ndim != 1:
        raise ValueError(f"Expected a mono waveform, got shape {values.shape}.")
    return values.astype(np.float64, copy=False)


def waveform_rms(waveform: torch.Tensor | np.ndarray) -> float:
    """Return root-mean-square amplitude for the waveform."""
    values = waveform_to_numpy(waveform)
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(values))))


def _frame_signal(values: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Slice a 1-D signal into overlapping frames, padding short signals."""
    if values.size == 0:
        return np.zeros((0, frame_length), dtype=np.float64)
    if values.size < frame_length:
        values = np.pad(values, (0, frame_length - values.size))

    frame_count = 1 + int(np.floor((values.size - frame_length) / hop_length))
    if frame_count <= 0:
        return np.zeros((0, frame_length), dtype=np.float64)

    shape = (frame_count, frame_length)
    strides = (values.strides[0] * hop_length, values.strides[0])
    return np.lib.stride_tricks.as_strided(values, shape=shape, strides=strides).copy()


def _longest_true_run_seconds(mask: np.ndarray, hop_seconds: float) -> float:
    """Return duration of the longest consecutive True run."""
    longest = 0
    current = 0
    for value in mask:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return float(longest * hop_seconds)


def _count_true_runs(mask: np.ndarray, hop_seconds: float, min_duration_seconds: float) -> int:
    """Count consecutive True runs whose duration reaches the threshold."""
    count = 0
    current = 0
    for value in mask:
        if bool(value):
            current += 1
        elif current:
            if current * hop_seconds >= min_duration_seconds:
                count += 1
            current = 0
    if current * hop_seconds >= min_duration_seconds:
        count += 1
    return count


def _count_energy_peaks(
    rms_values: np.ndarray,
    *,
    threshold: float,
    min_distance_frames: int,
) -> int:
    """Count separated local RMS peaks above a threshold."""
    if rms_values.size < 3:
        return int(np.any(rms_values > threshold))

    candidate_indices = np.flatnonzero(
        (rms_values[1:-1] >= rms_values[:-2])
        & (rms_values[1:-1] > rms_values[2:])
        & (rms_values[1:-1] > threshold)
    ) + 1
    if candidate_indices.size == 0:
        return 0

    # Greedy non-maximum suppression by descending RMS value. This makes the
    # proxy less sensitive to jitter in the energy envelope.
    sorted_indices = candidate_indices[np.argsort(rms_values[candidate_indices])[::-1]]
    kept: list[int] = []
    for index in sorted_indices:
        if all(abs(int(index) - kept_index) >= min_distance_frames for kept_index in kept):
            kept.append(int(index))
    return len(kept)


def _estimate_pitch_autocorrelation(
    frames: np.ndarray,
    rms_values: np.ndarray,
    sample_rate: int,
    config: AcousticConfig,
    silence_threshold: float,
) -> np.ndarray:
    """Estimate one F0 value per frame with normalized autocorrelation."""
    if frames.size == 0:
        return np.array([], dtype=np.float64)

    min_lag = max(1, int(sample_rate / config.pitch_max_hz))
    max_lag = min(frames.shape[1] - 1, int(sample_rate / config.pitch_min_hz))
    if min_lag >= max_lag:
        return np.full(frames.shape[0], np.nan, dtype=np.float64)

    window = np.hanning(frames.shape[1])
    f0_values = np.full(frames.shape[0], np.nan, dtype=np.float64)
    for frame_index, frame in enumerate(frames):
        if rms_values[frame_index] <= silence_threshold:
            continue

        centered = frame - np.mean(frame)
        windowed = centered * window
        autocorr = np.correlate(windowed, windowed, mode="full")[frames.shape[1] - 1 :]
        if autocorr[0] <= EPS:
            continue

        search = autocorr[min_lag : max_lag + 1] / autocorr[0]
        best_offset = int(np.argmax(search))
        best_score = float(search[best_offset])
        if best_score < config.voicing_threshold:
            continue

        best_lag = min_lag + best_offset
        f0_values[frame_index] = sample_rate / best_lag

    return f0_values


def extract_acoustic_features(
    waveform: torch.Tensor | np.ndarray,
    sample_rate: int,
    *,
    reference_rms: float | None = None,
    config: AcousticConfig | None = None,
) -> dict[str, float | int]:
    """Extract pitch, energy, pause, and speaking-rate proxy features.

    ``speaking_rate_proxy_peaks_per_sec`` is not a true lexical speaking rate.
    It counts separated energy-envelope peaks per second, which roughly tracks
    syllabic/accent activity when transcripts or forced alignment are absent.
    """
    config = config or AcousticConfig()
    values = waveform_to_numpy(waveform)
    duration_seconds = float(values.size / sample_rate) if sample_rate > 0 else 0.0
    if values.size == 0 or sample_rate <= 0:
        return _empty_feature_row(duration_seconds)

    frame_length = max(1, int(round(config.frame_ms * sample_rate / 1000)))
    hop_length = max(1, int(round(config.hop_ms * sample_rate / 1000)))
    hop_seconds = hop_length / sample_rate

    frames = _frame_signal(values, frame_length, hop_length)
    if frames.size == 0:
        return _empty_feature_row(duration_seconds)

    rms_values = np.sqrt(np.mean(np.square(frames), axis=1))
    global_rms = float(np.sqrt(np.mean(np.square(values))))
    if reference_rms is None or reference_rms <= 0:
        reference_rms = global_rms
    silence_threshold = max(
        float(config.absolute_silence_floor),
        float(reference_rms) * 10 ** (-config.silence_db_below_reference / 20),
    )

    silent_mask = rms_values <= silence_threshold
    min_pause_seconds = config.min_pause_ms / 1000
    pause_count = _count_true_runs(silent_mask, hop_seconds, min_pause_seconds)
    longest_pause_seconds = _longest_true_run_seconds(silent_mask, hop_seconds)

    f0_values = _estimate_pitch_autocorrelation(
        frames,
        rms_values,
        sample_rate,
        config,
        silence_threshold,
    )
    voiced_mask = np.isfinite(f0_values)
    voiced_f0 = f0_values[voiced_mask]

    # Use a relative threshold above silence to avoid counting tiny noise bumps.
    energy_peak_threshold = max(silence_threshold * 2.0, float(np.percentile(rms_values, 60)))
    min_distance_frames = max(
        1,
        int(round((config.min_peak_distance_ms / 1000) / hop_seconds)),
    )
    energy_peak_count = _count_energy_peaks(
        rms_values,
        threshold=energy_peak_threshold,
        min_distance_frames=min_distance_frames,
    )

    return {
        "duration_seconds": duration_seconds,
        "frame_count": int(rms_values.size),
        "rms_mean": float(np.mean(rms_values)),
        "rms_std": float(np.std(rms_values)),
        "rms_max": float(np.max(rms_values)),
        "rms_dbfs_mean": float(20 * np.log10(np.mean(rms_values) + EPS)),
        "silence_threshold_rms": float(silence_threshold),
        "silent_frame_count": int(np.sum(silent_mask)),
        "silence_fraction": float(np.mean(silent_mask)),
        "pause_count": int(pause_count),
        "longest_pause_seconds": float(longest_pause_seconds),
        "voiced_frame_count": int(np.sum(voiced_mask)),
        "voiced_fraction": float(np.mean(voiced_mask)),
        "f0_mean_hz": _nan_safe_stat(voiced_f0, np.mean),
        "f0_median_hz": _nan_safe_stat(voiced_f0, np.median),
        "f0_std_hz": _nan_safe_stat(voiced_f0, np.std),
        "f0_min_hz": _nan_safe_stat(voiced_f0, np.min),
        "f0_max_hz": _nan_safe_stat(voiced_f0, np.max),
        "energy_peak_count": int(energy_peak_count),
        "speaking_rate_proxy_peaks_per_sec": float(
            energy_peak_count / duration_seconds if duration_seconds > 0 else 0.0
        ),
    }


def aggregate_acoustic_feature_rows(rows: list[dict]) -> dict[str, float | int]:
    """Aggregate contiguous segment feature rows into one selection-level row."""
    if not rows:
        return _empty_feature_row(0.0)

    total_duration = float(sum(float(row["duration_seconds"]) for row in rows))
    total_frames = int(sum(int(row["frame_count"]) for row in rows))
    total_silent_frames = int(sum(int(row["silent_frame_count"]) for row in rows))
    total_voiced_frames = int(sum(int(row["voiced_frame_count"]) for row in rows))
    total_energy_peaks = int(sum(int(row["energy_peak_count"]) for row in rows))

    def duration_weighted(column: str) -> float:
        if total_duration <= 0:
            return 0.0
        weighted_sum = 0.0
        weight_sum = 0.0
        for row in rows:
            value = float(row[column])
            if np.isfinite(value):
                weight = float(row["duration_seconds"])
                weighted_sum += value * weight
                weight_sum += weight
        return float(weighted_sum / weight_sum) if weight_sum > 0 else float("nan")

    def voiced_weighted(column: str) -> float:
        weighted_sum = 0.0
        weight_sum = 0
        for row in rows:
            value = float(row[column])
            if np.isfinite(value):
                weight = int(row["voiced_frame_count"])
                weighted_sum += value * weight
                weight_sum += weight
        return float(weighted_sum / weight_sum) if weight_sum > 0 else float("nan")

    return {
        "duration_seconds": total_duration,
        "frame_count": total_frames,
        "rms_mean": duration_weighted("rms_mean"),
        "rms_std": duration_weighted("rms_std"),
        "rms_max": max(float(row["rms_max"]) for row in rows),
        "rms_dbfs_mean": duration_weighted("rms_dbfs_mean"),
        "silence_threshold_rms": duration_weighted("silence_threshold_rms"),
        "silent_frame_count": total_silent_frames,
        "silence_fraction": float(total_silent_frames / total_frames) if total_frames else 0.0,
        "pause_count": int(sum(int(row["pause_count"]) for row in rows)),
        "longest_pause_seconds": max(float(row["longest_pause_seconds"]) for row in rows),
        "voiced_frame_count": total_voiced_frames,
        "voiced_fraction": float(total_voiced_frames / total_frames) if total_frames else 0.0,
        "f0_mean_hz": voiced_weighted("f0_mean_hz"),
        "f0_median_hz": voiced_weighted("f0_median_hz"),
        "f0_std_hz": voiced_weighted("f0_std_hz"),
        "f0_min_hz": min(
            (float(row["f0_min_hz"]) for row in rows if np.isfinite(float(row["f0_min_hz"]))),
            default=float("nan"),
        ),
        "f0_max_hz": max(
            (float(row["f0_max_hz"]) for row in rows if np.isfinite(float(row["f0_max_hz"]))),
            default=float("nan"),
        ),
        "energy_peak_count": total_energy_peaks,
        "speaking_rate_proxy_peaks_per_sec": float(
            total_energy_peaks / total_duration if total_duration > 0 else 0.0
        ),
    }


def _nan_safe_stat(values: np.ndarray, statistic) -> float:
    if values.size == 0:
        return float("nan")
    return float(statistic(values))


def _empty_feature_row(duration_seconds: float) -> dict[str, float | int]:
    return {
        "duration_seconds": float(duration_seconds),
        "frame_count": 0,
        "rms_mean": 0.0,
        "rms_std": 0.0,
        "rms_max": 0.0,
        "rms_dbfs_mean": float("nan"),
        "silence_threshold_rms": 0.0,
        "silent_frame_count": 0,
        "silence_fraction": 0.0,
        "pause_count": 0,
        "longest_pause_seconds": 0.0,
        "voiced_frame_count": 0,
        "voiced_fraction": 0.0,
        "f0_mean_hz": float("nan"),
        "f0_median_hz": float("nan"),
        "f0_std_hz": float("nan"),
        "f0_min_hz": float("nan"),
        "f0_max_hz": float("nan"),
        "energy_peak_count": 0,
        "speaking_rate_proxy_peaks_per_sec": 0.0,
    }
