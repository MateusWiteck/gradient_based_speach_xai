"""Metrics for temporal explanation concentration and class-specificity."""

from __future__ import annotations

import numpy as np
import pandas as pd


SPARSITY_METRIC_COLUMNS = (
    "normalized_entropy",
    "effective_tokens",
    "gini",
    "top_5_percent_mass",
    "top_10_percent_mass",
)
CLASS_SPECIFICITY_METRIC_COLUMNS = (
    "pearson_correlation",
    "spearman_correlation",
)


def _as_1d_numpy(values) -> np.ndarray:
    """Convert tensors, lists, or arrays to a finite one-dimensional array."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=float).reshape(-1)
    return np.where(np.isfinite(array), array, 0.0)


def _normalize_nonnegative(values, eps: float = 1e-12) -> np.ndarray:
    """Return a non-negative distribution, with uniform fallback if empty."""
    array = np.clip(_as_1d_numpy(values), a_min=0.0, a_max=None)
    if array.size == 0:
        raise ValueError("Scores must contain at least one token.")
    total = float(array.sum())
    if total <= eps:
        return np.full(array.shape, 1.0 / array.size, dtype=float)
    return array / total


def concentration_metrics(scores) -> dict[str, float]:
    """Measure temporal concentration for one token-relevance distribution.

    Lower normalized entropy/effective tokens and higher Gini/top-k mass mean
    the explanation is more concentrated in a small set of time tokens.
    """
    values = _normalize_nonnegative(scores)
    token_count = values.size
    positive_values = values[values > 0]
    entropy = float(-(positive_values * np.log(positive_values)).sum())
    top_five_count = max(1, int(np.ceil(0.05 * token_count)))
    top_ten_count = max(1, int(np.ceil(0.10 * token_count)))
    normalized_entropy = entropy / float(np.log(token_count)) if token_count > 1 else 0.0
    gini = float(np.abs(values[:, None] - values[None, :]).sum() / (2 * token_count))
    return {
        "token_count": int(token_count),
        "normalized_entropy": float(normalized_entropy),
        "effective_tokens": float(np.exp(entropy)),
        "gini": gini,
        "top_5_percent_mass": float(np.sort(values)[-top_five_count:].sum()),
        "top_10_percent_mass": float(np.sort(values)[-top_ten_count:].sum()),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks, matching Spearman tie handling."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=float)

    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = average_rank
        start = end

    return ranks


def safe_correlation(first, second, *, method: str = "pearson") -> float:
    """Compute Pearson/Spearman correlation, returning NaN if undefined."""
    first_values = _as_1d_numpy(first)
    second_values = _as_1d_numpy(second)
    if first_values.shape != second_values.shape:
        raise ValueError(
            "Correlation inputs must have the same shape, got "
            f"{first_values.shape} and {second_values.shape}."
        )
    finite_mask = np.isfinite(first_values) & np.isfinite(second_values)
    first_values = first_values[finite_mask]
    second_values = second_values[finite_mask]
    if first_values.size < 2:
        return float("nan")
    if method == "spearman":
        first_values = _average_ranks(first_values)
        second_values = _average_ranks(second_values)
    elif method != "pearson":
        raise ValueError(f"Unsupported correlation method: {method}")

    if np.std(first_values) <= 0 or np.std(second_values) <= 0:
        return float("nan")
    return float(np.corrcoef(first_values, second_values)[0, 1])


def class_specificity_metrics(predicted_class_scores, runner_up_scores) -> dict[str, float]:
    """Compare predicted-class and runner-up relevance maps for one utterance.

    Lower correlations indicate that the method changes the temporal map more
    when the explained class changes, which is the desired class-specificity
    behavior for this diagnostic.
    """
    predicted_values = _normalize_nonnegative(predicted_class_scores)
    runner_values = _normalize_nonnegative(runner_up_scores)
    if predicted_values.shape != runner_values.shape:
        raise ValueError(
            "Predicted-class and runner-up maps must have the same token length, "
            f"got {predicted_values.shape} and {runner_values.shape}."
        )
    return {
        "token_count": int(predicted_values.size),
        "pearson_correlation": safe_correlation(
            predicted_values,
            runner_values,
            method="pearson",
        ),
        "spearman_correlation": safe_correlation(
            predicted_values,
            runner_values,
            method="spearman",
        ),
    }


def summarize_sparsity_by_method(sparsity_frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate concentration metrics into one row per explanation mode."""
    missing = sorted(set(SPARSITY_METRIC_COLUMNS + ("explanation_mode",)).difference(sparsity_frame.columns))
    if missing:
        raise ValueError(f"Sparsity frame is missing required columns: {', '.join(missing)}")

    audio_count_column = "audio_path" if "audio_path" in sparsity_frame.columns else "explanation_mode"
    summary = (
        sparsity_frame.groupby("explanation_mode", as_index=False)
        .agg(
            examples=(audio_count_column, "nunique"),
            records=("explanation_mode", "size"),
            mean_normalized_entropy=("normalized_entropy", "mean"),
            std_normalized_entropy=("normalized_entropy", "std"),
            mean_effective_tokens=("effective_tokens", "mean"),
            std_effective_tokens=("effective_tokens", "std"),
            mean_gini=("gini", "mean"),
            std_gini=("gini", "std"),
            mean_top_5_percent_mass=("top_5_percent_mass", "mean"),
            std_top_5_percent_mass=("top_5_percent_mass", "std"),
            mean_top_10_percent_mass=("top_10_percent_mass", "mean"),
            std_top_10_percent_mass=("top_10_percent_mass", "std"),
        )
        .fillna(0)
    )
    return summary.sort_values("explanation_mode").reset_index(drop=True)


def summarize_class_specificity_by_method(records: pd.DataFrame) -> pd.DataFrame:
    """Aggregate predicted-vs-runner-up correlation metrics by method."""
    missing = sorted(
        set(CLASS_SPECIFICITY_METRIC_COLUMNS + ("explanation_mode",)).difference(records.columns)
    )
    if missing:
        raise ValueError(
            "Class-specificity frame is missing required columns: "
            f"{', '.join(missing)}"
        )

    audio_count_column = "audio_path" if "audio_path" in records.columns else "explanation_mode"
    summary = (
        records.groupby("explanation_mode", as_index=False)
        .agg(
            examples=(audio_count_column, "nunique"),
            records=("explanation_mode", "size"),
            mean_pearson_correlation=("pearson_correlation", "mean"),
            std_pearson_correlation=("pearson_correlation", "std"),
            median_pearson_correlation=("pearson_correlation", "median"),
            invalid_pearson=("pearson_correlation", lambda values: int(values.isna().sum())),
            mean_spearman_correlation=("spearman_correlation", "mean"),
            std_spearman_correlation=("spearman_correlation", "std"),
            median_spearman_correlation=("spearman_correlation", "median"),
            invalid_spearman=("spearman_correlation", lambda values: int(values.isna().sum())),
        )
        .fillna(0)
    )
    return summary.sort_values("explanation_mode").reset_index(drop=True)
