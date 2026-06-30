"""Shared helpers for four-class speech-emotion evaluation datasets."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


# The loaded SUPERB HuBERT emotion-recognition checkpoint exposes exactly these
# four classifier outputs, in this order.
MODEL_CLASS_NAMES = ("neu", "hap", "ang", "sad")
PROVENANCE_COLUMNS = (
    "dataset",
    "audio_path",
    "relative_path",
    "actor_id",
    "session_id",
    "dialogue_id",
    "utterance_id",
    "ravdess_emotion",
    "iemocap_emotion",
    "annotation_path",
)


def create_unique_run_dir(output_root: str | Path, prefix: str) -> Path:
    """Create a new timestamped directory; never reuse or replace an old run."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    for attempt in range(10_000):
        suffix = "" if attempt == 0 else f"_{attempt:04d}"
        run_dir = output_root / f"{prefix}_{timestamp}{suffix}"
        try:
            run_dir.mkdir()
        except FileExistsError:
            continue
        return run_dir

    raise RuntimeError("Could not create a unique output directory after 10,000 attempts.")


def classification_metrics(
    true_classes: list[int],
    predicted_classes: list[int],
    class_names: tuple[str, ...] = MODEL_CLASS_NAMES,
) -> tuple[np.ndarray, dict]:
    """Return a confusion matrix plus accuracy, macro-F1, and class metrics."""
    if len(true_classes) != len(predicted_classes) or not true_classes:
        raise ValueError("true_classes and predicted_classes must be non-empty and equal length.")

    num_classes = len(class_names)
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for true_class, predicted_class in zip(true_classes, predicted_classes):
        if not 0 <= true_class < num_classes or not 0 <= predicted_class < num_classes:
            raise ValueError("Class id outside the configured model label space.")
        confusion[true_class, predicted_class] += 1

    support = confusion.sum(axis=1)
    predicted_count = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros(num_classes, dtype=float),
        where=predicted_count != 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros(num_classes, dtype=float),
        where=support != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(num_classes, dtype=float),
        where=(precision + recall) != 0,
    )

    per_class = {
        class_name: {
            "support": int(support[class_id]),
            "precision": float(precision[class_id]),
            "recall": float(recall[class_id]),
            "f1": float(f1[class_id]),
        }
        for class_id, class_name in enumerate(class_names)
    }
    evaluated_classes = support > 0
    metrics = {
        "num_examples": int(confusion.sum()),
        "evaluated_class_count": int(evaluated_classes.sum()),
        "accuracy": float(true_positive.sum() / confusion.sum()),
        "balanced_accuracy": float(recall[evaluated_classes].mean()),
        "macro_f1": float(f1[evaluated_classes].mean()),
        "per_class": per_class,
    }
    return confusion, metrics


def write_json(path: str | Path, value: dict) -> None:
    """Write UTF-8 JSON with stable readable formatting."""
    with Path(path).open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def parse_csv_values(value: str, value_type, name: str) -> list:
    """Parse comma-separated command-line values with a clear error message."""
    try:
        values = [value_type(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError(f"Could not parse --{name}: {value!r}") from error
    if not values:
        raise ValueError(f"--{name} must contain at least one value.")
    return values


def infer_dataset_name(predictions: pd.DataFrame, requested_name: str | None) -> str:
    """Use an explicit name or infer one from a standardized dataset column."""
    if requested_name:
        return requested_name
    if "dataset" in predictions:
        names = sorted(
            str(value) for value in predictions["dataset"].dropna().unique() if str(value)
        )
        if len(names) == 1:
            return names[0]
    return "dataset"


def dataset_slug(dataset_name: str) -> str:
    """Return a filesystem-safe, readable dataset name for run folders."""
    slug = re.sub(r"[^a-z0-9]+", "_", dataset_name.lower()).strip("_")
    return slug or "dataset"


def example_provenance(example: pd.Series) -> dict:
    """Preserve dataset-specific audio identifiers in downstream CSV files."""
    return {
        column: example[column]
        for column in PROVENANCE_COLUMNS
        if column in example.index and pd.notna(example[column])
    }


def select_correct_examples(
    predictions: pd.DataFrame,
    max_examples: int,
    seed: int,
) -> pd.DataFrame:
    """Select an approximately class-balanced, deterministic correct subset."""
    required_columns = {
        "audio_path",
        "true_class",
        "true_label",
        "predicted_class",
        "predicted_label",
        "is_correct",
    }
    missing_columns = sorted(required_columns.difference(predictions.columns))
    if missing_columns:
        raise ValueError(
            "The prediction CSV is missing required columns: "
            f"{', '.join(missing_columns)}."
        )
    correct = predictions[predictions["is_correct"].astype(bool)].copy()
    if correct.empty:
        raise RuntimeError("The prediction CSV has no correctly classified examples.")

    rng = np.random.default_rng(seed)
    groups = []
    class_ids = sorted(correct["true_class"].unique())
    per_class = max(1, int(np.ceil(max_examples / len(class_ids))))
    for class_id in class_ids:
        group = correct[correct["true_class"] == class_id]
        chosen_indices = rng.choice(
            group.index.to_numpy(),
            size=min(per_class, len(group)),
            replace=False,
        )
        groups.append(group.loc[chosen_indices])

    selected = pd.concat(groups).sort_values(["true_class", "audio_path"]).head(max_examples)
    return selected.reset_index(drop=True)
