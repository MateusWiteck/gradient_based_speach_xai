from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeletionResult:
    audio_id: str
    k: int
    method: str
    masked_duration: float
    original_label: str
    original_confidence: float
    masked_confidence: float
    drop: float
    relative_drop: float
    prediction_flipped: bool
    random_trial: int | None = None


def confidence_drop(original_confidence: float, masked_confidence: float) -> tuple[float, float]:
    drop = original_confidence - masked_confidence
    relative_drop = drop / original_confidence if original_confidence > 0 else 0.0
    return drop, relative_drop


def build_deletion_result(
    audio_id: str,
    k: int,
    method: str,
    masked_duration: float,
    original_label: str,
    masked_label: str,
    original_confidence: float,
    masked_confidence: float,
    random_trial: int | None = None,
) -> DeletionResult:
    drop, relative_drop = confidence_drop(original_confidence, masked_confidence)
    return DeletionResult(
        audio_id=audio_id,
        k=k,
        method=method,
        masked_duration=masked_duration,
        original_label=original_label,
        original_confidence=original_confidence,
        masked_confidence=masked_confidence,
        drop=drop,
        relative_drop=relative_drop,
        prediction_flipped=masked_label != original_label,
        random_trial=random_trial,
    )
