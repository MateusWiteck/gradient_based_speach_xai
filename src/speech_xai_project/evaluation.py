from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeletionResult:
    audio_id: str
    method: str
    budget: float
    original_label: str
    original_confidence: float
    masked_confidence: float
    drop: float
    relative_drop: float
    prediction_flipped: bool


def confidence_drop(original_confidence: float, masked_confidence: float) -> tuple[float, float]:
    drop = original_confidence - masked_confidence
    relative_drop = drop / original_confidence if original_confidence > 0 else 0.0
    return drop, relative_drop


def build_deletion_result(
    audio_id: str,
    method: str,
    budget: float,
    original_label: str,
    masked_label: str,
    original_confidence: float,
    masked_confidence: float,
) -> DeletionResult:
    drop, relative_drop = confidence_drop(original_confidence, masked_confidence)
    return DeletionResult(
        audio_id=audio_id,
        method=method,
        budget=budget,
        original_label=original_label,
        original_confidence=original_confidence,
        masked_confidence=masked_confidence,
        drop=drop,
        relative_drop=relative_drop,
        prediction_flipped=masked_label != original_label,
    )

