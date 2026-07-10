"""RAVDESS parsing, strict label mapping, metrics, and safe output helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from src.evaluation.common import (
    MODEL_CLASS_NAMES,
    classification_metrics,
    create_unique_run_dir,
    write_json,
)


# RAVDESS speech-file emotion codes. The current HuBERT classifier has outputs
# only for the four entries in MODEL_CLASS_BY_RAVDESS_EMOTION.
RAVDESS_EMOTION_NAMES = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}
MODEL_CLASS_BY_RAVDESS_EMOTION = {
    "01": 0,  # neutral -> neu
    "03": 1,  # happy -> hap
    "05": 2,  # angry -> ang
    "04": 3,  # sad -> sad
}


@dataclass(frozen=True)
class RavdessRecord:
    """Metadata parsed from a canonical RAVDESS speech filename."""

    path: Path
    actor_id: int
    emotion_code: str
    emotion_name: str
    true_class: int

    def to_dict(self) -> dict:
        record = asdict(self)
        record["path"] = str(self.path)
        return record


def parse_ravdess_speech_file(path: Path) -> RavdessRecord | None:
    """Parse one RAVDESS audio-speech file or return ``None`` when unsupported.

    RAVDESS filenames contain seven dash-separated fields. This evaluator uses
    only modality ``03`` (audio-only) and vocal channel ``01`` (speech), then
    filters to the four labels that have an exact output class in the current
    SUPERB HuBERT checkpoint.
    """
    fields = path.stem.split("-")
    if len(fields) != 7:
        return None

    modality, vocal_channel, emotion_code, *_unused, actor = fields
    if modality != "03" or vocal_channel != "01":
        return None
    if emotion_code not in MODEL_CLASS_BY_RAVDESS_EMOTION:
        return None

    try:
        actor_id = int(actor)
    except ValueError as error:
        raise ValueError(f"Invalid RAVDESS actor id in {path.name!r}.") from error

    return RavdessRecord(
        path=path,
        actor_id=actor_id,
        emotion_code=emotion_code,
        emotion_name=RAVDESS_EMOTION_NAMES[emotion_code],
        true_class=MODEL_CLASS_BY_RAVDESS_EMOTION[emotion_code],
    )


def collect_ravdess_records(dataset_root: str | Path) -> list[RavdessRecord]:
    """Collect strict four-class RAVDESS speech records in stable path order."""
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"RAVDESS directory does not exist: {root}")

    records = []
    for path in sorted(root.rglob("*.wav")):
        record = parse_ravdess_speech_file(path)
        if record is not None:
            records.append(record)

    if not records:
        raise RuntimeError(
            "No compatible RAVDESS audio-speech files found. Expected paths like "
            "Audio_Speech_Actors_01-24/Actor_01/03-01-01-01-01-01-01.wav."
        )
    return records

