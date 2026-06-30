"""IEMOCAP parsing for the standard four-class utterance-level protocol.

The checkpoint has labels ``neu``, ``hap``, ``ang``, and ``sad``. IEMOCAP's
``exc`` label is conventionally merged into ``hap``; all remaining source
labels are excluded instead of being silently relabelled.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re

from src.evaluation.common import MODEL_CLASS_NAMES


IEMOCAP_CLASS_BY_EMOTION = {
    "neu": 0,
    "hap": 1,
    "exc": 1,
    "ang": 2,
    "sad": 3,
}
IEMOCAP_SOURCE_LABELS = tuple(IEMOCAP_CLASS_BY_EMOTION)
STANDARD_SESSION_IDS = (1, 2, 3, 4, 5)
_EMOTION_LINE = re.compile(
    r"^\s*\[[^\]]+\]\s+(?P<utterance_id>\S+)\s+(?P<emotion>\S+)"
)


@dataclass(frozen=True)
class IemocapRecord:
    """One IEMOCAP utterance with an official categorical emotion label."""

    path: Path
    session_id: int
    dialogue_id: str
    utterance_id: str
    source_emotion: str
    true_class: int
    annotation_path: Path

    def to_dict(self) -> dict:
        record = asdict(self)
        record["path"] = str(self.path)
        record["annotation_path"] = str(self.annotation_path)
        return record


def resolve_iemocap_root(dataset_root: str | Path) -> Path:
    """Accept either IEMOCAP's release root or the local ``data/iemocap`` root."""
    root = Path(dataset_root).expanduser()
    nested_release = root / "archive" / "IEMOCAP_full_release"
    if nested_release.is_dir():
        root = nested_release
    if not root.is_dir():
        raise FileNotFoundError(f"IEMOCAP directory does not exist: {root}")
    return root.resolve()


def _records_from_annotation(
    annotation_path: Path,
    session_id: int,
    dataset_root: Path,
) -> list[IemocapRecord]:
    """Parse the official per-dialogue evaluation file for supported labels."""
    records = []
    for line_number, line in enumerate(annotation_path.read_text(encoding="utf-8").splitlines(), start=1):
        match = _EMOTION_LINE.match(line)
        if match is None:
            continue

        utterance_id = match.group("utterance_id")
        source_emotion = match.group("emotion").lower()
        if source_emotion not in IEMOCAP_CLASS_BY_EMOTION:
            continue

        dialogue_id, separator, _turn_id = utterance_id.rpartition("_")
        if not separator:
            raise ValueError(
                f"Could not infer IEMOCAP dialogue ID from {utterance_id!r} in "
                f"{annotation_path}:{line_number}."
            )
        audio_path = (
            dataset_root
            / f"Session{session_id}"
            / "sentences"
            / "wav"
            / dialogue_id
            / f"{utterance_id}.wav"
        )
        if not audio_path.is_file():
            raise FileNotFoundError(
                "IEMOCAP annotation refers to a missing utterance WAV: "
                f"{audio_path} (from {annotation_path}:{line_number})."
            )
        records.append(
            IemocapRecord(
                path=audio_path,
                session_id=session_id,
                dialogue_id=dialogue_id,
                utterance_id=utterance_id,
                source_emotion=source_emotion,
                true_class=IEMOCAP_CLASS_BY_EMOTION[source_emotion],
                annotation_path=annotation_path,
            )
        )
    return records


def collect_iemocap_records(
    dataset_root: str | Path,
    session_ids: tuple[int, ...] = STANDARD_SESSION_IDS,
) -> list[IemocapRecord]:
    """Collect the standard four-class IEMOCAP utterance subset.

    ``exc`` is mapped to model class ``hap``. The source classes ``fru``,
    ``fea``, ``dis``, ``sur``, ``oth``, and ``xxx`` are excluded. Passing a
    subset of sessions is useful for an explicit session-based protocol.
    """
    root = resolve_iemocap_root(dataset_root)
    invalid_sessions = sorted(set(session_ids).difference(STANDARD_SESSION_IDS))
    if invalid_sessions:
        raise ValueError(
            f"IEMOCAP session IDs must be in {STANDARD_SESSION_IDS}, got {invalid_sessions}."
        )
    if not session_ids:
        raise ValueError("At least one IEMOCAP session must be selected.")

    records = []
    seen_utterance_ids = set()
    for session_id in sorted(set(session_ids)):
        evaluation_dir = root / f"Session{session_id}" / "dialog" / "EmoEvaluation"
        if not evaluation_dir.is_dir():
            raise FileNotFoundError(f"IEMOCAP evaluation directory does not exist: {evaluation_dir}")

        annotation_paths = sorted(evaluation_dir.glob("*.txt"))
        if not annotation_paths:
            raise RuntimeError(f"No IEMOCAP annotation files found in {evaluation_dir}.")
        for annotation_path in annotation_paths:
            for record in _records_from_annotation(annotation_path, session_id, root):
                if record.utterance_id in seen_utterance_ids:
                    raise RuntimeError(
                        f"Duplicate IEMOCAP utterance annotation: {record.utterance_id}."
                    )
                seen_utterance_ids.add(record.utterance_id)
                records.append(record)

    if not records:
        raise RuntimeError(
            "No compatible IEMOCAP four-class utterances found. Expected the "
            "official Session*/dialog/EmoEvaluation/*.txt structure."
        )
    return sorted(records, key=lambda record: record.path)


def model_label_for_source_emotion(source_emotion: str) -> str:
    """Return the checkpoint label corresponding to a supported IEMOCAP label."""
    return MODEL_CLASS_NAMES[IEMOCAP_CLASS_BY_EMOTION[source_emotion]]
