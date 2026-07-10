"""Evaluate the IEMOCAP-trained HuBERT checkpoint on IEMOCAP utterance labels.

This script implements the standard four-class mapping: ``neu``, ``hap`` plus
``exc``, ``ang``, and ``sad``. It is an in-domain diagnostic, not an external
generalization result. All outputs go to a new timestamped directory.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.common import (
    MODEL_CLASS_NAMES,
    classification_metrics,
    create_unique_run_dir,
    write_json,
)
from src.evaluation.iemocap import (
    IEMOCAP_CLASS_BY_EMOTION,
    STANDARD_SESSION_IDS,
    collect_iemocap_records,
    resolve_iemocap_root,
)


DEFAULT_IEMOCAP_ROOT = "data/iemocap/archive/IEMOCAP_full_release"


def iter_batches(items, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def select_deterministic_subset(records, max_samples: int | None):
    """Return a small class-balanced subset for a representative smoke test."""
    if max_samples is None:
        return records
    class_count = len(MODEL_CLASS_NAMES)
    base_count, remainder = divmod(max_samples, class_count)
    selected = []
    for class_id in range(class_count):
        class_records = [record for record in records if record.true_class == class_id]
        requested_count = base_count + (1 if class_id < remainder else 0)
        selected.extend(class_records[:requested_count])
    return sorted(selected, key=lambda record: record.path)


def parse_session_ids(value: str) -> tuple[int, ...]:
    """Parse a comma-separated, stable set of IEMOCAP session IDs."""
    try:
        session_ids = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    except ValueError as error:
        raise ValueError(f"Could not parse --sessions: {value!r}") from error
    if not session_ids:
        raise ValueError("--sessions must contain at least one session ID.")
    invalid_sessions = sorted(set(session_ids).difference(STANDARD_SESSION_IDS))
    if invalid_sessions:
        raise ValueError(
            f"--sessions must use IDs from {STANDARD_SESSION_IDS}, got {invalid_sessions}."
        )
    return session_ids


def model_class_names(model) -> tuple[str, ...]:
    """Validate that the loaded checkpoint has the expected four output labels."""
    labels = tuple(str(model.config.id2label[index]) for index in range(model.config.num_labels))
    if labels != MODEL_CLASS_NAMES:
        raise RuntimeError(
            "The evaluator expects model classes "
            f"{MODEL_CLASS_NAMES}, but loaded {labels}."
        )
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iemocap-root", default=DEFAULT_IEMOCAP_ROOT)
    parser.add_argument(
        "--sessions",
        default="1,2,3,4,5",
        help="Comma-separated IEMOCAP sessions to include (default: all five).",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Inference device (default: automatically select CUDA when available).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional small deterministic class-balanced subset for a smoke test.",
    )
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the checkpoint only from the local Hugging Face cache.",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive when supplied.")
    try:
        session_ids = parse_session_ids(args.sessions)
    except ValueError as error:
        parser.error(str(error))

    dataset_root = resolve_iemocap_root(args.iemocap_root)
    records = collect_iemocap_records(dataset_root, session_ids=session_ids)
    records = select_deterministic_subset(records, args.max_samples)

    run_dir = create_unique_run_dir(args.output_root, "iemocap_in_domain_eval")
    # Delay the heavyweight model imports so ``--help`` and label parsing remain
    # available even in lightweight environments without Torch installed.
    import torch

    from src.models.load_model import load_hubert_emotion_model
    from src.utils.audio import load_audio_mono_16k

    model, processor, device = load_hubert_emotion_model(
        device=args.device,
        local_files_only=args.local_files_only,
    )
    class_names = model_class_names(model)

    rows = []
    for batch_index, batch_records in enumerate(iter_batches(records, args.batch_size), start=1):
        waveforms = []
        for record in batch_records:
            waveform, sampling_rate = load_audio_mono_16k(record.path)
            if sampling_rate != 16_000:
                raise RuntimeError(f"Expected 16 kHz audio after loading {record.path}.")
            waveforms.append(waveform.numpy())

        inputs = processor(
            waveforms,
            sampling_rate=16_000,
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits.detach().cpu()
        probabilities = torch.softmax(logits, dim=-1)
        predicted_classes = logits.argmax(dim=-1)

        for record, row_logits, row_probabilities, predicted_class in zip(
            batch_records,
            logits,
            probabilities,
            predicted_classes,
        ):
            predicted_class = int(predicted_class.item())
            row = {
                "dataset": "IEMOCAP",
                "audio_path": str(record.path),
                "relative_path": record.path.relative_to(dataset_root).as_posix(),
                "session_id": record.session_id,
                "dialogue_id": record.dialogue_id,
                "utterance_id": record.utterance_id,
                "iemocap_emotion": record.source_emotion,
                "annotation_path": str(record.annotation_path),
                "true_class": record.true_class,
                "true_label": class_names[record.true_class],
                "predicted_class": predicted_class,
                "predicted_label": class_names[predicted_class],
                "is_correct": predicted_class == record.true_class,
                "confidence": float(row_probabilities[predicted_class].item()),
            }
            for class_id, class_name in enumerate(class_names):
                row[f"logit_{class_name}"] = float(row_logits[class_id].item())
                row[f"probability_{class_name}"] = float(row_probabilities[class_id].item())
            rows.append(row)

        print(f"Processed batch {batch_index}: {len(rows)}/{len(records)} utterances")

    predictions = pd.DataFrame(rows)
    confusion, metrics = classification_metrics(
        predictions["true_class"].tolist(),
        predictions["predicted_class"].tolist(),
        class_names,
    )
    source_label_counts = Counter(record.source_emotion for record in records)
    metrics.update(
        {
            "dataset": "IEMOCAP full release standard four-class utterance subset",
            "evaluation_type": "in-domain diagnostic; not a held-out external evaluation",
            "protocol_note": (
                "The public checkpoint may have been trained on some or all selected "
                "IEMOCAP sessions. Use an explicitly documented held-out protocol "
                "before interpreting this as generalization performance."
            ),
            "iemocap_root": str(dataset_root),
            "selected_sessions": list(session_ids),
            "label_mapping": {
                "neu": "neu",
                "hap": "hap",
                "exc": "hap",
                "ang": "ang",
                "sad": "sad",
            },
            "included_source_label_counts": dict(sorted(source_label_counts.items())),
            "excluded_source_labels": ["fru", "fea", "dis", "sur", "oth", "xxx"],
            "model_labels": list(class_names),
            "batch_size": args.batch_size,
            "device": device,
            "max_samples": args.max_samples,
            "local_files_only": args.local_files_only,
        }
    )

    predictions.to_csv(run_dir / "predictions.csv", index=False)
    pd.DataFrame(confusion, index=class_names, columns=class_names).rename_axis(
        "true_label", axis="index"
    ).rename_axis("predicted_label", axis="columns").to_csv(run_dir / "confusion_matrix.csv")
    write_json(run_dir / "metrics.json", metrics)
    write_json(
        run_dir / "run_manifest.json",
        {
            "dataset_root": str(dataset_root),
            "input_audio_manifest": "predictions.csv (audio_path column)",
            "annotation_source": "Session*/dialog/EmoEvaluation/*.txt",
            "prediction_file": "predictions.csv",
            "metrics_file": "metrics.json",
            "confusion_matrix_file": "confusion_matrix.csv",
            "num_evaluated_files": len(predictions),
            "selected_sessions": list(session_ids),
            "source_to_model_label_mapping": {
                label: class_names[class_id]
                for label, class_id in IEMOCAP_CLASS_BY_EMOTION.items()
            },
            "model_labels": list(class_names),
            "device": device,
        },
    )

    print("\nIEMOCAP in-domain evaluation complete")
    print("Examples:", metrics["num_examples"])
    print("Accuracy:", f"{metrics['accuracy']:.4f}")
    print("Balanced accuracy / UA:", f"{metrics['balanced_accuracy']:.4f}")
    print("Macro-F1:", f"{metrics['macro_f1']:.4f}")
    print("Saved new run to:", run_dir)


if __name__ == "__main__":
    main()
