"""Evaluate the IEMOCAP-trained HuBERT checkpoint on strict RAVDESS labels.

This is an external, zero-shot evaluation: it does not train or tune the model.
Every invocation writes to a fresh timestamped directory under ``outputs/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.ravdess import (
    MODEL_CLASS_NAMES,
    classification_metrics,
    collect_ravdess_records,
    create_unique_run_dir,
    write_json,
)
from src.models.load_model import load_hubert_emotion_model
from src.utils.audio import load_audio_mono_16k


DEFAULT_RAVDESS_ROOT = "data/ravdess/Audio_Speech_Actors_01-24"


def iter_batches(items, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


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
    parser.add_argument("--ravdess-root", default=DEFAULT_RAVDESS_ROOT)
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
        help="Optional small deterministic subset for a smoke test.",
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

    records = collect_ravdess_records(args.ravdess_root)
    if args.max_samples is not None:
        records = records[: args.max_samples]

    run_dir = create_unique_run_dir(args.output_root, "ravdess_external_eval")
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
                "audio_path": str(record.path.resolve()),
                "relative_path": str(record.path),
                "actor_id": record.actor_id,
                "ravdess_emotion": record.emotion_name,
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
    metrics.update(
        {
            "dataset": "RAVDESS Audio_Speech strict four-class subset",
            "evaluation_type": "zero-shot external cross-corpus evaluation",
            "ravdess_root": str(Path(args.ravdess_root).resolve()),
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
            "dataset_root": str(Path(args.ravdess_root).resolve()),
            "input_audio_manifest": "predictions.csv (audio_path column)",
            "prediction_file": "predictions.csv",
            "metrics_file": "metrics.json",
            "confusion_matrix_file": "confusion_matrix.csv",
            "num_evaluated_files": len(predictions),
            "strict_ravdess_labels": ["neutral", "happy", "angry", "sad"],
            "model_labels": list(class_names),
            "device": device,
        },
    )

    print("\nExternal RAVDESS evaluation complete")
    print("Examples:", metrics["num_examples"])
    print("Accuracy:", f"{metrics['accuracy']:.4f}")
    print("Balanced accuracy / UA:", f"{metrics['balanced_accuracy']:.4f}")
    print("Macro-F1:", f"{metrics['macro_f1']:.4f}")
    print("Saved new run to:", run_dir)


if __name__ == "__main__":
    main()
