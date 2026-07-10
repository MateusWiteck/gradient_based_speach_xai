"""Evaluate class-specificity of temporal explanation maps over many audios.

For each correctly classified utterance, this script computes the relevance map
for the predicted class and for the runner-up class. It then records the
Pearson/Spearman correlation between the two temporal maps for each explanation
mode. Lower correlations mean the explanation changes more when the explained
class changes, which is the desired class-specificity behavior.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.common import (
    MODEL_CLASS_NAMES,
    PROVENANCE_COLUMNS,
    create_unique_run_dir,
    dataset_slug,
    example_provenance,
    infer_dataset_name,
    parse_csv_values,
    select_correct_examples,
    write_json,
)
from src.evaluation.explanation_metrics import (
    class_specificity_metrics,
    summarize_class_specificity_by_method,
)
from src.explainers.transformer_relevance.attention_extractor import (
    extract_hubert_attentions,
)
from src.explainers.transformer_relevance.gradient_attention import (
    extract_gradient_weighted_attentions,
)
from src.explainers.transformer_relevance.legrad_hubert import (
    LEGRAD_HUBERT_EXPLANATION_MODES,
    extract_legrad_hubert_relevance,
)
from src.explainers.transformer_relevance.score_pipeline import (
    EXPLANATION_MODES,
    compute_relevance_scores,
)
from src.models.load_model import load_hubert_emotion_model
from src.utils.audio import load_audio_mono_16k


def runner_up_class(logits, predicted_class: int) -> int:
    """Return the highest-logit class other than the predicted class."""
    candidates = logits.detach().clone()
    candidates[:, predicted_class] = -float("inf")
    return int(candidates.argmax(dim=-1).item())


def compute_scores_for_target(
    *,
    model,
    processor,
    waveform,
    sampling_rate: int,
    device: str,
    base_result: dict,
    target_class: int,
    include_legrad: bool,
) -> dict:
    """Compute all explanation modes for one explicit target class."""
    grad_result = extract_gradient_weighted_attentions(
        model=model,
        processor=processor,
        waveform=waveform,
        sampling_rate=sampling_rate,
        device=device,
        target_class=target_class,
    )
    legrad_result = None
    if include_legrad:
        legrad_result = extract_legrad_hubert_relevance(
            model=model,
            processor=processor,
            waveform=waveform,
            sampling_rate=sampling_rate,
            device=device,
            target_class=target_class,
        )
    return compute_relevance_scores(
        model,
        base_result,
        grad_result,
        legrad_result=legrad_result,
    )


def summarize_by_predicted_class(records: pd.DataFrame) -> pd.DataFrame:
    """Aggregate class-specificity by predicted emotion and explanation mode."""
    if records.empty:
        return records
    return (
        records.groupby(["predicted_label", "explanation_mode"], as_index=False)
        .agg(
            examples=("audio_path", "nunique"),
            records=("explanation_mode", "size"),
            mean_pearson_correlation=("pearson_correlation", "mean"),
            std_pearson_correlation=("pearson_correlation", "std"),
            median_pearson_correlation=("pearson_correlation", "median"),
            mean_spearman_correlation=("spearman_correlation", "mean"),
            std_spearman_correlation=("spearman_correlation", "std"),
            median_spearman_correlation=("spearman_correlation", "median"),
        )
        .fillna(0)
        .sort_values(["predicted_label", "explanation_mode"])
        .reset_index(drop=True)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Optional dataset name for output-folder provenance (inferred when available).",
    )
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--max-examples", type=int, default=8)
    parser.add_argument("--modes", default=",".join(EXPLANATION_MODES))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Inference device (default: automatically select CUDA when available).",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the checkpoint only from the local Hugging Face cache.",
    )
    args = parser.parse_args()

    if args.max_examples < 1:
        parser.error("--max-examples must be positive.")
    modes = parse_csv_values(args.modes, str, "modes")
    invalid_modes = sorted(set(modes).difference(EXPLANATION_MODES))
    if invalid_modes:
        parser.error(f"Unsupported explanation modes: {invalid_modes}")
    include_legrad = any(mode in LEGRAD_HUBERT_EXPLANATION_MODES for mode in modes)

    predictions = pd.read_csv(args.predictions_csv)
    dataset_name = infer_dataset_name(predictions, args.dataset_name)
    selected_examples = select_correct_examples(predictions, args.max_examples, args.seed)
    run_dir = create_unique_run_dir(
        args.output_root,
        f"{dataset_slug(dataset_name)}_class_specificity",
    )
    selected_examples.to_csv(run_dir / "selected_examples.csv", index=False)
    audio_manifest_columns = [
        column
        for column in (*PROVENANCE_COLUMNS, "true_class", "true_label", "predicted_class", "predicted_label")
        if column in selected_examples.columns
    ]
    selected_examples[audio_manifest_columns].to_csv(
        run_dir / "audio_manifest.csv",
        index=False,
    )

    model, processor, device = load_hubert_emotion_model(
        device=args.device,
        local_files_only=args.local_files_only,
    )
    class_names = [
        str(model.config.id2label[index])
        for index in range(model.config.num_labels)
    ]

    records = []
    for example_index, example in selected_examples.iterrows():
        audio_path = Path(example["audio_path"])
        provenance = example_provenance(example)
        waveform, sampling_rate = load_audio_mono_16k(audio_path)
        base_result = extract_hubert_attentions(
            model=model,
            processor=processor,
            waveform=waveform,
            sampling_rate=sampling_rate,
            device=device,
        )
        predicted_class = int(base_result["predicted_class"])
        runner_up = runner_up_class(base_result["logits"], predicted_class)

        predicted_relevance = compute_scores_for_target(
            model=model,
            processor=processor,
            waveform=waveform,
            sampling_rate=sampling_rate,
            device=device,
            base_result=base_result,
            target_class=predicted_class,
            include_legrad=include_legrad,
        )
        runner_up_relevance = compute_scores_for_target(
            model=model,
            processor=processor,
            waveform=waveform,
            sampling_rate=sampling_rate,
            device=device,
            base_result=base_result,
            target_class=runner_up,
            include_legrad=include_legrad,
        )

        for mode in modes:
            metrics = class_specificity_metrics(
                predicted_relevance["scores"][mode].squeeze(0),
                runner_up_relevance["scores"][mode].squeeze(0),
            )
            records.append(
                {
                    **provenance,
                    "true_class": int(example["true_class"]),
                    "true_label": str(example["true_label"]),
                    "predicted_class": predicted_class,
                    "predicted_label": class_names[predicted_class],
                    "runner_up_class": runner_up,
                    "runner_up_label": class_names[runner_up],
                    "predicted_target_contrast_class": int(
                        predicted_relevance["contrast_classes"][0].item()
                    ),
                    "runner_up_target_contrast_class": int(
                        runner_up_relevance["contrast_classes"][0].item()
                    ),
                    "explanation_mode": mode,
                    **metrics,
                }
            )

        print(f"Compared {example_index + 1}/{len(selected_examples)}: {audio_path.name}")

    record_frame = pd.DataFrame(records)
    summary_by_method = summarize_class_specificity_by_method(record_frame)
    summary_by_class = summarize_by_predicted_class(record_frame)
    selected_counts = selected_examples.groupby("true_label").size()
    pd.DataFrame(
        {
            "true_label": MODEL_CLASS_NAMES,
            "selected_correct_examples": [
                int(selected_counts.get(label, 0)) for label in MODEL_CLASS_NAMES
            ],
        }
    ).to_csv(run_dir / "class_coverage.csv", index=False)
    record_frame.to_csv(run_dir / "class_specificity_records.csv", index=False)
    summary_by_method.to_csv(run_dir / "class_specificity_summary_by_method.csv", index=False)
    summary_by_class.to_csv(run_dir / "class_specificity_summary_by_class.csv", index=False)
    write_json(
        run_dir / "config.json",
        {
            "dataset_name": dataset_name,
            "predictions_csv": str(Path(args.predictions_csv).resolve()),
            "audio_manifest": "audio_manifest.csv",
            "selected_examples_file": "selected_examples.csv",
            "class_specificity_records_file": "class_specificity_records.csv",
            "class_specificity_summary_by_method_file": (
                "class_specificity_summary_by_method.csv"
            ),
            "source_audio_column": "audio_path",
            "selection": "correct-only, approximately class-balanced",
            "max_examples": args.max_examples,
            "modes": modes,
            "seed": args.seed,
            "device": device,
            "local_files_only": args.local_files_only,
            "metric_interpretation": (
                "Lower predicted-vs-runner-up correlation indicates stronger "
                "class-specificity."
            ),
        },
    )
    print("\nClass-specificity evaluation complete")
    print("Examples:", len(selected_examples))
    print("Records:", len(record_frame))
    print("Saved new run to:", run_dir)


if __name__ == "__main__":
    main()
