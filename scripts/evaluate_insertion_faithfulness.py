"""Compare temporal explanation modes with insertion-faithfulness tests.

Insertion is the positive counterpart of deletion: start from a silent waveform
and progressively insert the original audio regions selected by an explanation.
If the relevance map is faithful, inserting top-ranked regions should recover
the original target-class score faster than inserting bottom-ranked or random
regions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


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
from src.explainers.transformer_relevance.attention_extractor import (
    extract_hubert_attentions,
)
from src.explainers.transformer_relevance.gradient_attention import (
    extract_gradient_weighted_attentions,
)
from src.explainers.transformer_relevance.score_pipeline import (
    EXPLANATION_MODES,
    compute_relevance_scores,
)
from src.models.load_model import load_hubert_emotion_model
from src.utils.audio import load_audio_mono_16k


CONTRASTIVE_MODES = {"level3-contrastive", "contrastive-conditioned-rollout"}
EPS = 1e-8


def compute_explanation_scores(model, processor, waveform, sampling_rate: int, device: str):
    """Compute every shared score variant from one original audio utterance."""
    base_result = extract_hubert_attentions(
        model=model,
        processor=processor,
        waveform=waveform,
        sampling_rate=sampling_rate,
        device=device,
    )
    grad_result = extract_gradient_weighted_attentions(
        model=model,
        processor=processor,
        waveform=waveform,
        sampling_rate=sampling_rate,
        device=device,
    )
    relevance = compute_relevance_scores(model, base_result, grad_result)
    return (
        relevance["scores"],
        relevance["target_class"],
        int(relevance["contrast_classes"][0].item()),
    )


def select_token_indices(scores: torch.Tensor, fraction: float, strategy: str, rng) -> np.ndarray:
    """Select a fixed number of top, bottom, or random time-token indices."""
    if not 0 < fraction <= 1:
        raise ValueError("Insertion fractions must be in (0, 1].")
    values = scores.detach().cpu().numpy().reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("Explanation scores must be finite before token selection.")
    token_count = values.size
    selected_count = min(token_count, max(1, int(np.ceil(fraction * token_count))))

    if strategy == "top":
        return np.argsort(values)[-selected_count:]
    if strategy == "bottom":
        return np.argsort(values)[:selected_count]
    if strategy == "random":
        return rng.choice(token_count, size=selected_count, replace=False)
    raise ValueError(f"Unknown insertion strategy: {strategy}")


def _token_sample_mask(
    sample_count: int,
    token_indices: np.ndarray,
    token_count: int,
    device,
) -> torch.Tensor:
    """Return a boolean waveform mask for uniformly mapped token regions."""
    sample_mask = torch.zeros(sample_count, dtype=torch.bool, device=device)
    for token_index in token_indices:
        start = int(np.floor(token_index * sample_count / token_count))
        end = int(np.ceil((token_index + 1) * sample_count / token_count))
        sample_mask[start:end] = True
    return sample_mask


def insert_token_regions(
    waveform: torch.Tensor,
    token_indices: np.ndarray,
    token_count: int,
) -> torch.Tensor:
    """Create a silent waveform with selected original token regions inserted.

    This intentionally mirrors the deletion evaluator's uniform token-to-time
    approximation. HuBERT convolutional receptive fields overlap, so these
    regions are approximate temporal bins rather than exact acoustic frames.
    """
    inserted = torch.zeros_like(waveform)
    if len(token_indices) == 0:
        return inserted
    sample_mask = _token_sample_mask(
        sample_count=waveform.numel(),
        token_indices=token_indices,
        token_count=token_count,
        device=waveform.device,
    )
    inserted[sample_mask] = waveform[sample_mask]
    return inserted


def target_scores(model, processor, waveform, sampling_rate: int, target_class: int, device: str) -> tuple[float, float]:
    """Return the selected target logit and softmax probability for one waveform."""
    inputs = processor(
        waveform,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probability = torch.softmax(logits, dim=-1)[0, target_class]
    return float(logits[0, target_class].item()), float(probability.item())


def normalized_recovery(inserted_score: float, silent_score: float, original_score: float) -> float:
    """Measure how much of the original-vs-silence score gap was recovered."""
    denominator = original_score - silent_score
    if abs(denominator) <= EPS:
        return float("nan")
    return float((inserted_score - silent_score) / denominator)


def trapezoid_auc(x_values: np.ndarray, y_values: np.ndarray) -> float:
    """Compute area under a curve with compatibility across NumPy versions."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y_values, x_values))
    return float(np.trapz(y_values, x_values))


def compute_auc_tables(record_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-example and aggregate insertion AUC tables."""
    group_columns = [
        "dataset",
        "audio_path",
        "true_label",
        "explanation_mode",
        "strategy",
        "trial",
    ]
    available_group_columns = [column for column in group_columns if column in record_frame.columns]
    auc_records = []
    for group_values, group in record_frame.groupby(available_group_columns, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group = group.sort_values("fraction")
        fractions = group["fraction"].to_numpy(dtype=float)
        max_fraction = float(fractions.max())
        normalizer = max(max_fraction - float(fractions.min()), EPS)
        row = dict(zip(available_group_columns, group_values))
        row.update(
            {
                "min_fraction": float(fractions.min()),
                "max_fraction": max_fraction,
                "auc_inserted_target_probability": trapezoid_auc(
                    fractions,
                    group["inserted_target_probability"].to_numpy(dtype=float),
                ),
                "auc_inserted_target_logit": trapezoid_auc(
                    fractions,
                    group["inserted_target_logit"].to_numpy(dtype=float),
                ),
                "auc_probability_recovery": trapezoid_auc(
                    fractions,
                    group["probability_recovery"].fillna(0).to_numpy(dtype=float),
                ),
                "auc_logit_recovery": trapezoid_auc(
                    fractions,
                    group["logit_recovery"].fillna(0).to_numpy(dtype=float),
                ),
                "mean_probability_recovery_over_curve": trapezoid_auc(
                    fractions,
                    group["probability_recovery"].fillna(0).to_numpy(dtype=float),
                )
                / normalizer,
                "mean_logit_recovery_over_curve": trapezoid_auc(
                    fractions,
                    group["logit_recovery"].fillna(0).to_numpy(dtype=float),
                )
                / normalizer,
            }
        )
        auc_records.append(row)

    auc_by_example = pd.DataFrame(auc_records)
    auc_by_method = (
        auc_by_example.groupby(["explanation_mode", "strategy"], as_index=False)
        .agg(
            examples=("audio_path", "nunique"),
            curves=("audio_path", "size"),
            mean_auc_probability_recovery=("auc_probability_recovery", "mean"),
            std_auc_probability_recovery=("auc_probability_recovery", "std"),
            mean_auc_logit_recovery=("auc_logit_recovery", "mean"),
            std_auc_logit_recovery=("auc_logit_recovery", "std"),
            mean_probability_recovery_over_curve=(
                "mean_probability_recovery_over_curve",
                "mean",
            ),
            std_probability_recovery_over_curve=(
                "mean_probability_recovery_over_curve",
                "std",
            ),
            mean_logit_recovery_over_curve=("mean_logit_recovery_over_curve", "mean"),
            std_logit_recovery_over_curve=("mean_logit_recovery_over_curve", "std"),
        )
        .fillna(0)
    )
    return auc_by_example, auc_by_method


def plot_insertion_curves(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot mean normalized probability recovery by insertion fraction."""
    figure, axis = plt.subplots(figsize=(10, 5.5))
    for (mode, strategy), group in summary.groupby(["explanation_mode", "strategy"]):
        group = group.sort_values("fraction")
        axis.plot(
            group["fraction"],
            group["mean_probability_recovery"],
            marker="o",
            label=f"{mode} / {strategy}",
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.axhline(1, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    axis.set_xlabel("Fraction of token regions inserted")
    axis.set_ylabel("Mean normalized target-probability recovery")
    axis.set_title("Insertion-faithfulness curves")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


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
    parser.add_argument("--fractions", default="0.05,0.1,0.2,0.3")
    parser.add_argument("--random-trials", type=int, default=3)
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

    if args.max_examples < 1 or args.random_trials < 1:
        parser.error("--max-examples and --random-trials must be positive.")
    fractions = parse_csv_values(args.fractions, float, "fractions")
    modes = parse_csv_values(args.modes, str, "modes")
    invalid_modes = sorted(set(modes).difference(EXPLANATION_MODES))
    if invalid_modes:
        parser.error(f"Unsupported explanation modes: {invalid_modes}")
    if any(not 0 < fraction <= 1 for fraction in fractions):
        parser.error("--fractions values must be in (0, 1].")
    fractions = sorted(set(fractions))

    predictions = pd.read_csv(args.predictions_csv)
    dataset_name = infer_dataset_name(predictions, args.dataset_name)
    selected_examples = select_correct_examples(predictions, args.max_examples, args.seed)
    run_dir = create_unique_run_dir(
        args.output_root,
        f"{dataset_slug(dataset_name)}_insertion_faithfulness",
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
    selected_counts = selected_examples.groupby("true_label").size()
    pd.DataFrame(
        {
            "true_label": MODEL_CLASS_NAMES,
            "selected_correct_examples": [
                int(selected_counts.get(label, 0)) for label in MODEL_CLASS_NAMES
            ],
        }
    ).to_csv(run_dir / "class_coverage.csv", index=False)

    model, processor, device = load_hubert_emotion_model(
        device=args.device,
        local_files_only=args.local_files_only,
    )
    records = []

    for example_index, example in selected_examples.iterrows():
        audio_path = Path(example["audio_path"])
        provenance = example_provenance(example)
        waveform, sampling_rate = load_audio_mono_16k(audio_path)
        score_by_mode, target_class, contrast_class = compute_explanation_scores(
            model,
            processor,
            waveform,
            sampling_rate,
            device,
        )
        original_logit, original_probability = target_scores(
            model,
            processor,
            waveform,
            sampling_rate,
            target_class,
            device,
        )
        silent_waveform = torch.zeros_like(waveform)
        silent_logit, silent_probability = target_scores(
            model,
            processor,
            silent_waveform,
            sampling_rate,
            target_class,
            device,
        )

        token_count = score_by_mode[modes[0]].numel()
        for mode in modes:
            if score_by_mode[mode].numel() != token_count:
                raise RuntimeError("All explanation modes must have the same token count.")

        for fraction_index, fraction in enumerate(fractions):
            random_token_sets = []
            selected_count = min(token_count, max(1, int(np.ceil(fraction * token_count))))
            for trial in range(args.random_trials):
                # Reuse the same random token windows across methods for a fair
                # top-vs-random insertion comparison.
                random_rng = np.random.default_rng(
                    np.random.SeedSequence([args.seed, example_index, fraction_index, trial])
                )
                random_token_sets.append(
                    random_rng.choice(token_count, size=selected_count, replace=False)
                )

            for mode in modes:
                scores = score_by_mode[mode].squeeze(0)
                for strategy in ("top", "bottom", "random"):
                    trial_count = args.random_trials if strategy == "random" else 1
                    for trial in range(trial_count):
                        if strategy == "random":
                            selected_tokens = random_token_sets[trial]
                        else:
                            selected_tokens = select_token_indices(
                                scores,
                                fraction,
                                strategy,
                                rng=None,
                            )
                        inserted_waveform = insert_token_regions(
                            waveform,
                            selected_tokens,
                            token_count,
                        )
                        inserted_logit, inserted_probability = target_scores(
                            model,
                            processor,
                            inserted_waveform,
                            sampling_rate,
                            target_class,
                            device,
                        )
                        records.append(
                            {
                                **provenance,
                                "example_index": example_index,
                                "true_class": int(example["true_class"]),
                                "true_label": str(example["true_label"]),
                                "predicted_class": int(example["predicted_class"]),
                                "target_class": target_class,
                                "contrast_class": contrast_class if mode in CONTRASTIVE_MODES else None,
                                "explanation_mode": mode,
                                "strategy": strategy,
                                "fraction": fraction,
                                "trial": trial,
                                "token_count": token_count,
                                "inserted_token_count": len(selected_tokens),
                                "original_target_logit": original_logit,
                                "silent_target_logit": silent_logit,
                                "inserted_target_logit": inserted_logit,
                                "logit_gain_from_silence": inserted_logit - silent_logit,
                                "logit_gap_to_original": original_logit - inserted_logit,
                                "logit_recovery": normalized_recovery(
                                    inserted_logit,
                                    silent_logit,
                                    original_logit,
                                ),
                                "original_target_probability": original_probability,
                                "silent_target_probability": silent_probability,
                                "inserted_target_probability": inserted_probability,
                                "probability_gain_from_silence": (
                                    inserted_probability - silent_probability
                                ),
                                "probability_gap_to_original": (
                                    original_probability - inserted_probability
                                ),
                                "probability_recovery": normalized_recovery(
                                    inserted_probability,
                                    silent_probability,
                                    original_probability,
                                ),
                            }
                        )

        # Add explicit zero-insertion rows for clean curve/AUC baselines.
        for mode in modes:
            for strategy in ("top", "bottom", "random"):
                trial_count = args.random_trials if strategy == "random" else 1
                for trial in range(trial_count):
                    records.append(
                        {
                            **provenance,
                            "example_index": example_index,
                            "true_class": int(example["true_class"]),
                            "true_label": str(example["true_label"]),
                            "predicted_class": int(example["predicted_class"]),
                            "target_class": target_class,
                            "contrast_class": (
                                contrast_class if mode in CONTRASTIVE_MODES else None
                            ),
                            "explanation_mode": mode,
                            "strategy": strategy,
                            "fraction": 0.0,
                            "trial": trial,
                            "token_count": token_count,
                            "inserted_token_count": 0,
                            "original_target_logit": original_logit,
                            "silent_target_logit": silent_logit,
                            "inserted_target_logit": silent_logit,
                            "logit_gain_from_silence": 0.0,
                            "logit_gap_to_original": original_logit - silent_logit,
                            "logit_recovery": 0.0,
                            "original_target_probability": original_probability,
                            "silent_target_probability": silent_probability,
                            "inserted_target_probability": silent_probability,
                            "probability_gain_from_silence": 0.0,
                            "probability_gap_to_original": (
                                original_probability - silent_probability
                            ),
                            "probability_recovery": 0.0,
                        }
                    )

        print(f"Inserted {example_index + 1}/{len(selected_examples)}: {audio_path.name}")

    record_frame = pd.DataFrame(records)
    summary = (
        record_frame.groupby(["explanation_mode", "strategy", "fraction"], as_index=False)
        .agg(
            mean_inserted_target_logit=("inserted_target_logit", "mean"),
            std_inserted_target_logit=("inserted_target_logit", "std"),
            mean_logit_gain_from_silence=("logit_gain_from_silence", "mean"),
            mean_logit_recovery=("logit_recovery", "mean"),
            std_logit_recovery=("logit_recovery", "std"),
            mean_inserted_target_probability=("inserted_target_probability", "mean"),
            std_inserted_target_probability=("inserted_target_probability", "std"),
            mean_probability_gain_from_silence=("probability_gain_from_silence", "mean"),
            mean_probability_recovery=("probability_recovery", "mean"),
            std_probability_recovery=("probability_recovery", "std"),
            num_trials=("inserted_target_probability", "size"),
        )
        .fillna(0)
    )
    auc_by_example, auc_by_method = compute_auc_tables(record_frame)
    robustness_by_class = (
        record_frame.groupby(
            ["true_label", "explanation_mode", "strategy", "fraction"],
            as_index=False,
        )
        .agg(
            examples=("audio_path", "nunique"),
            mean_inserted_target_probability=("inserted_target_probability", "mean"),
            std_inserted_target_probability=("inserted_target_probability", "std"),
            mean_probability_recovery=("probability_recovery", "mean"),
            std_probability_recovery=("probability_recovery", "std"),
            mean_inserted_target_logit=("inserted_target_logit", "mean"),
            std_inserted_target_logit=("inserted_target_logit", "std"),
            mean_logit_recovery=("logit_recovery", "mean"),
            std_logit_recovery=("logit_recovery", "std"),
            num_trials=("inserted_target_probability", "size"),
        )
        .fillna(0)
    )

    record_frame.to_csv(run_dir / "insertion_records.csv", index=False)
    summary.to_csv(run_dir / "insertion_summary.csv", index=False)
    auc_by_example.to_csv(run_dir / "insertion_auc_by_example.csv", index=False)
    auc_by_method.to_csv(run_dir / "insertion_auc_by_method.csv", index=False)
    robustness_by_class.to_csv(run_dir / "insertion_robustness_by_class.csv", index=False)
    plot_insertion_curves(summary, run_dir / "insertion_curves.png")
    write_json(
        run_dir / "config.json",
        {
            "dataset_name": dataset_name,
            "predictions_csv": str(Path(args.predictions_csv).resolve()),
            "audio_manifest": "audio_manifest.csv",
            "selected_examples_file": "selected_examples.csv",
            "insertion_records_file": "insertion_records.csv",
            "insertion_summary_file": "insertion_summary.csv",
            "insertion_auc_by_method_file": "insertion_auc_by_method.csv",
            "source_audio_column": "audio_path",
            "selection": "correct-only, approximately class-balanced",
            "max_examples": args.max_examples,
            "fractions": fractions,
            "random_trials": args.random_trials,
            "modes": modes,
            "seed": args.seed,
            "device": device,
            "local_files_only": args.local_files_only,
            "baseline": "silent waveform (zero samples)",
            "operation": "copy selected original waveform regions into the silent baseline",
            "token_to_waveform_mapping": "uniform temporal approximation",
            "metric_interpretation": (
                "Higher and earlier recovery for top-ranked regions indicates "
                "stronger insertion faithfulness."
            ),
        },
    )
    print("\nInsertion-faithfulness evaluation complete")
    print("Examples:", len(selected_examples))
    print("Records:", len(record_frame))
    print("Saved new run to:", run_dir)


if __name__ == "__main__":
    main()
