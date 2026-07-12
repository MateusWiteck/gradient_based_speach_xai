"""Compare linear head relevance with gradient × hidden-state relevance.

The compared maps are:

* ``head-relevance``: lightweight linear head/pooling contribution.
* ``gradient-hidden``: ``sum_d |H[t, d] * d logit_c / d H[t, d]|``.

The script records sparsity, map correlation, and deletion-faithfulness for
both token-level relevance scores.
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
from src.evaluation.explanation_metrics import (
    concentration_metrics,
    safe_correlation,
    summarize_sparsity_by_method,
)
from src.explainers.transformer_relevance.head_relevance import (
    compute_head_relevance,
    infer_pooling_type,
)
from src.explainers.transformer_relevance.hidden_gradient import (
    compute_gradient_hidden_relevance,
    extract_hidden_state_gradients,
)
from src.models.load_model import load_hubert_emotion_model
from src.utils.audio import load_audio_mono_16k


def select_token_indices(scores: torch.Tensor, fraction: float, strategy: str, rng) -> np.ndarray:
    """Select a fixed number of top, bottom, or random time-token indices."""
    if not 0 < fraction <= 1:
        raise ValueError("Deletion fractions must be in (0, 1].")
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
    raise ValueError(f"Unknown deletion strategy: {strategy}")


def mask_token_regions(
    waveform: torch.Tensor,
    token_indices: np.ndarray,
    token_count: int,
) -> torch.Tensor:
    """Silence uniformly mapped waveform regions belonging to selected tokens."""
    masked = waveform.clone()
    sample_count = masked.numel()
    sample_mask = torch.zeros(sample_count, dtype=torch.bool, device=masked.device)
    for token_index in token_indices:
        start = int(np.floor(token_index * sample_count / token_count))
        end = int(np.ceil((token_index + 1) * sample_count / token_count))
        sample_mask[start:end] = True
    masked[sample_mask] = 0
    return masked


def target_scores(
    model,
    processor,
    waveform,
    sampling_rate: int,
    target_class: int,
    device: str,
) -> tuple[float, float]:
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


def compute_head_and_gradient_hidden_scores(model, processor, waveform, sampling_rate, device):
    """Compute both head and gradient-hidden token relevance for one utterance."""
    gradient_result = extract_hidden_state_gradients(
        model=model,
        processor=processor,
        waveform=waveform,
        sampling_rate=sampling_rate,
        device=device,
    )
    target_class = gradient_result["target_class"]
    token_mask = gradient_result.get("feature_attention_mask")
    head_score = compute_head_relevance(
        last_hidden_states=gradient_result["head_input_hidden_states"],
        logits=gradient_result["logits"],
        target_class=target_class,
        model=model,
        pooling=infer_pooling_type(model),
        token_mask=token_mask,
    )
    gradient_hidden_score = compute_gradient_hidden_relevance(
        hidden_states=gradient_result["head_input_hidden_states"],
        hidden_gradients=gradient_result["head_input_gradients"],
        token_mask=token_mask,
    )
    return {
        "scores": {
            "head-relevance": head_score,
            "gradient-hidden": gradient_hidden_score,
        },
        "target_class": target_class,
        "logits": gradient_result["logits"],
        "predicted_class": gradient_result["predicted_class"],
    }


def map_comparison_metrics(head_scores: torch.Tensor, gradient_hidden_scores: torch.Tensor) -> dict:
    """Compare head relevance and gradient-hidden relevance for one audio."""
    head_values = head_scores.detach().cpu().numpy().reshape(-1)
    gradient_values = gradient_hidden_scores.detach().cpu().numpy().reshape(-1)
    return {
        "token_count": int(head_values.size),
        "pearson_correlation": safe_correlation(head_values, gradient_values),
        "spearman_correlation": safe_correlation(
            head_values,
            gradient_values,
            method="spearman",
        ),
        "mean_absolute_difference": float(np.mean(np.abs(head_values - gradient_values))),
        "l1_distance": float(np.abs(head_values - gradient_values).sum()),
    }


def plot_deletion_curves(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot mean target-logit drop by deletion fraction for both methods."""
    figure, axis = plt.subplots(figsize=(9, 5.2))
    for (mode, strategy), group in summary.groupby(["explanation_mode", "strategy"]):
        group = group.sort_values("fraction")
        axis.plot(
            group["fraction"],
            group["mean_logit_drop"],
            marker="o",
            label=f"{mode} / {strategy}",
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Fraction of token regions silenced")
    axis.set_ylabel("Mean drop in original target logit")
    axis.set_title("Head relevance vs gradient-hidden deletion")
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
    try:
        fractions = sorted(set(parse_csv_values(args.fractions, float, "fractions")))
    except ValueError as error:
        parser.error(str(error))
    if any(not 0 < fraction <= 1 for fraction in fractions):
        parser.error("--fractions values must be in (0, 1].")

    predictions = pd.read_csv(args.predictions_csv)
    dataset_name = infer_dataset_name(predictions, args.dataset_name)
    selected_examples = select_correct_examples(predictions, args.max_examples, args.seed)
    run_dir = create_unique_run_dir(
        args.output_root,
        f"{dataset_slug(dataset_name)}_hidden_gradient_relevance",
    )
    selected_examples.to_csv(run_dir / "selected_examples.csv", index=False)
    audio_manifest_columns = [
        column
        for column in (
            *PROVENANCE_COLUMNS,
            "true_class",
            "true_label",
            "predicted_class",
            "predicted_label",
        )
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
    sparsity_records = []
    comparison_records = []
    deletion_records = []

    for example_index, example in selected_examples.iterrows():
        audio_path = Path(example["audio_path"])
        provenance = example_provenance(example)
        waveform, sampling_rate = load_audio_mono_16k(audio_path)
        relevance = compute_head_and_gradient_hidden_scores(
            model,
            processor,
            waveform,
            sampling_rate,
            device,
        )
        score_by_mode = relevance["scores"]
        target_class = relevance["target_class"]
        original_logit, original_probability = target_scores(
            model,
            processor,
            waveform,
            sampling_rate,
            target_class,
            device,
        )

        comparison_records.append(
            {
                **provenance,
                "true_class": int(example["true_class"]),
                "true_label": str(example["true_label"]),
                "predicted_class": int(example["predicted_class"]),
                "target_class": target_class,
                **map_comparison_metrics(
                    score_by_mode["head-relevance"],
                    score_by_mode["gradient-hidden"],
                ),
            }
        )

        token_count = score_by_mode["head-relevance"].numel()
        for mode, scores in score_by_mode.items():
            sparsity_records.append(
                {
                    **provenance,
                    "true_class": int(example["true_class"]),
                    "true_label": str(example["true_label"]),
                    "predicted_class": int(example["predicted_class"]),
                    "target_class": target_class,
                    "explanation_mode": mode,
                    **concentration_metrics(scores),
                }
            )

        for fraction_index, fraction in enumerate(fractions):
            selected_count = min(token_count, max(1, int(np.ceil(fraction * token_count))))
            random_token_sets = []
            for trial in range(args.random_trials):
                random_rng = np.random.default_rng(
                    np.random.SeedSequence([args.seed, example_index, fraction_index, trial])
                )
                random_token_sets.append(
                    random_rng.choice(token_count, size=selected_count, replace=False)
                )

            for mode, scores in score_by_mode.items():
                scores = scores.squeeze(0)
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
                        masked_waveform = mask_token_regions(
                            waveform,
                            selected_tokens,
                            token_count,
                        )
                        masked_logit, masked_probability = target_scores(
                            model,
                            processor,
                            masked_waveform,
                            sampling_rate,
                            target_class,
                            device,
                        )
                        deletion_records.append(
                            {
                                **provenance,
                                "true_class": int(example["true_class"]),
                                "true_label": str(example["true_label"]),
                                "predicted_class": int(example["predicted_class"]),
                                "target_class": target_class,
                                "explanation_mode": mode,
                                "strategy": strategy,
                                "fraction": fraction,
                                "trial": trial,
                                "token_count": token_count,
                                "masked_token_count": len(selected_tokens),
                                "original_target_logit": original_logit,
                                "masked_target_logit": masked_logit,
                                "logit_drop": original_logit - masked_logit,
                                "original_target_probability": original_probability,
                                "masked_target_probability": masked_probability,
                                "probability_drop": original_probability - masked_probability,
                            }
                        )

        print(f"Compared hidden-gradient {example_index + 1}/{len(selected_examples)}: {audio_path.name}")

    sparsity_frame = pd.DataFrame(sparsity_records)
    comparison_frame = pd.DataFrame(comparison_records)
    deletion_frame = pd.DataFrame(deletion_records)
    sparsity_summary_by_method = summarize_sparsity_by_method(sparsity_frame)
    comparison_summary = pd.DataFrame(
        [
            {
                "examples": int(comparison_frame["audio_path"].nunique()),
                "mean_pearson_correlation": comparison_frame["pearson_correlation"].mean(),
                "std_pearson_correlation": comparison_frame["pearson_correlation"].std(),
                "mean_spearman_correlation": comparison_frame["spearman_correlation"].mean(),
                "std_spearman_correlation": comparison_frame["spearman_correlation"].std(),
                "mean_l1_distance": comparison_frame["l1_distance"].mean(),
                "std_l1_distance": comparison_frame["l1_distance"].std(),
                "mean_absolute_difference": comparison_frame[
                    "mean_absolute_difference"
                ].mean(),
                "std_absolute_difference": comparison_frame[
                    "mean_absolute_difference"
                ].std(),
            }
        ]
    ).fillna(0)
    deletion_summary = (
        deletion_frame.groupby(["explanation_mode", "strategy", "fraction"], as_index=False)
        .agg(
            mean_logit_drop=("logit_drop", "mean"),
            std_logit_drop=("logit_drop", "std"),
            mean_probability_drop=("probability_drop", "mean"),
            std_probability_drop=("probability_drop", "std"),
            num_trials=("logit_drop", "size"),
        )
        .fillna(0)
    )
    robustness_by_class = (
        deletion_frame.groupby(
            ["true_label", "explanation_mode", "strategy", "fraction"],
            as_index=False,
        )
        .agg(
            examples=("audio_path", "nunique"),
            mean_logit_drop=("logit_drop", "mean"),
            std_logit_drop=("logit_drop", "std"),
            mean_probability_drop=("probability_drop", "mean"),
            std_probability_drop=("probability_drop", "std"),
            num_trials=("logit_drop", "size"),
        )
        .fillna(0)
    )

    sparsity_frame.to_csv(run_dir / "sparsity_records.csv", index=False)
    sparsity_summary_by_method.to_csv(run_dir / "sparsity_summary_by_method.csv", index=False)
    comparison_frame.to_csv(run_dir / "head_vs_gradient_hidden_records.csv", index=False)
    comparison_summary.to_csv(run_dir / "head_vs_gradient_hidden_summary.csv", index=False)
    deletion_frame.to_csv(run_dir / "deletion_records.csv", index=False)
    deletion_summary.to_csv(run_dir / "deletion_summary.csv", index=False)
    robustness_by_class.to_csv(run_dir / "robustness_by_class.csv", index=False)
    plot_deletion_curves(deletion_summary, run_dir / "deletion_curves.png")
    write_json(
        run_dir / "config.json",
        {
            "dataset_name": dataset_name,
            "predictions_csv": str(Path(args.predictions_csv).resolve()),
            "audio_manifest": "audio_manifest.csv",
            "selected_examples_file": "selected_examples.csv",
            "methods": ["head-relevance", "gradient-hidden"],
            "gradient_hidden_formula": "sum_d abs(H[t,d] * d logit_c / d H[t,d])",
            "fractions": fractions,
            "random_trials": args.random_trials,
            "seed": args.seed,
            "device": device,
            "local_files_only": args.local_files_only,
            "mask_baseline": "silence (zero waveform samples)",
            "token_to_waveform_mapping": "uniform temporal approximation",
            "metric_interpretation": (
                "Compare concentration, head-vs-gradient-hidden correlation, "
                "and deletion top-vs-bottom/random drops."
            ),
        },
    )
    print("\nHidden-gradient relevance comparison complete")
    print("Examples:", len(selected_examples))
    print("Sparsity records:", len(sparsity_frame))
    print("Deletion records:", len(deletion_frame))
    print("Saved new run to:", run_dir)


if __name__ == "__main__":
    main()
