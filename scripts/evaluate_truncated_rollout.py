"""Evaluate truncated gradient-attention rollout using only the last k layers.

Full attention rollout across all HuBERT layers can become very diffuse. This
script tests a targeted alternative: compute rollout only over the final
``k`` Transformer layers, then evaluate whether those maps are more
concentrated and more deletion-faithful.
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
    summarize_sparsity_by_method,
)
from src.explainers.transformer_relevance.gradient_attention import (
    extract_gradient_weighted_attentions,
)
from src.explainers.transformer_relevance.rollout import (
    compute_rollout_attention,
    rollout_to_temporal_relevance,
)
from src.explainers.transformer_relevance.score_pipeline import normalize_token_scores
from src.models.load_model import load_hubert_emotion_model
from src.utils.audio import load_audio_mono_16k


def compute_truncated_rollout_scores(
    grad_attentions: list[torch.Tensor],
    last_k_values: list[int],
) -> dict[str, dict]:
    """Compute normalized temporal rollout scores for each requested last-k.

    Returns a dictionary keyed by labels such as ``rollout-last-4``. Each value
    stores the score tensor plus metadata about how many layers were used.
    """
    total_layers = len(grad_attentions)
    if total_layers == 0:
        raise ValueError("No gradient-weighted attention layers were returned.")

    score_by_variant = {}
    for requested_last_k in last_k_values:
        if requested_last_k < 1:
            raise ValueError("last-k values must be positive.")
        if requested_last_k > total_layers:
            raise ValueError(
                f"Cannot use last {requested_last_k} layers; model returned only "
                f"{total_layers} attention layers."
            )

        start_layer = total_layers - requested_last_k
        joint_attention = compute_rollout_attention(
            grad_attentions,
            start_layer=start_layer,
        )
        raw_rollout = rollout_to_temporal_relevance(joint_attention, strategy="mean")
        score = normalize_token_scores(raw_rollout)
        score_by_variant[f"rollout-last-{requested_last_k}"] = {
            "scores": score,
            "requested_last_k": requested_last_k,
            "layers_used": requested_last_k,
            "total_layers": total_layers,
            "start_layer": start_layer,
        }

    return score_by_variant


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


def plot_deletion_curves(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot mean target-logit drop by deletion fraction for every variant."""
    figure, axis = plt.subplots(figsize=(10, 5.5))
    for (variant, strategy), group in summary.groupby(["explanation_mode", "strategy"]):
        group = group.sort_values("fraction")
        axis.plot(
            group["fraction"],
            group["mean_logit_drop"],
            marker="o",
            label=f"{variant} / {strategy}",
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Fraction of token regions silenced")
    axis.set_ylabel("Mean drop in original target logit")
    axis.set_title("Truncated rollout deletion-faithfulness curves")
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
    parser.add_argument(
        "--last-k-layers",
        default="1,2,4,6",
        help="Comma-separated final-layer counts to use for rollout.",
    )
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
        last_k_values = sorted(set(parse_csv_values(args.last_k_layers, int, "last-k-layers")))
        fractions = sorted(set(parse_csv_values(args.fractions, float, "fractions")))
    except ValueError as error:
        parser.error(str(error))
    if any(value < 1 for value in last_k_values):
        parser.error("--last-k-layers values must be positive integers.")
    if any(not 0 < fraction <= 1 for fraction in fractions):
        parser.error("--fractions values must be in (0, 1].")

    predictions = pd.read_csv(args.predictions_csv)
    dataset_name = infer_dataset_name(predictions, args.dataset_name)
    selected_examples = select_correct_examples(predictions, args.max_examples, args.seed)
    run_dir = create_unique_run_dir(
        args.output_root,
        f"{dataset_slug(dataset_name)}_truncated_rollout",
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
    deletion_records = []
    sparsity_records = []

    for example_index, example in selected_examples.iterrows():
        audio_path = Path(example["audio_path"])
        provenance = example_provenance(example)
        waveform, sampling_rate = load_audio_mono_16k(audio_path)
        grad_result = extract_gradient_weighted_attentions(
            model=model,
            processor=processor,
            waveform=waveform,
            sampling_rate=sampling_rate,
            device=device,
        )
        target_class = grad_result["target_class"]
        original_logit, original_probability = target_scores(
            model,
            processor,
            waveform,
            sampling_rate,
            target_class,
            device,
        )
        score_by_variant = compute_truncated_rollout_scores(
            grad_result["grad_attentions"],
            last_k_values,
        )
        first_variant = next(iter(score_by_variant.values()))
        token_count = first_variant["scores"].numel()

        for variant_name, variant in score_by_variant.items():
            scores = variant["scores"]
            sparsity_records.append(
                {
                    **provenance,
                    "true_class": int(example["true_class"]),
                    "true_label": str(example["true_label"]),
                    "predicted_class": int(example["predicted_class"]),
                    "target_class": target_class,
                    "explanation_mode": variant_name,
                    "requested_last_k": variant["requested_last_k"],
                    "layers_used": variant["layers_used"],
                    "total_layers": variant["total_layers"],
                    "start_layer": variant["start_layer"],
                    **concentration_metrics(scores),
                }
            )

        for fraction_index, fraction in enumerate(fractions):
            selected_count = min(token_count, max(1, int(np.ceil(fraction * token_count))))
            random_token_sets = []
            for trial in range(args.random_trials):
                # Keep random token windows identical across variants for fair
                # top-vs-random comparisons on this utterance/fraction.
                random_rng = np.random.default_rng(
                    np.random.SeedSequence([args.seed, example_index, fraction_index, trial])
                )
                random_token_sets.append(
                    random_rng.choice(token_count, size=selected_count, replace=False)
                )

            for variant_name, variant in score_by_variant.items():
                scores = variant["scores"].squeeze(0)
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
                                "explanation_mode": variant_name,
                                "requested_last_k": variant["requested_last_k"],
                                "layers_used": variant["layers_used"],
                                "total_layers": variant["total_layers"],
                                "start_layer": variant["start_layer"],
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

        print(
            f"Evaluated truncated rollout {example_index + 1}/"
            f"{len(selected_examples)}: {audio_path.name}"
        )

    deletion_frame = pd.DataFrame(deletion_records)
    deletion_summary = (
        deletion_frame.groupby(["explanation_mode", "strategy", "fraction"], as_index=False)
        .agg(
            requested_last_k=("requested_last_k", "first"),
            layers_used=("layers_used", "first"),
            start_layer=("start_layer", "first"),
            mean_logit_drop=("logit_drop", "mean"),
            std_logit_drop=("logit_drop", "std"),
            mean_probability_drop=("probability_drop", "mean"),
            std_probability_drop=("probability_drop", "std"),
            num_trials=("logit_drop", "size"),
        )
        .fillna(0)
    )
    sparsity_frame = pd.DataFrame(sparsity_records)
    sparsity_summary_by_method = summarize_sparsity_by_method(sparsity_frame)
    robustness_by_class = (
        deletion_frame.groupby(
            ["true_label", "explanation_mode", "strategy", "fraction"],
            as_index=False,
        )
        .agg(
            requested_last_k=("requested_last_k", "first"),
            examples=("audio_path", "nunique"),
            mean_logit_drop=("logit_drop", "mean"),
            std_logit_drop=("logit_drop", "std"),
            mean_probability_drop=("probability_drop", "mean"),
            std_probability_drop=("probability_drop", "std"),
            num_trials=("logit_drop", "size"),
        )
        .fillna(0)
    )

    deletion_frame.to_csv(run_dir / "deletion_records.csv", index=False)
    deletion_summary.to_csv(run_dir / "deletion_summary.csv", index=False)
    sparsity_frame.to_csv(run_dir / "sparsity_records.csv", index=False)
    sparsity_summary_by_method.to_csv(run_dir / "sparsity_summary_by_method.csv", index=False)
    robustness_by_class.to_csv(run_dir / "robustness_by_class.csv", index=False)
    plot_deletion_curves(deletion_summary, run_dir / "deletion_curves.png")
    write_json(
        run_dir / "config.json",
        {
            "dataset_name": dataset_name,
            "predictions_csv": str(Path(args.predictions_csv).resolve()),
            "audio_manifest": "audio_manifest.csv",
            "selected_examples_file": "selected_examples.csv",
            "deletion_records_file": "deletion_records.csv",
            "deletion_summary_file": "deletion_summary.csv",
            "sparsity_records_file": "sparsity_records.csv",
            "sparsity_summary_by_method_file": "sparsity_summary_by_method.csv",
            "source_audio_column": "audio_path",
            "selection": "correct-only, approximately class-balanced",
            "max_examples": args.max_examples,
            "last_k_layers": last_k_values,
            "fractions": fractions,
            "random_trials": args.random_trials,
            "seed": args.seed,
            "device": device,
            "local_files_only": args.local_files_only,
            "mask_baseline": "silence (zero waveform samples)",
            "token_to_waveform_mapping": "uniform temporal approximation",
            "metric_interpretation": (
                "Lower entropy/higher Gini indicates reduced diffusion; larger "
                "top-deletion drops indicate stronger deletion faithfulness."
            ),
        },
    )
    print("\nTruncated-rollout evaluation complete")
    print("Examples:", len(selected_examples))
    print("Deletion records:", len(deletion_frame))
    print("Sparsity records:", len(sparsity_frame))
    print("Saved new run to:", run_dir)


if __name__ == "__main__":
    main()
