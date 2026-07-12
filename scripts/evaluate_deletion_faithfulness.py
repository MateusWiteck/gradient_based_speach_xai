"""Compare temporal explanation modes with deletion-faithfulness tests.

The script accepts a standardized prediction CSV from either dataset evaluator.
It evaluates only correctly classified utterances, masks uniformly mapped raw
audio regions for highest/lowest/random relevance tokens, and records the drop
in the original target-class logit. Each run creates a new output folder.
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


CONTRASTIVE_MODES = {"level3-contrastive", "contrastive-conditioned-rollout"}


def compute_explanation_scores(
    model,
    processor,
    waveform,
    sampling_rate: int,
    device: str,
    *,
    include_legrad: bool,
):
    """Compute every shared score variant from one audio utterance."""
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
    legrad_result = None
    if include_legrad:
        legrad_result = extract_legrad_hubert_relevance(
            model=model,
            processor=processor,
            waveform=waveform,
            sampling_rate=sampling_rate,
            device=device,
            target_class=grad_result["target_class"],
        )
    relevance = compute_relevance_scores(
        model,
        base_result,
        grad_result,
        legrad_result=legrad_result,
    )
    return (
        relevance["scores"],
        relevance["target_class"],
        int(relevance["contrast_classes"][0].item()),
    )


def select_token_indices(scores: torch.Tensor, fraction: float, strategy: str, rng) -> np.ndarray:
    """Select a fixed number of top, bottom, or random time-token indices."""
    if not 0 < fraction <= 1:
        raise ValueError("Deletion fractions must be in (0, 1].")
    values = scores.detach().cpu().numpy().reshape(-1)
    token_count = values.size
    selected_count = min(token_count, max(1, int(np.ceil(fraction * token_count))))

    if strategy == "top":
        return np.argsort(values)[-selected_count:]
    if strategy == "bottom":
        return np.argsort(values)[:selected_count]
    if strategy == "random":
        return rng.choice(token_count, size=selected_count, replace=False)
    raise ValueError(f"Unknown deletion strategy: {strategy}")


def mask_token_regions(waveform: torch.Tensor, token_indices: np.ndarray, token_count: int) -> torch.Tensor:
    """Silence uniformly mapped waveform regions belonging to selected tokens.

    This follows the project's existing token-to-time approximation. HuBERT's
    convolutional receptive fields overlap, so the mapping is intentionally
    documented as approximate rather than presented as exact frame alignment.
    """
    masked = waveform.clone()
    sample_count = masked.numel()
    sample_mask = torch.zeros(sample_count, dtype=torch.bool, device=masked.device)
    for token_index in token_indices:
        start = int(np.floor(token_index * sample_count / token_count))
        end = int(np.ceil((token_index + 1) * sample_count / token_count))
        sample_mask[start:end] = True
    masked[sample_mask] = 0
    return masked


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


def plot_deletion_curves(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot mean target-logit drop by deletion fraction for every method."""
    figure, axis = plt.subplots(figsize=(10, 5.5))
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
    axis.set_title("Deletion-faithfulness curves")
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
    parser.add_argument(
        "--max-examples",
        type=int,
        default=8,
        help="Maximum correct examples to evaluate (default: 8).",
    )
    parser.add_argument(
        "--all-examples",
        action="store_true",
        help="Evaluate every correctly classified example; overrides --max-examples.",
    )
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
    include_legrad = any(mode in LEGRAD_HUBERT_EXPLANATION_MODES for mode in modes)
    if any(not 0 < fraction <= 1 for fraction in fractions):
        parser.error("--fractions values must be in (0, 1].")

    predictions = pd.read_csv(args.predictions_csv)
    dataset_name = infer_dataset_name(predictions, args.dataset_name)
    max_examples = (
        int(predictions["is_correct"].astype(bool).sum())
        if args.all_examples
        else args.max_examples
    )
    selected_examples = select_correct_examples(predictions, max_examples, args.seed)
    run_dir = create_unique_run_dir(
        args.output_root,
        f"{dataset_slug(dataset_name)}_deletion_faithfulness",
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
            "selected_correct_examples": [int(selected_counts.get(label, 0)) for label in MODEL_CLASS_NAMES],
        }
    ).to_csv(run_dir / "class_coverage.csv", index=False)

    model, processor, device = load_hubert_emotion_model(
        device=args.device,
        local_files_only=args.local_files_only,
    )
    records = []
    sparsity_records = []

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
            include_legrad=include_legrad,
        )
        original_logit, original_probability = target_scores(
            model, processor, waveform, sampling_rate, target_class, device
        )

        token_count = score_by_mode[modes[0]].numel()
        for mode in modes:
            sparsity_records.append(
                {
                    **provenance,
                    "true_class": int(example["true_class"]),
                    "true_label": str(example["true_label"]),
                    "predicted_class": int(example["predicted_class"]),
                    "target_class": target_class,
                    "contrast_class": contrast_class if mode in CONTRASTIVE_MODES else None,
                    "explanation_mode": mode,
                    **concentration_metrics(score_by_mode[mode]),
                }
            )
        for fraction_index, fraction in enumerate(fractions):
            random_token_sets = []
            selected_count = min(token_count, max(1, int(np.ceil(fraction * token_count))))
            for trial in range(args.random_trials):
                # Keep random windows identical across methods for a fair
                # top-vs-random comparison on this utterance and fraction.
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
                        masked_waveform = mask_token_regions(waveform, selected_tokens, token_count)
                        masked_logit, masked_probability = target_scores(
                            model,
                            processor,
                            masked_waveform,
                            sampling_rate,
                            target_class,
                            device,
                        )
                        records.append(
                            {
                                **provenance,
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
                                "masked_token_count": len(selected_tokens),
                                "original_target_logit": original_logit,
                                "masked_target_logit": masked_logit,
                                "logit_drop": original_logit - masked_logit,
                                "original_target_probability": original_probability,
                                "masked_target_probability": masked_probability,
                                "probability_drop": original_probability - masked_probability,
                            }
                        )

        print(f"Explained {example_index + 1}/{len(selected_examples)}: {audio_path.name}")

    record_frame = pd.DataFrame(records)
    summary = (
        record_frame.groupby(["explanation_mode", "strategy", "fraction"], as_index=False)
        .agg(
            mean_logit_drop=("logit_drop", "mean"),
            std_logit_drop=("logit_drop", "std"),
            mean_probability_drop=("probability_drop", "mean"),
            num_trials=("logit_drop", "size"),
        )
        .fillna(0)
    )
    sparsity_frame = pd.DataFrame(sparsity_records)
    sparsity_summary = (
        sparsity_frame.groupby(["true_label", "explanation_mode"], as_index=False)
        .agg(
            examples=("audio_path", "size"),
            mean_normalized_entropy=("normalized_entropy", "mean"),
            mean_effective_tokens=("effective_tokens", "mean"),
            mean_gini=("gini", "mean"),
            mean_top_5_percent_mass=("top_5_percent_mass", "mean"),
            mean_top_10_percent_mass=("top_10_percent_mass", "mean"),
        )
        .fillna(0)
    )
    sparsity_summary_by_method = summarize_sparsity_by_method(sparsity_frame)
    robustness_by_class = (
        record_frame.groupby(
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
    record_frame.to_csv(run_dir / "deletion_records.csv", index=False)
    summary.to_csv(run_dir / "deletion_summary.csv", index=False)
    sparsity_frame.to_csv(run_dir / "sparsity_records.csv", index=False)
    sparsity_summary_by_method.to_csv(run_dir / "sparsity_summary_by_method.csv", index=False)
    sparsity_summary.to_csv(run_dir / "sparsity_summary_by_class.csv", index=False)
    robustness_by_class.to_csv(run_dir / "robustness_by_class.csv", index=False)
    plot_deletion_curves(summary, run_dir / "deletion_curves.png")
    write_json(
        run_dir / "config.json",
        {
            "dataset_name": dataset_name,
            "predictions_csv": str(Path(args.predictions_csv).resolve()),
            "audio_manifest": "audio_manifest.csv",
            "selected_examples_file": "selected_examples.csv",
            "deletion_records_file": "deletion_records.csv",
            "sparsity_summary_by_method_file": "sparsity_summary_by_method.csv",
            "source_audio_column": "audio_path",
            "selection": "correct-only, approximately class-balanced",
            "max_examples": args.max_examples,
            "fractions": fractions,
            "random_trials": args.random_trials,
            "modes": modes,
            "seed": args.seed,
            "device": device,
            "local_files_only": args.local_files_only,
            "mask_baseline": "silence (zero waveform samples)",
            "token_to_waveform_mapping": "uniform temporal approximation",
            "concentration_metrics": [
                "normalized_entropy",
                "effective_tokens",
                "gini",
                "top_5_percent_mass",
                "top_10_percent_mass",
            ],
        },
    )
    print("\nDeletion-faithfulness evaluation complete")
    print("Examples:", len(selected_examples))
    print("Records:", len(record_frame))
    print("Sparsity records:", len(sparsity_frame))
    print("Saved new run to:", run_dir)


if __name__ == "__main__":
    main()
