"""Generate predicted-class and runner-up-class relevance heatmaps for one audio."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.common import create_unique_run_dir, write_json
from src.evaluation.explanation_metrics import class_specificity_metrics
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
from src.explainers.transformer_relevance.visualization import (
    plot_class_relevance_heatmaps,
)
from src.models.load_model import load_hubert_emotion_model
from src.utils.audio import load_audio_mono_16k


def runner_up_class(logits, predicted_class: int) -> int:
    """Return the highest-logit class other than the predicted class."""
    candidates = logits.detach().clone()
    candidates[:, predicted_class] = -float("inf")
    return int(candidates.argmax(dim=-1).item())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True)
    parser.add_argument(
        "--explanation-mode",
        choices=EXPLANATION_MODES,
        default="contrastive-conditioned-rollout",
        help="Score variant to compare for the two target classes.",
    )
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    model, processor, device = load_hubert_emotion_model(
        device=args.device,
        local_files_only=args.local_files_only,
    )
    waveform, sampling_rate = load_audio_mono_16k(args.audio)
    base_result = extract_hubert_attentions(
        model=model,
        processor=processor,
        waveform=waveform,
        sampling_rate=sampling_rate,
        device=device,
    )
    predicted_class = base_result["predicted_class"]
    runner_up = runner_up_class(base_result["logits"], predicted_class)
    class_names = [str(model.config.id2label[index]) for index in range(model.config.num_labels)]
    include_legrad = args.explanation_mode in LEGRAD_HUBERT_EXPLANATION_MODES

    relevance_by_label = {}
    rows = []
    contrast_by_target = {}
    score_by_target = {}
    for target_class in (predicted_class, runner_up):
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
        relevance = compute_relevance_scores(
            model,
            base_result,
            grad_result,
            legrad_result=legrad_result,
        )
        label = f"{class_names[target_class]} (class {target_class})"
        relevance_by_label[label] = relevance["scores"][args.explanation_mode].squeeze(0)
        score_by_target[target_class] = relevance_by_label[label]
        contrast_by_target[str(target_class)] = int(relevance["contrast_classes"][0].item())

        token_count = relevance["scores"][args.explanation_mode].shape[-1]
        duration = waveform.numel() / sampling_rate
        for token_index, score in enumerate(relevance_by_label[label].detach().cpu().tolist()):
            rows.append(
                {
                    "audio_path": str(Path(args.audio).resolve()),
                    "target_class": target_class,
                    "target_label": class_names[target_class],
                    "token_idx": token_index,
                    "start_time": token_index * duration / token_count,
                    "end_time": (token_index + 1) * duration / token_count,
                    "relevance": score,
                }
            )

    run_dir = create_unique_run_dir(args.output_root, "class_specificity")
    pd.DataFrame(rows).to_csv(run_dir / "class_specificity_scores.csv", index=False)
    specificity_metrics = class_specificity_metrics(
        score_by_target[predicted_class],
        score_by_target[runner_up],
    )
    pd.DataFrame(
        [
            {
                "audio_path": str(Path(args.audio).resolve()),
                "explanation_mode": args.explanation_mode,
                "predicted_class": predicted_class,
                "predicted_label": class_names[predicted_class],
                "runner_up_class": runner_up,
                "runner_up_label": class_names[runner_up],
                **specificity_metrics,
            }
        ]
    ).to_csv(run_dir / "class_specificity_summary.csv", index=False)
    plot_class_relevance_heatmaps(
        waveform=waveform,
        sampling_rate=sampling_rate,
        relevance_by_label=relevance_by_label,
        output_path=str(run_dir / "class_specificity_heatmap.png"),
        title=f"{args.explanation_mode}: predicted vs runner-up",
    )
    write_json(
        run_dir / "metadata.json",
        {
            "audio_path": str(Path(args.audio).resolve()),
            "explanation_mode": args.explanation_mode,
            "device": device,
            "predicted_class": predicted_class,
            "predicted_label": class_names[predicted_class],
            "runner_up_class": runner_up,
            "runner_up_label": class_names[runner_up],
            "class_specificity_metrics": specificity_metrics,
            "contrast_class_by_target": contrast_by_target,
            "local_files_only": args.local_files_only,
        },
    )
    print("Predicted class:", predicted_class, class_names[predicted_class])
    print("Runner-up class:", runner_up, class_names[runner_up])
    print("Pearson correlation:", specificity_metrics["pearson_correlation"])
    print("Spearman correlation:", specificity_metrics["spearman_correlation"])
    print("Saved new class-specific comparison to:", run_dir)


if __name__ == "__main__":
    main()
