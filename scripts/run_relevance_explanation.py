"""Create timestamped HuBERT rollout and lightweight relevance explanations."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
from src.explainers.transformer_relevance.token_mapping import (
    temporal_relevance_to_dataframe,
)
from src.explainers.transformer_relevance.visualization import (
    plot_relevance_timeline,
)
from src.models.load_model import load_hubert_emotion_model
from src.utils.audio import load_audio_mono_16k


def make_unique_output_paths(output_root: str | Path, output_stem: str) -> tuple[Path, Path]:
    """Create timestamped output paths without replacing prior research runs."""
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for attempt in range(10_000):
        suffix = "" if attempt == 0 else f"_{attempt:04d}"
        base_path = output_dir / f"{output_stem}_{timestamp}{suffix}"
        csv_path = base_path.with_name(f"{base_path.name}_temporal.csv")
        plot_path = base_path.with_name(f"{base_path.name}_timeline.png")
        if not csv_path.exists() and not plot_path.exists():
            return csv_path, plot_path
    raise RuntimeError("Could not create unique output paths after 10,000 attempts.")


def _print_score_summary(name: str, scores: torch.Tensor) -> None:
    print(f"{name} shape:", tuple(scores.shape))
    print(f"{name} sum per sample:", scores.sum(dim=-1).detach().cpu().tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio",
        "-a",
        default="data/test_speech.wav",
        help="Audio file to explain (default: the bundled synthetic speech fixture).",
    )
    parser.add_argument(
        "--explanation-mode",
        "-m",
        choices=EXPLANATION_MODES,
        default="level3",
        help="Temporal relevance method to save.",
    )
    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        help="Class logit to explain (default: the model's predicted class).",
    )
    parser.add_argument(
        "--contrast-class",
        type=int,
        default=None,
        help="Competitor for contrastive modes (default: highest-logit non-target class).",
    )
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
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Directory for this test's timestamped CSV, PNG, and metadata JSON.",
    )
    args = parser.parse_args()

    contrastive_modes = {"level3-contrastive", "contrastive-conditioned-rollout"}
    if args.contrast_class is not None and args.explanation_mode not in contrastive_modes:
        parser.error("--contrast-class requires a contrastive explanation mode.")

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
    num_classes = base_result["logits"].shape[-1]
    if args.target_class is not None and not 0 <= args.target_class < num_classes:
        raise ValueError(
            f"--target-class must be in [0, {num_classes - 1}], got {args.target_class}."
        )
    if args.contrast_class is not None and not 0 <= args.contrast_class < num_classes:
        raise ValueError(
            f"--contrast-class must be in [0, {num_classes - 1}], got {args.contrast_class}."
        )

    grad_result = extract_gradient_weighted_attentions(
        model=model,
        processor=processor,
        waveform=waveform,
        sampling_rate=sampling_rate,
        device=device,
        target_class=args.target_class,
    )
    relevance = compute_relevance_scores(
        model=model,
        base_result=base_result,
        grad_result=grad_result,
        contrast_class=args.contrast_class,
    )
    final_score = relevance["scores"][args.explanation_mode]
    num_tokens = final_score.shape[-1]
    duration_seconds = waveform.shape[-1] / sampling_rate

    print("\nExplanation configuration:")
    print("Mode:", args.explanation_mode)
    print("Device:", device)
    print("Logits shape:", tuple(base_result["logits"].shape))
    print("Predicted class:", base_result["predicted_class"])
    print("Target class used:", relevance["target_class"])
    print("Contrast class(es):", relevance["contrast_classes"].detach().cpu().tolist())
    print("Number of gradient-attention layers:", len(grad_result["grad_attentions"]))
    print("Joint attention shape:", tuple(relevance["joint_attention"].shape))
    print("Raw rollout shape:", tuple(relevance["raw_rollout"].shape))
    _print_score_summary("Head relevance", relevance["head_relevance"])
    _print_score_summary("Contrastive head relevance", relevance["contrastive_head_relevance"])
    _print_score_summary("Final relevance", final_score)
    print("Number of time tokens:", num_tokens)
    print("Audio duration (s):", duration_seconds)

    if not torch.isfinite(final_score).all():
        raise RuntimeError("Final relevance contains non-finite values.")
    if not torch.allclose(
        final_score.sum(dim=-1),
        torch.ones(final_score.shape[0], device=final_score.device),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise RuntimeError("Final relevance must sum to 1 per sample.")

    dataframe = temporal_relevance_to_dataframe(
        temporal_relevance=final_score,
        audio_duration_seconds=duration_seconds,
    )
    audio_path = str(Path(args.audio).resolve())
    dataframe.insert(0, "audio_path", audio_path)
    dataframe.insert(1, "explanation_mode", args.explanation_mode)
    dataframe.insert(2, "target_class", relevance["target_class"])
    output_stem = args.explanation_mode.replace("-", "_")
    csv_path, plot_path = make_unique_output_paths(args.output_root, output_stem)
    metadata_path = csv_path.with_name(
        csv_path.name.replace("_temporal.csv", "_metadata.json")
    )
    dataframe.to_csv(csv_path, index=False)
    plot_relevance_timeline(
        waveform=waveform,
        sampling_rate=sampling_rate,
        df_relevance=dataframe,
        output_path=str(plot_path),
        relevance_title=args.explanation_mode.replace("-", " ").title(),
    )
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(
            {
                "audio_path": audio_path,
                "explanation_mode": args.explanation_mode,
                "device": device,
                "sampling_rate": sampling_rate,
                "duration_seconds": duration_seconds,
                "num_tokens": num_tokens,
                "predicted_class": base_result["predicted_class"],
                "target_class": relevance["target_class"],
                "contrast_classes": relevance["contrast_classes"].detach().cpu().tolist(),
                "temporal_csv": str(csv_path),
                "timeline_png": str(plot_path),
            },
            metadata_file,
            indent=2,
            sort_keys=True,
        )
        metadata_file.write("\n")
    print("Saved temporal relevance to:", csv_path)
    print("Saved timeline to:", plot_path)
    print("Saved metadata to:", metadata_path)


if __name__ == "__main__":
    main()
