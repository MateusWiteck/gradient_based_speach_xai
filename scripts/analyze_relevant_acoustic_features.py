"""Extract acoustic descriptors from relevance-selected audio regions.

For each correctly classified utterance, this script computes temporal
relevance maps, selects top/bottom/random token regions, merges them into
contiguous excerpts, and extracts lightweight acoustic features:

* pitch/F0 estimated by frame-wise autocorrelation;
* RMS energy;
* pause/silence statistics;
* a speaking-rate proxy based on energy-envelope peaks per second.

The speaking-rate value is intentionally named as a proxy because no transcript
or forced alignment is used.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.acoustic_features import (  # noqa: E402
    AcousticConfig,
    aggregate_acoustic_feature_rows,
    extract_acoustic_features,
    waveform_rms,
)
from src.evaluation.common import (  # noqa: E402
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
from src.explainers.transformer_relevance.attention_extractor import (  # noqa: E402
    extract_hubert_attentions,
)
from src.explainers.transformer_relevance.gradient_attention import (  # noqa: E402
    extract_gradient_weighted_attentions,
)
from src.explainers.transformer_relevance.hidden_gradient import (  # noqa: E402
    compute_gradient_hidden_relevance,
    extract_hidden_state_gradients,
)
from src.explainers.transformer_relevance.score_pipeline import (  # noqa: E402
    EXPLANATION_MODES,
    compute_relevance_scores,
)
from src.models.load_model import load_hubert_emotion_model  # noqa: E402
from src.utils.audio import load_audio_mono_16k  # noqa: E402


ACOUSTIC_SCORE_MODES = (*EXPLANATION_MODES, "head-relevance", "gradient-hidden")
DEFAULT_MODES = (
    "rollout",
    "level3",
    "level3-contrastive",
    "head-conditioned-rollout",
    "contrastive-conditioned-rollout",
    "head-relevance",
)
FEATURE_COLUMNS = (
    "duration_seconds",
    "rms_mean",
    "rms_std",
    "rms_max",
    "rms_dbfs_mean",
    "silence_fraction",
    "pause_count",
    "longest_pause_seconds",
    "voiced_fraction",
    "f0_mean_hz",
    "f0_median_hz",
    "f0_std_hz",
    "f0_min_hz",
    "f0_max_hz",
    "energy_peak_count",
    "speaking_rate_proxy_peaks_per_sec",
)


def compute_score_maps(
    model,
    processor,
    waveform: torch.Tensor,
    sampling_rate: int,
    device: str,
    modes: list[str],
) -> tuple[dict[str, torch.Tensor], int, int | None]:
    """Compute requested relevance maps for one utterance."""
    requested_modes = set(modes)
    shared_modes = requested_modes.intersection((*EXPLANATION_MODES, "head-relevance"))
    scores: dict[str, torch.Tensor] = {}
    target_class = None
    contrast_class = None

    if shared_modes:
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
        target_class = int(relevance["target_class"])
        contrast_class = int(relevance["contrast_classes"][0].item())
        for mode in EXPLANATION_MODES:
            if mode in requested_modes:
                scores[mode] = relevance["scores"][mode]
        if "head-relevance" in requested_modes:
            scores["head-relevance"] = relevance["head_relevance"]

    if "gradient-hidden" in requested_modes:
        hidden_result = extract_hidden_state_gradients(
            model=model,
            processor=processor,
            waveform=waveform,
            sampling_rate=sampling_rate,
            device=device,
            target_class=target_class,
        )
        target_class = int(hidden_result["target_class"])
        scores["gradient-hidden"] = compute_gradient_hidden_relevance(
            hidden_states=hidden_result["head_input_hidden_states"],
            hidden_gradients=hidden_result["head_input_gradients"],
            token_mask=hidden_result.get("feature_attention_mask"),
        )

    if target_class is None:
        raise RuntimeError("No relevance mode was computed.")
    return scores, target_class, contrast_class


def select_token_indices(
    scores: torch.Tensor,
    fraction: float,
    selection_type: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select top, bottom, or random token indices from a relevance vector."""
    if not 0 < fraction <= 1:
        raise ValueError("Fractions must be in (0, 1].")
    values = scores.detach().cpu().numpy().reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Scores must contain finite values.")

    selected_count = min(values.size, max(1, int(np.ceil(fraction * values.size))))
    if selection_type == "top":
        return np.argsort(values)[-selected_count:]
    if selection_type == "bottom":
        return np.argsort(values)[:selected_count]
    if selection_type == "random":
        return rng.choice(values.size, size=selected_count, replace=False)
    raise ValueError(f"Unknown selection type: {selection_type}")


def expand_token_indices(
    indices: np.ndarray,
    token_count: int,
    context_tokens: int,
) -> np.ndarray:
    """Expand selected tokens by a symmetric context window."""
    expanded: set[int] = set()
    for token_index in np.asarray(indices, dtype=int):
        start = max(0, int(token_index) - context_tokens)
        end = min(token_count, int(token_index) + context_tokens + 1)
        expanded.update(range(start, end))
    return np.array(sorted(expanded), dtype=int)


def merge_token_indices(
    indices: np.ndarray,
    token_count: int,
    audio_duration_seconds: float,
    merge_gap_tokens: int,
) -> list[dict[str, float | int]]:
    """Merge selected token indices into contiguous time intervals."""
    indices = np.array(sorted(set(int(index) for index in indices)), dtype=int)
    if indices.size == 0:
        return []

    intervals = []
    start_token = int(indices[0])
    previous_token = int(indices[0])
    for token_index in indices[1:]:
        token_index = int(token_index)
        if token_index <= previous_token + merge_gap_tokens + 1:
            previous_token = token_index
            continue
        intervals.append(
            token_interval_to_time(
                start_token,
                previous_token + 1,
                token_count,
                audio_duration_seconds,
            )
        )
        start_token = token_index
        previous_token = token_index

    intervals.append(
        token_interval_to_time(
            start_token,
            previous_token + 1,
            token_count,
            audio_duration_seconds,
        )
    )
    return intervals


def token_interval_to_time(
    start_token: int,
    end_token: int,
    token_count: int,
    audio_duration_seconds: float,
) -> dict[str, float | int]:
    """Convert a half-open token interval into start/end seconds."""
    start_time = start_token * audio_duration_seconds / token_count
    end_time = end_token * audio_duration_seconds / token_count
    return {
        "start_token": int(start_token),
        "end_token": int(end_token),
        "token_span": int(end_token - start_token),
        "start_time": float(start_time),
        "end_time": float(end_time),
        "segment_duration_seconds": float(max(0.0, end_time - start_time)),
    }


def slice_waveform_by_time(
    waveform: torch.Tensor,
    sample_rate: int,
    start_time: float,
    end_time: float,
) -> torch.Tensor:
    """Return a waveform slice for a time interval."""
    start_sample = max(0, int(np.floor(start_time * sample_rate)))
    end_sample = min(waveform.numel(), int(np.ceil(end_time * sample_rate)))
    if end_sample <= start_sample:
        end_sample = min(waveform.numel(), start_sample + 1)
    return waveform[start_sample:end_sample]


def summarize_by_method(selection_frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize acoustic features by method, fraction, and selection type."""
    if selection_frame.empty:
        return pd.DataFrame()

    aggregations = {
        "examples": ("audio_path", "nunique"),
        "records": ("audio_path", "size"),
        "mean_selected_token_count": ("selected_token_count", "mean"),
        "mean_expanded_token_count": ("expanded_token_count", "mean"),
        "mean_relevance_mass": ("selection_relevance_mass", "mean"),
    }
    for column in FEATURE_COLUMNS:
        aggregations[f"mean_{column}"] = (column, "mean")
        aggregations[f"std_{column}"] = (column, "std")

    return (
        selection_frame.groupby(
            ["explanation_mode", "selection_type", "fraction"],
            as_index=False,
        )
        .agg(**aggregations)
        .fillna(0)
    )


def summarize_by_class(selection_frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize acoustic features by emotion class and method."""
    if selection_frame.empty:
        return pd.DataFrame()

    aggregations = {
        "examples": ("audio_path", "nunique"),
        "records": ("audio_path", "size"),
        "mean_relevance_mass": ("selection_relevance_mass", "mean"),
    }
    for column in FEATURE_COLUMNS:
        aggregations[f"mean_{column}"] = (column, "mean")
        aggregations[f"std_{column}"] = (column, "std")

    return (
        selection_frame.groupby(
            ["true_label", "explanation_mode", "selection_type", "fraction"],
            as_index=False,
        )
        .agg(**aggregations)
        .fillna(0)
    )


def top_bottom_differences(selection_frame: pd.DataFrame) -> pd.DataFrame:
    """Compute top-minus-bottom acoustic differences for each example/method."""
    records = []
    index_columns = [
        column
        for column in (
            *PROVENANCE_COLUMNS,
            "true_class",
            "true_label",
            "predicted_class",
            "target_class",
            "explanation_mode",
            "fraction",
            "trial",
        )
        if column in selection_frame.columns
    ]
    for _, group in selection_frame.groupby(index_columns, dropna=False):
        top = group[group["selection_type"] == "top"]
        bottom = group[group["selection_type"] == "bottom"]
        if top.empty or bottom.empty:
            continue
        top_row = top.iloc[0]
        bottom_row = bottom.iloc[0]
        record = {column: top_row[column] for column in index_columns}
        for column in FEATURE_COLUMNS:
            record[f"top_minus_bottom_{column}"] = top_row[column] - bottom_row[column]
        records.append(record)
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Optional dataset name for output-folder provenance (inferred when available).",
    )
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--max-examples", type=int, default=8)
    parser.add_argument("--fractions", default="0.05,0.1")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--selection-types", default="top,bottom,random")
    parser.add_argument("--random-trials", type=int, default=3)
    parser.add_argument("--context-tokens", type=int, default=2)
    parser.add_argument("--merge-gap-tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--frame-ms", type=float, default=40.0)
    parser.add_argument("--hop-ms", type=float, default=10.0)
    parser.add_argument("--pitch-min-hz", type=float, default=50.0)
    parser.add_argument("--pitch-max-hz", type=float, default=500.0)
    parser.add_argument("--voicing-threshold", type=float, default=0.3)
    parser.add_argument("--silence-db-below-reference", type=float, default=35.0)
    parser.add_argument("--min-pause-ms", type=float, default=100.0)
    parser.add_argument("--min-peak-distance-ms", type=float, default=120.0)
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
    if args.context_tokens < 0 or args.merge_gap_tokens < 0:
        parser.error("--context-tokens and --merge-gap-tokens must be non-negative.")

    try:
        fractions = sorted(set(parse_csv_values(args.fractions, float, "fractions")))
        modes = parse_csv_values(args.modes, str, "modes")
        selection_types = parse_csv_values(args.selection_types, str, "selection-types")
    except ValueError as error:
        parser.error(str(error))

    invalid_modes = sorted(set(modes).difference(ACOUSTIC_SCORE_MODES))
    if invalid_modes:
        parser.error(f"Unsupported modes: {invalid_modes}")
    invalid_selection_types = sorted(set(selection_types).difference({"top", "bottom", "random"}))
    if invalid_selection_types:
        parser.error(f"Unsupported selection types: {invalid_selection_types}")
    if any(not 0 < fraction <= 1 for fraction in fractions):
        parser.error("--fractions values must be in (0, 1].")

    acoustic_config = AcousticConfig(
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        pitch_min_hz=args.pitch_min_hz,
        pitch_max_hz=args.pitch_max_hz,
        voicing_threshold=args.voicing_threshold,
        silence_db_below_reference=args.silence_db_below_reference,
        min_pause_ms=args.min_pause_ms,
        min_peak_distance_ms=args.min_peak_distance_ms,
    )

    predictions = pd.read_csv(args.predictions_csv)
    dataset_name = infer_dataset_name(predictions, args.dataset_name)
    selected_examples = select_correct_examples(predictions, args.max_examples, args.seed)
    run_dir = create_unique_run_dir(
        args.output_root,
        f"{dataset_slug(dataset_name)}_relevant_acoustic_features",
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

    segment_records = []
    selection_records = []
    full_audio_records = []

    for example_index, example in selected_examples.iterrows():
        audio_path = Path(example["audio_path"])
        provenance = example_provenance(example)
        waveform, sampling_rate = load_audio_mono_16k(audio_path)
        audio_duration_seconds = waveform.numel() / sampling_rate
        reference_rms = waveform_rms(waveform)

        full_audio_records.append(
            {
                **provenance,
                "true_class": int(example["true_class"]),
                "true_label": str(example["true_label"]),
                "predicted_class": int(example["predicted_class"]),
                "predicted_label": str(example["predicted_label"]),
                **extract_acoustic_features(
                    waveform,
                    sampling_rate,
                    reference_rms=reference_rms,
                    config=acoustic_config,
                ),
            }
        )

        score_by_mode, target_class, contrast_class = compute_score_maps(
            model=model,
            processor=processor,
            waveform=waveform,
            sampling_rate=sampling_rate,
            device=device,
            modes=modes,
        )
        token_count = next(iter(score_by_mode.values())).numel()
        for score in score_by_mode.values():
            if score.numel() != token_count:
                raise RuntimeError("All score maps must have the same token count.")

        for mode in modes:
            scores = score_by_mode[mode].squeeze(0)
            values = scores.detach().cpu().numpy().reshape(-1)
            for fraction_index, fraction in enumerate(fractions):
                for selection_type in selection_types:
                    trial_count = args.random_trials if selection_type == "random" else 1
                    for trial in range(trial_count):
                        rng = np.random.default_rng(
                            np.random.SeedSequence(
                                [args.seed, example_index, fraction_index, trial]
                            )
                        )
                        selected_tokens = select_token_indices(
                            scores,
                            fraction,
                            selection_type,
                            rng,
                        )
                        expanded_tokens = expand_token_indices(
                            selected_tokens,
                            token_count,
                            args.context_tokens,
                        )
                        intervals = merge_token_indices(
                            expanded_tokens,
                            token_count,
                            audio_duration_seconds,
                            args.merge_gap_tokens,
                        )

                        current_segment_rows = []
                        for segment_index, interval in enumerate(intervals):
                            segment_waveform = slice_waveform_by_time(
                                waveform,
                                sampling_rate,
                                float(interval["start_time"]),
                                float(interval["end_time"]),
                            )
                            acoustic_features = extract_acoustic_features(
                                segment_waveform,
                                sampling_rate,
                                reference_rms=reference_rms,
                                config=acoustic_config,
                            )
                            segment_relevance_mass = float(
                                values[
                                    int(interval["start_token"]) : int(interval["end_token"])
                                ].sum()
                            )
                            row = {
                                **provenance,
                                "true_class": int(example["true_class"]),
                                "true_label": str(example["true_label"]),
                                "predicted_class": int(example["predicted_class"]),
                                "predicted_label": str(example["predicted_label"]),
                                "target_class": target_class,
                                "contrast_class": contrast_class,
                                "explanation_mode": mode,
                                "selection_type": selection_type,
                                "fraction": fraction,
                                "trial": trial,
                                "segment_index": segment_index,
                                "token_count": token_count,
                                "selected_token_count": int(len(selected_tokens)),
                                "expanded_token_count": int(len(expanded_tokens)),
                                "selection_relevance_mass": float(values[selected_tokens].sum()),
                                "segment_relevance_mass": segment_relevance_mass,
                                **interval,
                                **acoustic_features,
                            }
                            segment_records.append(row)
                            current_segment_rows.append(row)

                        aggregated_features = aggregate_acoustic_feature_rows(
                            current_segment_rows
                        )
                        selection_records.append(
                            {
                                **provenance,
                                "true_class": int(example["true_class"]),
                                "true_label": str(example["true_label"]),
                                "predicted_class": int(example["predicted_class"]),
                                "predicted_label": str(example["predicted_label"]),
                                "target_class": target_class,
                                "contrast_class": contrast_class,
                                "explanation_mode": mode,
                                "selection_type": selection_type,
                                "fraction": fraction,
                                "trial": trial,
                                "token_count": token_count,
                                "selected_token_count": int(len(selected_tokens)),
                                "expanded_token_count": int(len(expanded_tokens)),
                                "merged_segment_count": int(len(intervals)),
                                "selection_relevance_mass": float(values[selected_tokens].sum()),
                                "expanded_relevance_mass": float(values[expanded_tokens].sum()),
                                **aggregated_features,
                            }
                        )

        print(
            f"Extracted acoustic features {example_index + 1}/"
            f"{len(selected_examples)}: {audio_path.name}"
        )

    segment_frame = pd.DataFrame(segment_records)
    selection_frame = pd.DataFrame(selection_records)
    full_audio_frame = pd.DataFrame(full_audio_records)
    summary_by_method = summarize_by_method(selection_frame)
    summary_by_class = summarize_by_class(selection_frame)
    differences = top_bottom_differences(selection_frame)
    if not differences.empty:
        diff_aggregations = {
            "examples": ("audio_path", "nunique"),
            "records": ("audio_path", "size"),
        }
        for column in FEATURE_COLUMNS:
            diff_column = f"top_minus_bottom_{column}"
            diff_aggregations[f"mean_{diff_column}"] = (diff_column, "mean")
            diff_aggregations[f"std_{diff_column}"] = (diff_column, "std")
        differences_summary = (
            differences.groupby(["explanation_mode", "fraction"], as_index=False)
            .agg(**diff_aggregations)
            .fillna(0)
        )
    else:
        differences_summary = pd.DataFrame()

    segment_frame.to_csv(run_dir / "acoustic_segment_records.csv", index=False)
    selection_frame.to_csv(run_dir / "acoustic_selection_records.csv", index=False)
    full_audio_frame.to_csv(run_dir / "acoustic_full_audio_records.csv", index=False)
    summary_by_method.to_csv(run_dir / "acoustic_summary_by_method.csv", index=False)
    summary_by_class.to_csv(run_dir / "acoustic_summary_by_class.csv", index=False)
    differences.to_csv(run_dir / "acoustic_top_bottom_differences.csv", index=False)
    differences_summary.to_csv(
        run_dir / "acoustic_top_bottom_differences_summary.csv",
        index=False,
    )

    write_json(
        run_dir / "config.json",
        {
            "dataset_name": dataset_name,
            "predictions_csv": str(Path(args.predictions_csv).resolve()),
            "audio_manifest": "audio_manifest.csv",
            "selected_examples_file": "selected_examples.csv",
            "modes": modes,
            "fractions": fractions,
            "selection_types": selection_types,
            "random_trials": args.random_trials,
            "context_tokens": args.context_tokens,
            "merge_gap_tokens": args.merge_gap_tokens,
            "seed": args.seed,
            "device": device,
            "local_files_only": args.local_files_only,
            "feature_notes": {
                "pitch": "Frame-wise autocorrelation F0 estimate.",
                "energy": "Frame-wise RMS amplitude.",
                "pauses": (
                    "Frames below a threshold relative to each full utterance "
                    "RMS are counted as silent."
                ),
                "speaking_rate_proxy": (
                    "Energy-envelope peaks per second; not true words/s or "
                    "syllables/s because no transcript/alignment is used."
                ),
                "token_to_time_mapping": "Uniform temporal token approximation.",
            },
            "acoustic_config": acoustic_config.__dict__,
        },
    )

    print("\nRelevant acoustic feature analysis complete")
    print("Examples:", len(selected_examples))
    print("Segment records:", len(segment_frame))
    print("Selection records:", len(selection_frame))
    print("Saved new run to:", run_dir)


if __name__ == "__main__":
    main()
