"""Run duration-matched SpeechXAI/Pastor vs HuBERT explanation evaluation.

By default the script evaluates every compatible audio in both IEMOCAP and
RAVDESS. SpeechXAI top-k words define the masked duration for each k, and each
HuBERT explanation mode plus the random deletion by silence masking baseline
receives that same duration.
"""

from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
import platform
from pathlib import Path
import subprocess
import sys
import traceback

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.common import (  # noqa: E402
    MODEL_CLASS_NAMES,
    create_unique_run_dir,
    dataset_slug,
    parse_csv_values,
    write_json,
)
from src.evaluation.duration_matched_speechxai import (  # noqa: E402
    DEFAULT_TOP_K_VALUES,
    AudioEvaluationExample,
    SpeechXAITranscriber,
    classify_waveform,
    disable_huggingface_symlink_cache_on_windows,
    evaluate_duration_matched_audio,
    summarize_duration_matched_records,
)
from src.evaluation.iemocap import (  # noqa: E402
    STANDARD_SESSION_IDS,
    collect_iemocap_records,
)
from src.evaluation.ravdess import collect_ravdess_records  # noqa: E402
from src.explainers.transformer_relevance.score_pipeline import EXPLANATION_MODES  # noqa: E402
from src.speech_xai_project.config import load_config, project_path  # noqa: E402
from src.utils.audio import load_audio_mono_16k  # noqa: E402


REPRODUCIBILITY_PACKAGES = (
    "torch",
    "torchaudio",
    "transformers",
    "whisperx",
    "faster-whisper",
    "ctranslate2",
    "numpy",
    "pandas",
    "scipy",
    "soundfile",
    "pydub",
    "imageio-ffmpeg",
    "pyyaml",
)


def _relative_to_or_string(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _run_text_command(command: list[str], *, cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _git_repository_state(path: Path) -> dict[str, object]:
    path = path.resolve()
    state: dict[str, object] = {
        "path": _relative_to_or_string(path, PROJECT_ROOT),
        "exists": path.exists(),
    }
    if not path.exists():
        return state

    commit = _run_text_command(["git", "rev-parse", "HEAD"], cwd=path)
    remote = _run_text_command(["git", "config", "--get", "remote.origin.url"], cwd=path)
    status = _run_text_command(["git", "status", "--short"], cwd=path)
    state.update(
        {
            "remote": remote,
            "commit": commit,
            "dirty": bool(status),
            "status_short": status.splitlines() if status else [],
        }
    )
    return state


def _installed_package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package_name in REPRODUCIBILITY_PACKAGES:
        try:
            versions[package_name] = importlib_metadata.version(package_name)
        except importlib_metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def _ffmpeg_version() -> str | None:
    version_text = _run_text_command(["ffmpeg", "-version"])
    if not version_text:
        return None
    return version_text.splitlines()[0] if version_text.splitlines() else None


def collect_reproducibility_metadata(*, speechxai_root: str | Path) -> dict[str, object]:
    """Capture enough local state to reproduce or audit an evaluation run."""
    return {
        "command": sys.argv,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "repositories": {
            "project": _git_repository_state(PROJECT_ROOT),
            "speechxai": _git_repository_state(Path(speechxai_root)),
            "legrad": _git_repository_state(PROJECT_ROOT / "third_party" / "LeGrad"),
        },
        "packages": _installed_package_versions(),
        "ffmpeg": _ffmpeg_version(),
    }


def _iemocap_examples(dataset_root: str | Path, session_ids: tuple[int, ...]):
    root = Path(dataset_root)
    examples = []
    for record in collect_iemocap_records(root, session_ids=session_ids):
        examples.append(
            AudioEvaluationExample(
                dataset="IEMOCAP",
                audio_path=str(record.path),
                audio_id=record.utterance_id,
                relative_path=_relative_to_or_string(record.path, root),
                true_class=record.true_class,
                true_label=MODEL_CLASS_NAMES[record.true_class],
                session_id=record.session_id,
                dialogue_id=record.dialogue_id,
                utterance_id=record.utterance_id,
                iemocap_emotion=record.source_emotion,
                annotation_path=str(record.annotation_path),
            )
        )
    return examples


def _ravdess_examples(dataset_root: str | Path):
    root = Path(dataset_root)
    examples = []
    for record in collect_ravdess_records(root):
        examples.append(
            AudioEvaluationExample(
                dataset="RAVDESS",
                audio_path=str(record.path),
                audio_id=record.path.stem,
                relative_path=_relative_to_or_string(record.path, root),
                true_class=record.true_class,
                true_label=MODEL_CLASS_NAMES[record.true_class],
                actor_id=record.actor_id,
                ravdess_emotion=record.emotion_name,
            )
        )
    return examples


def collect_examples(
    *,
    datasets: list[str],
    iemocap_root: str | Path,
    iemocap_sessions: tuple[int, ...],
    ravdess_root: str | Path,
) -> list[AudioEvaluationExample]:
    examples = []
    normalized = {dataset.lower() for dataset in datasets}
    unknown = normalized.difference({"iemocap", "ravdess"})
    if unknown:
        raise ValueError(f"Unsupported datasets: {sorted(unknown)}")
    if "iemocap" in normalized:
        examples.extend(_iemocap_examples(iemocap_root, iemocap_sessions))
    if "ravdess" in normalized:
        examples.extend(_ravdess_examples(ravdess_root))
    return sorted(examples, key=lambda example: (example.dataset, example.audio_id))


def filter_examples(
    examples: list[AudioEvaluationExample],
    *,
    audio_ids: list[str] | None,
    max_audios: int | None,
) -> list[AudioEvaluationExample]:
    selected = examples
    if audio_ids:
        wanted = set(audio_ids)
        selected = [
            example
            for example in selected
            if example.audio_id in wanted or Path(example.audio_path).stem in wanted
        ]
        missing = sorted(wanted.difference({example.audio_id for example in selected}))
        if missing:
            raise RuntimeError(f"No matching audio_id values found: {missing}")
    if max_audios is not None:
        selected = selected[:max_audios]
    if not selected:
        raise RuntimeError("No audio examples selected for evaluation.")
    return selected


def append_frame(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def parse_args():
    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    speechxai_config = config.get("speechxai", {})
    evaluation_config = config.get("evaluation", {})
    paths_config = config.get("paths", {})

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default="iemocap,ravdess",
        help="Comma-separated datasets to evaluate: iemocap,ravdess.",
    )
    parser.add_argument(
        "--iemocap-root",
        default=str(project_path(paths_config.get("iemocap_root", "data"))),
    )
    parser.add_argument(
        "--iemocap-sessions",
        default=",".join(str(value) for value in STANDARD_SESSION_IDS),
    )
    parser.add_argument(
        "--ravdess-root",
        default=str(PROJECT_ROOT / "data" / "ravdess" / "Audio_Speech_Actors_01-24"),
    )
    parser.add_argument(
        "--audio-ids",
        default=None,
        help="Optional comma-separated audio ids for a targeted smoke run.",
    )
    parser.add_argument(
        "--max-audios",
        type=int,
        default=None,
        help="Evaluate only the first N selected audios. Default: all selected audios.",
    )
    parser.add_argument(
        "--correct-only",
        action="store_true",
        help="Skip examples whose original HuBERT prediction is not the dataset label.",
    )
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument(
        "--speechxai-root",
        default=str(project_path(speechxai_config.get("repository_dir", "third_party/SpeechXAI"))),
    )
    parser.add_argument(
        "--speechxai-cache-dir",
        default=str(PROJECT_ROOT / "outputs" / "cache" / "speechxai_words"),
    )
    parser.add_argument(
        "--refresh-speechxai-cache",
        action="store_true",
        help="Ignore existing WhisperX word-alignment cache files.",
    )
    parser.add_argument("--whisper-model", default=speechxai_config.get("whisper_model", "large-v2"))
    parser.add_argument("--speechxai-language", default=speechxai_config.get("language", "en"))
    parser.add_argument(
        "--speechxai-batch-size",
        type=int,
        default=int(speechxai_config.get("batch_size", 2)),
    )
    parser.add_argument(
        "--speechxai-compute-type",
        default=None,
        help="Default: config cpu_compute_type or cuda_compute_type based on --device.",
    )
    parser.add_argument(
        "--ks",
        default=",".join(str(value) for value in DEFAULT_TOP_K_VALUES),
        help="Comma-separated SpeechXAI top-k word counts.",
    )
    parser.add_argument(
        "--random-trials",
        type=int,
        default=int(evaluation_config.get("random_trials", 20)),
    )
    parser.add_argument(
        "--random-bin-seconds",
        type=float,
        default=float(evaluation_config.get("legrad_bin_seconds", 0.05)),
    )
    parser.add_argument("--modes", default=",".join(EXPLANATION_MODES))
    parser.add_argument(
        "--no-bottom",
        action="store_true",
        help="Only evaluate top HuBERT intervals, not bottom intervals.",
    )
    parser.add_argument("--seed", type=int, default=int(evaluation_config.get("random_seed", 13)))
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Inference device. Default: CUDA when available.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the HuBERT checkpoint only from the local Hugging Face cache.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first failed audio instead of writing failures.csv and continuing.",
    )
    args = parser.parse_args()

    args.datasets = parse_csv_values(args.datasets, str, "datasets")
    args.iemocap_sessions = tuple(
        parse_csv_values(args.iemocap_sessions, int, "iemocap-sessions")
    )
    args.audio_ids = (
        parse_csv_values(args.audio_ids, str, "audio-ids") if args.audio_ids else None
    )
    args.ks = parse_csv_values(args.ks, int, "ks")
    args.modes = parse_csv_values(args.modes, str, "modes")
    invalid_modes = sorted(set(args.modes).difference(EXPLANATION_MODES))
    if invalid_modes:
        parser.error(f"Unsupported explanation modes: {invalid_modes}")
    if args.random_trials < 1:
        parser.error("--random-trials must be positive.")
    if args.random_bin_seconds <= 0:
        parser.error("--random-bin-seconds must be positive.")
    if args.max_audios is not None and args.max_audios < 1:
        parser.error("--max-audios must be positive when provided.")
    args.speechxai_config = speechxai_config
    return args


def main():
    args = parse_args()
    disable_huggingface_symlink_cache_on_windows()
    from src.models.load_model import load_hubert_emotion_model

    examples = collect_examples(
        datasets=args.datasets,
        iemocap_root=args.iemocap_root,
        iemocap_sessions=args.iemocap_sessions,
        ravdess_root=args.ravdess_root,
    )
    selected_examples = filter_examples(
        examples,
        audio_ids=args.audio_ids,
        max_audios=args.max_audios,
    )

    dataset_name = "_".join(sorted({example.dataset for example in selected_examples}))
    run_dir = create_unique_run_dir(
        args.output_root,
        f"{dataset_slug(dataset_name)}_duration_matched_speechxai",
    )

    deletion_path = run_dir / "duration_matched_records.csv"
    word_scores_path = run_dir / "speechxai_word_scores.csv"
    selected_intervals_path = run_dir / "selected_intervals.csv"
    original_predictions_path = run_dir / "original_predictions.csv"
    failures_path = run_dir / "failures.csv"
    summary_path = run_dir / "duration_matched_summary.csv"

    manifest = pd.DataFrame([example.to_record() for example in selected_examples])
    manifest.to_csv(run_dir / "audio_manifest.csv", index=False)

    model, processor, device = load_hubert_emotion_model(
        device=args.device,
        local_files_only=args.local_files_only,
    )
    speechxai_compute_type = args.speechxai_compute_type
    if speechxai_compute_type is None:
        if device == "cuda":
            speechxai_compute_type = args.speechxai_config.get("cuda_compute_type", "float16")
        else:
            speechxai_compute_type = args.speechxai_config.get("cpu_compute_type", "int8")

    transcriber = SpeechXAITranscriber(
        speechxai_root=args.speechxai_root,
        device=device,
        batch_size=args.speechxai_batch_size,
        compute_type=speechxai_compute_type,
        language=args.speechxai_language,
        whisper_model=args.whisper_model,
        cache_dir=args.speechxai_cache_dir,
        use_cache=not args.refresh_speechxai_cache,
    )

    failures = []
    completed = 0
    skipped = 0
    for index, example in enumerate(selected_examples, start=1):
        try:
            waveform, sampling_rate = load_audio_mono_16k(example.audio_path)
            original_prediction = classify_waveform(
                model,
                processor,
                waveform,
                sampling_rate,
                device=device,
            )
            append_frame(
                original_predictions_path,
                pd.DataFrame(
                    [
                        {
                            **example.to_record(),
                            "predicted_class": original_prediction.predicted_class,
                            "predicted_label": original_prediction.predicted_label,
                            "target_class": original_prediction.target_class,
                            "target_label": original_prediction.target_label,
                            "target_confidence": original_prediction.target_confidence,
                            "is_correct": (
                                original_prediction.predicted_class == example.true_class
                                if example.true_class is not None
                                else None
                            ),
                        }
                    ]
                ),
            )
            if (
                args.correct_only
                and example.true_class is not None
                and original_prediction.predicted_class != example.true_class
            ):
                skipped += 1
                print(
                    f"[{index}/{len(selected_examples)}] skipped incorrect "
                    f"{example.dataset}/{example.audio_id}"
                )
                continue

            transcript, words = transcriber.transcribe(example.audio_path, example.audio_id)
            result = evaluate_duration_matched_audio(
                model,
                processor,
                waveform,
                sampling_rate,
                example=example,
                words=words,
                transcript=transcript,
                device=device,
                top_k_values=args.ks,
                random_trials=args.random_trials,
                random_bin_seconds=args.random_bin_seconds,
                modes=args.modes,
                include_bottom=not args.no_bottom,
                seed=args.seed,
                padding_before_seconds=float(
                    args.speechxai_config.get("mask_padding_before_seconds", 0.10)
                ),
                padding_after_seconds=float(
                    args.speechxai_config.get("mask_padding_after_seconds", 0.04)
                ),
                original_prediction=original_prediction,
            )
            append_frame(deletion_path, result.deletion_records)
            append_frame(word_scores_path, result.speechxai_word_scores)
            append_frame(selected_intervals_path, result.selected_intervals)
            completed += 1
            print(
                f"[{index}/{len(selected_examples)}] evaluated "
                f"{example.dataset}/{example.audio_id}: "
                f"{len(result.deletion_records)} deletion rows"
            )
        except Exception as error:  # noqa: BLE001
            failure = {
                **example.to_record(),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            append_frame(failures_path, pd.DataFrame([failure]))
            print(
                f"[{index}/{len(selected_examples)}] failed "
                f"{example.dataset}/{example.audio_id}: {type(error).__name__}: {error}"
            )
            if args.fail_fast:
                raise

    if deletion_path.is_file():
        records = pd.read_csv(deletion_path)
        summary = summarize_duration_matched_records(records)
        summary.to_csv(summary_path, index=False)
    else:
        summary = pd.DataFrame()

    pd.DataFrame(
        [
            {
                "method": "speechxai_pastor_loo_words",
                "method_family": "speechxai_pastor",
                "selection": "top_words",
                "description": "WhisperX word intervals scored by Pastor/SpeechXAI leave-one-out target-confidence drop.",
            },
            *[
                {
                    "method": mode,
                    "method_family": "hubert_temporal_relevance",
                    "selection": "top/bottom" if not args.no_bottom else "top",
                    "description": "HuBERT temporal relevance bins selected to match SpeechXAI duration.",
                }
                for mode in args.modes
            ],
            {
                "method": "random_duration_matched_bins",
                "method_family": "random_baseline",
                "selection": "random",
                "description": "Random deletion by silence masking: random fixed-size audio bins selected to match SpeechXAI duration and zeroed without changing waveform length.",
            },
        ]
    ).to_csv(run_dir / "method_catalog.csv", index=False)

    write_json(
        run_dir / "config.json",
        {
            "datasets": args.datasets,
            "iemocap_root": str(Path(args.iemocap_root).resolve()),
            "iemocap_sessions": args.iemocap_sessions,
            "ravdess_root": str(Path(args.ravdess_root).resolve()),
            "selected_audio_count": len(selected_examples),
            "completed_audio_count": completed,
            "skipped_audio_count": skipped,
            "failed_audio_count": len(failures),
            "correct_only": args.correct_only,
            "ks": args.ks,
            "random_trials": args.random_trials,
            "random_bin_seconds": args.random_bin_seconds,
            "modes": args.modes,
            "include_bottom": not args.no_bottom,
            "seed": args.seed,
            "device": device,
            "local_files_only": args.local_files_only,
            "speechxai_root": str(Path(args.speechxai_root).resolve()),
            "speechxai_cache_dir": str(Path(args.speechxai_cache_dir).resolve()),
            "speechxai_use_cache": not args.refresh_speechxai_cache,
            "whisper_model": args.whisper_model,
            "speechxai_language": args.speechxai_language,
            "speechxai_batch_size": args.speechxai_batch_size,
            "speechxai_compute_type": speechxai_compute_type,
            "outputs": {
                "audio_manifest": "audio_manifest.csv",
                "original_predictions": "original_predictions.csv",
                "duration_matched_records": "duration_matched_records.csv",
                "duration_matched_summary": "duration_matched_summary.csv",
                "speechxai_word_scores": "speechxai_word_scores.csv",
                "selected_intervals": "selected_intervals.csv",
                "failures": "failures.csv",
                "method_catalog": "method_catalog.csv",
            },
            "reproducibility": collect_reproducibility_metadata(
                speechxai_root=args.speechxai_root,
            ),
        },
    )

    print("\nDuration-matched SpeechXAI comparison complete")
    print("Run directory:", run_dir)
    print("Selected audios:", len(selected_examples))
    print("Completed audios:", completed)
    print("Skipped audios:", skipped)
    print("Failed audios:", len(failures))
    if not summary.empty:
        print("Deletion records:", len(pd.read_csv(deletion_path)))
        print("Summary rows:", len(summary))


if __name__ == "__main__":
    main()
