"""Prepare Colab and run the full duration-matched evaluation on GPU.

This script keeps the Colab notebook cell small. It handles Google Drive,
Kaggle authentication, third-party repositories, dataset downloads, validation,
dependency installation, GPU checks, and finally launches
``evaluate_duration_matched_speechxai.py`` with unbuffered output.
"""

from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DRIVE_OUTPUT_BASE = Path("/content/drive/MyDrive/gradient_based_speech_xai_outputs")
DEFAULT_IEMOCAP_DOWNLOAD_DIR = Path("/content/data/iemocap")
DEFAULT_RAVDESS_DOWNLOAD_DIR = Path("/content/data/ravdess")
DEFAULT_RAVDESS_URL = (
    "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
)
DEFAULT_SPEECHXAI_COMMIT = "7c43d0ce90c82ca3d2f860534136f06d3640e8d0"


def run(command: list[object], *, cwd: Path | None = None) -> None:
    command = [str(part) for part in command]
    print("+", " ".join(command), flush=True)
    try:
        subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        print("\nCommand failed:", " ".join(command), flush=True)
        print("Exit code:", error.returncode, flush=True)
        raise


def safe_extract_zip(zip_path: Path, destination: Path) -> None:
    """Validate ZIP paths before extraction."""
    if not zipfile.is_zipfile(zip_path):
        raise RuntimeError(f"Invalid ZIP file: {zip_path}")

    destination.mkdir(parents=True, exist_ok=True)
    resolved_destination = destination.resolve()

    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            member_destination = (resolved_destination / member.filename).resolve()
            if (
                member_destination != resolved_destination
                and resolved_destination not in member_destination.parents
            ):
                raise RuntimeError(f"Unsafe path found inside ZIP: {member.filename}")
        archive.extractall(resolved_destination)


def find_valid_zip(directory: Path) -> Path | None:
    if not directory.exists():
        return None
    candidates = sorted(
        directory.glob("*.zip"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if zipfile.is_zipfile(candidate):
            return candidate
    return None


def remove_invalid_zip_files(directory: Path) -> None:
    if not directory.exists():
        return
    for candidate in directory.glob("*.zip"):
        if not zipfile.is_zipfile(candidate):
            print("Removing invalid ZIP:", candidate, flush=True)
            candidate.unlink()


def contains_iemocap_sessions(path: Path) -> bool:
    return all((path / f"Session{session_id}").is_dir() for session_id in range(1, 6))


def find_iemocap_root(search_root: Path) -> Path | None:
    if not search_root.exists():
        return None
    if contains_iemocap_sessions(search_root):
        return search_root
    for session1_path in search_root.rglob("Session1"):
        if not session1_path.is_dir():
            continue
        candidate = session1_path.parent
        if contains_iemocap_sessions(candidate):
            return candidate
    return None


def is_ravdess_root(path: Path) -> bool:
    return (path / "Actor_01").is_dir() and (path / "Actor_24").is_dir()


def find_ravdess_root(search_root: Path) -> Path | None:
    if not search_root.exists():
        return None
    if is_ravdess_root(search_root):
        return search_root
    for actor1_path in search_root.rglob("Actor_01"):
        if not actor1_path.is_dir():
            continue
        candidate = actor1_path.parent
        if is_ravdess_root(candidate):
            return candidate
    return None


def mount_drive_and_read_kaggle_secret(secret_name: str) -> None:
    try:
        from google.colab import drive, userdata
    except ImportError as error:
        raise RuntimeError("This script is intended to run inside Google Colab.") from error

    drive.mount("/content/drive")

    try:
        kaggle_api_token = userdata.get(secret_name)
    except Exception as error:
        raise RuntimeError(
            f"Could not read the Colab Secret {secret_name!r}. "
            "Open the Secrets panel in Colab, create it, and enable Notebook access."
        ) from error

    if not kaggle_api_token or not kaggle_api_token.strip():
        raise RuntimeError(f"The Colab Secret {secret_name!r} does not exist or is empty.")

    os.environ["KAGGLE_API_TOKEN"] = kaggle_api_token.strip()
    print(f"Colab Secret {secret_name!r} loaded.", flush=True)


def require_gpu() -> None:
    run(["nvidia-smi"])
    print("NVIDIA GPU is visible. PyTorch CUDA will be checked after install.", flush=True)


def require_torch_cuda_after_install() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available after dependency installation.")
    print("CUDA device:", torch.cuda.get_device_name(0), flush=True)
    print("torch:", torch.__version__, flush=True)
    print("CUDA runtime:", torch.version.cuda, flush=True)


def ensure_legrad_submodule() -> None:
    legrad_dir = PROJECT_ROOT / "third_party" / "LeGrad"
    gitmodules_path = PROJECT_ROOT / ".gitmodules"
    if not gitmodules_path.is_file():
        raise FileNotFoundError(f"{gitmodules_path} was not found.")

    if legrad_dir.exists():
        git_marker = legrad_dir / ".git"
        if not git_marker.exists() or git_marker.is_dir():
            print("Removing incompatible LeGrad directory:", legrad_dir, flush=True)
            shutil.rmtree(legrad_dir)

    run(["git", "submodule", "sync", "--recursive"], cwd=PROJECT_ROOT)
    run(
        ["git", "submodule", "update", "--init", "--recursive", "third_party/LeGrad"],
        cwd=PROJECT_ROOT,
    )
    run(["git", "submodule", "status", "--recursive"], cwd=PROJECT_ROOT)
    run(["git", "log", "-1", "--oneline"], cwd=legrad_dir)


def ensure_speechxai_repository(speechxai_url: str, speechxai_commit: str) -> Path:
    speechxai_dir = PROJECT_ROOT / "third_party" / "SpeechXAI"

    if speechxai_dir.exists() and not (speechxai_dir / ".git").exists():
        print("Removing invalid SpeechXAI directory:", speechxai_dir, flush=True)
        shutil.rmtree(speechxai_dir)

    speechxai_dir.parent.mkdir(parents=True, exist_ok=True)
    if not speechxai_dir.exists():
        run(["git", "clone", speechxai_url, speechxai_dir])

    run(["git", "remote", "set-url", "origin", speechxai_url], cwd=speechxai_dir)
    run(["git", "fetch", "--all", "--tags", "--prune"], cwd=speechxai_dir)
    run(["git", "checkout", "--detach", speechxai_commit], cwd=speechxai_dir)
    run(["git", "reset", "--hard", speechxai_commit], cwd=speechxai_dir)
    run(["git", "log", "-1", "--oneline"], cwd=speechxai_dir)
    return speechxai_dir


def verify_project_structure(speechxai_dir: Path) -> None:
    required_paths = [
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "configs" / "default.yaml",
        PROJECT_ROOT / "scripts" / "evaluate_duration_matched_speechxai.py",
        PROJECT_ROOT / "src" / "evaluation" / "iemocap.py",
        PROJECT_ROOT / "src" / "evaluation" / "ravdess.py",
        PROJECT_ROOT / "src" / "evaluation" / "duration_matched_speechxai.py",
        PROJECT_ROOT / "src" / "explainers" / "transformer_relevance" / "score_pipeline.py",
        PROJECT_ROOT / "third_party" / "LeGrad",
        speechxai_dir,
    ]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required project files are missing:\n"
            + "\n".join(f"- {path}" for path in missing)
        )


def install_dependencies() -> None:
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "wheel", "kaggle"])
    run([sys.executable, "-m", "pip", "install", "-r", PROJECT_ROOT / "requirements.txt"])


def verify_kaggle_auth(dataset: str) -> None:
    run(
        [
            sys.executable,
            "-m",
            "kaggle",
            "datasets",
            "files",
            dataset,
            "--page-size",
            "1",
        ]
    )


def prepare_iemocap(args: argparse.Namespace) -> Path:
    download_dir = args.iemocap_download_dir
    extract_dir = download_dir / "extracted"
    download_dir.mkdir(parents=True, exist_ok=True)

    iemocap_root = find_iemocap_root(extract_dir)
    iemocap_zip = find_valid_zip(download_dir)

    if iemocap_root is not None:
        print("Using existing complete IEMOCAP extraction:", iemocap_root, flush=True)
        return iemocap_root.resolve()

    if iemocap_zip is None:
        remove_invalid_zip_files(download_dir)
        print("Downloading compressed IEMOCAP dataset:", args.kaggle_iemocap_dataset, flush=True)
        run(
            [
                sys.executable,
                "-m",
                "kaggle",
                "datasets",
                "download",
                "--dataset",
                args.kaggle_iemocap_dataset,
                "--path",
                download_dir,
            ]
        )
        iemocap_zip = find_valid_zip(download_dir)

    if iemocap_zip is None:
        files_found = sorted(download_dir.iterdir())
        listing = "\n".join(f"- {path}" for path in files_found) if files_found else "- none"
        raise FileNotFoundError(
            "The Kaggle command completed, but no valid ZIP was found.\n" + listing
        )

    print("IEMOCAP ZIP:", iemocap_zip, flush=True)
    print("IEMOCAP ZIP size:", f"{iemocap_zip.stat().st_size / (1024 ** 3):.2f} GB", flush=True)

    if extract_dir.exists():
        print("Removing incomplete previous IEMOCAP extraction:", extract_dir, flush=True)
        shutil.rmtree(extract_dir)
    safe_extract_zip(iemocap_zip, extract_dir)

    iemocap_root = find_iemocap_root(extract_dir)
    if iemocap_root is None:
        directories = sorted(path for path in extract_dir.rglob("*") if path.is_dir())
        preview = "\n".join(f"- {path}" for path in directories[:100])
        raise FileNotFoundError(
            "IEMOCAP was extracted, but Session1 through Session5 were not found.\n"
            f"{preview or '- none'}"
        )

    if args.delete_archives and iemocap_zip.exists():
        iemocap_zip.unlink()
    print("IEMOCAP root:", iemocap_root.resolve(), flush=True)
    return iemocap_root.resolve()


def prepare_ravdess(args: argparse.Namespace) -> Path:
    download_dir = args.ravdess_download_dir
    ravdess_zip = download_dir / "Audio_Speech_Actors_01-24.zip"
    download_dir.mkdir(parents=True, exist_ok=True)

    ravdess_root = find_ravdess_root(download_dir)
    if ravdess_root is not None:
        print("Using existing complete RAVDESS extraction:", ravdess_root, flush=True)
        return ravdess_root.resolve()

    if ravdess_zip.exists() and not zipfile.is_zipfile(ravdess_zip):
        print("Removing invalid RAVDESS ZIP:", ravdess_zip, flush=True)
        ravdess_zip.unlink()

    if not ravdess_zip.exists():
        temporary_zip = download_dir / "Audio_Speech_Actors_01-24.zip.part"
        if temporary_zip.exists():
            temporary_zip.unlink()
        print("Downloading compressed RAVDESS dataset:", args.ravdess_url, flush=True)
        urllib.request.urlretrieve(args.ravdess_url, temporary_zip)
        temporary_zip.replace(ravdess_zip)

    print("RAVDESS ZIP:", ravdess_zip, flush=True)
    print("RAVDESS ZIP size:", f"{ravdess_zip.stat().st_size / (1024 ** 3):.2f} GB", flush=True)
    safe_extract_zip(ravdess_zip, download_dir)

    ravdess_root = find_ravdess_root(download_dir)
    if ravdess_root is None:
        raise FileNotFoundError("RAVDESS was extracted, but Actor_01 through Actor_24 were not found.")

    if args.delete_archives and ravdess_zip.exists():
        ravdess_zip.unlink()
    print("RAVDESS root:", ravdess_root.resolve(), flush=True)
    return ravdess_root.resolve()


def verify_dataset_counts(
    *,
    iemocap_root: Path,
    ravdess_root: Path,
    expected_iemocap_count: int,
    expected_ravdess_count: int,
) -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.evaluate_duration_matched_speechxai import collect_examples
    from src.evaluation.iemocap import STANDARD_SESSION_IDS

    examples = collect_examples(
        datasets=["iemocap", "ravdess"],
        iemocap_root=iemocap_root,
        iemocap_sessions=STANDARD_SESSION_IDS,
        ravdess_root=ravdess_root,
    )
    counts = Counter(example.dataset for example in examples)
    print("Selected records:", dict(sorted(counts.items())), "total=", len(examples), flush=True)

    expected = Counter({"IEMOCAP": expected_iemocap_count, "RAVDESS": expected_ravdess_count})
    if counts != expected:
        raise RuntimeError(f"Unexpected dataset counts. Expected {dict(expected)}, got {dict(counts)}.")


def run_full_evaluation(
    *,
    args: argparse.Namespace,
    iemocap_root: Path,
    ravdess_root: Path,
    speechxai_dir: Path,
) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.speechxai_cache_dir.mkdir(parents=True, exist_ok=True)
    print("Output root:", args.output_root, flush=True)
    print("SpeechXAI cache:", args.speechxai_cache_dir, flush=True)

    command = [
        sys.executable,
        "-u",
        "scripts/evaluate_duration_matched_speechxai.py",
        "--datasets",
        "iemocap,ravdess",
        "--iemocap-root",
        iemocap_root,
        "--ravdess-root",
        ravdess_root,
        "--speechxai-root",
        speechxai_dir,
        "--speechxai-cache-dir",
        args.speechxai_cache_dir,
        "--output-root",
        args.output_root,
        "--ks",
        args.ks,
        "--random-trials",
        str(args.random_trials),
        "--device",
        "cuda" if args.require_gpu else args.device,
        "--speechxai-compute-type",
        args.speechxai_compute_type,
        "--speechxai-batch-size",
        str(args.speechxai_batch_size),
    ]
    if args.whisper_model:
        command.extend(["--whisper-model", args.whisper_model])
    run(command, cwd=PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kaggle-secret-name", default="KAGGLE_API_TOKEN")
    parser.add_argument("--kaggle-iemocap-dataset", default="jamaliasultanajisha/iemocap-full")
    parser.add_argument("--speechxai-url", default="https://github.com/elianap/SpeechXAI.git")
    parser.add_argument("--speechxai-commit", default=DEFAULT_SPEECHXAI_COMMIT)
    parser.add_argument("--iemocap-download-dir", type=Path, default=DEFAULT_IEMOCAP_DOWNLOAD_DIR)
    parser.add_argument("--ravdess-download-dir", type=Path, default=DEFAULT_RAVDESS_DOWNLOAD_DIR)
    parser.add_argument("--ravdess-url", default=DEFAULT_RAVDESS_URL)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_DRIVE_OUTPUT_BASE / "test_06_duration_matched_full_both_datasets",
    )
    parser.add_argument(
        "--speechxai-cache-dir",
        type=Path,
        default=DEFAULT_DRIVE_OUTPUT_BASE / "cache" / "speechxai_words",
    )
    parser.add_argument("--ks", default="1,2,3,5")
    parser.add_argument("--random-trials", type=int, default=20)
    parser.add_argument("--speechxai-batch-size", type=int, default=1)
    parser.add_argument("--speechxai-compute-type", default="float16")
    parser.add_argument("--whisper-model", default=None)
    parser.add_argument("--expected-iemocap-count", type=int, default=5531)
    parser.add_argument("--expected-ravdess-count", type=int, default=672)
    parser.add_argument("--delete-archives", action="store_true")
    parser.add_argument(
        "--require-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail when CUDA is not visible. Enabled by default.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Fallback device only used with --no-require-gpu.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mount_drive_and_read_kaggle_secret(args.kaggle_secret_name)
    if args.require_gpu:
        require_gpu()

    ensure_legrad_submodule()
    speechxai_dir = ensure_speechxai_repository(args.speechxai_url, args.speechxai_commit)
    verify_project_structure(speechxai_dir)
    install_dependencies()
    verify_kaggle_auth(args.kaggle_iemocap_dataset)
    if args.require_gpu:
        require_torch_cuda_after_install()

    iemocap_root = prepare_iemocap(args)
    ravdess_root = prepare_ravdess(args)
    verify_dataset_counts(
        iemocap_root=iemocap_root,
        ravdess_root=ravdess_root,
        expected_iemocap_count=args.expected_iemocap_count,
        expected_ravdess_count=args.expected_ravdess_count,
    )
    run_full_evaluation(
        args=args,
        iemocap_root=iemocap_root,
        ravdess_root=ravdess_root,
        speechxai_dir=speechxai_dir,
    )


if __name__ == "__main__":
    main()
