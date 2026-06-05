from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speech_xai_project import config


EVALUATION_LINE_PATTERN = re.compile(
    r"^\[(?P<start>[\d.]+)\s+-\s+(?P<end>[\d.]+)\]\s+"
    r"(?P<audio_id>\S+)\s+(?P<label>\S+)\s+"
    r"\[(?P<valence>[\d.]+),\s*(?P<activation>[\d.]+),\s*(?P<dominance>[\d.]+)\]"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build metadata.csv from an IEMOCAP Session* layout.")
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--keep-labels",
        nargs="*",
        default=["ang", "hap", "exc", "sad", "neu"],
        help="Emotion labels to keep. Use an empty value list only by editing the script.",
    )
    return parser.parse_args()


def build_audio_index(iemocap_root: Path) -> dict[str, Path]:
    audio_paths = sorted(iemocap_root.glob("Session*/sentences/wav/**/*.wav"))
    return {path.stem: path for path in audio_paths}


def parse_evaluation_files(iemocap_root: Path) -> pd.DataFrame:
    rows = []
    for evaluation_path in sorted(iemocap_root.glob("Session*/dialog/EmoEvaluation/*.txt")):
        session = evaluation_path.parts[-4]
        for line in evaluation_path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = EVALUATION_LINE_PATTERN.match(line)
            if not match:
                continue
            row = match.groupdict()
            row["session"] = session
            row["dialog_id"] = evaluation_path.stem
            rows.append(row)

    metadata_table = pd.DataFrame(rows)
    if metadata_table.empty:
        return metadata_table

    numeric_columns = ["start", "end", "valence", "activation", "dominance"]
    metadata_table[numeric_columns] = metadata_table[numeric_columns].astype(float)
    metadata_table = metadata_table.rename(columns={"label": "true_label"})
    return metadata_table


def main() -> int:
    args = parse_args()
    project_config = config.load_config(args.config)
    iemocap_root = config.project_path(project_config["paths"]["iemocap_root"])
    output_path = Path(args.output) if args.output else config.project_path(project_config["paths"]["metadata_csv"])

    audio_index = build_audio_index(iemocap_root)
    metadata_table = parse_evaluation_files(iemocap_root)
    if metadata_table.empty:
        print(f"No IEMOCAP evaluation labels found under {iemocap_root}")
        return 1

    metadata_table["audio_path"] = metadata_table["audio_id"].map(
        lambda audio_id: str(audio_index.get(audio_id, ""))
    )
    metadata_table = metadata_table[metadata_table["audio_path"] != ""].copy()

    if args.keep_labels:
        metadata_table = metadata_table[metadata_table["true_label"].isin(args.keep_labels)].copy()

    metadata_table = metadata_table[
        [
            "audio_id",
            "audio_path",
            "true_label",
            "session",
            "dialog_id",
            "start",
            "end",
            "valence",
            "activation",
            "dominance",
        ]
    ].sort_values(["session", "dialog_id", "audio_id"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_table.to_csv(output_path, index=False)
    print(f"Saved {len(metadata_table)} rows to {output_path}")
    print(metadata_table["true_label"].value_counts().sort_index())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

