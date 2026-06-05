from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speech_xai_project import audio, config, model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step 1 SpeechBrain IEMOCAP inference.")
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument("--limit", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_config = config.load_config(args.config)

    iemocap_root = config.project_path(project_config["paths"]["iemocap_root"])
    metadata_csv = config.project_path(project_config["paths"]["metadata_csv"])
    predictions_csv = config.project_path(project_config["paths"]["predictions_csv"])
    model_source = project_config["model"]["speechbrain_source"]
    model_savedir = config.project_path(project_config["model"]["savedir"])
    target_sample_rate = project_config["model"]["sample_rate"]

    metadata_table = audio.load_metadata_or_discover(metadata_csv, iemocap_root, limit=args.limit)
    if metadata_table.empty:
        print(
            "No audio files found. Add IEMOCAP audio under "
            f"{iemocap_root} or create {metadata_csv} with audio_id,audio_path,true_label."
        )
        return 1

    predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    classifier = model.load_speechbrain_classifier(model_source, model_savedir)
    prediction_rows = []

    for row in metadata_table.itertuples(index=False):
        waveform, sample_rate = audio.load_waveform(row.audio_path, target_sample_rate)
        prediction = model.classify_waveform(classifier, audio.waveform_to_batch(waveform))
        prediction_rows.append(
            {
                "audio_id": row.audio_id,
                "audio_path": row.audio_path,
                "true_label": getattr(row, "true_label", None),
                "duration_seconds": audio.duration_seconds(waveform, sample_rate),
                "predicted_label": prediction.predicted_label,
                "predicted_index": prediction.predicted_index,
                "predicted_confidence": prediction.predicted_confidence,
            }
        )

    predictions_table = pd.DataFrame(prediction_rows)
    predictions_table.to_csv(predictions_csv, index=False)
    print(f"Saved {len(predictions_table)} predictions to {predictions_csv}")
    print(predictions_table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
