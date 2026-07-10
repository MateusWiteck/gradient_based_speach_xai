"""Create compact summary tables from existing explanation metric CSV files.

This is intentionally post-processing only: it reads existing result files and
writes a new timestamped folder, so old research outputs are not modified.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.common import create_unique_run_dir, write_json
from src.evaluation.explanation_metrics import (
    summarize_class_specificity_by_method,
    summarize_sparsity_by_method,
)


def resolve_optional_file(explicit_file: str | None, run_dir: str | None, filename: str) -> Path | None:
    """Resolve either a direct CSV path or a run directory plus default filename."""
    if explicit_file:
        return Path(explicit_file)
    if run_dir:
        return Path(run_dir) / filename
    return None


def write_markdown_table(frame: pd.DataFrame, path: Path) -> None:
    """Write a simple Markdown table without requiring optional tabulate."""
    with path.open("w", encoding="utf-8") as output_file:
        output_file.write("| " + " | ".join(frame.columns) + " |\n")
        output_file.write("| " + " | ".join(["---"] * len(frame.columns)) + " |\n")
        for _, row in frame.iterrows():
            values = []
            for value in row:
                if isinstance(value, float):
                    values.append(f"{value:.6g}")
                else:
                    values.append(str(value))
            output_file.write("| " + " | ".join(values) + " |\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--faithfulness-run",
        default=None,
        help="Directory containing sparsity_records.csv.",
    )
    parser.add_argument(
        "--sparsity-records",
        default=None,
        help="Direct path to sparsity_records.csv.",
    )
    parser.add_argument(
        "--class-specificity-run",
        default=None,
        help="Directory containing class_specificity_records.csv.",
    )
    parser.add_argument(
        "--class-specificity-records",
        default=None,
        help="Direct path to class_specificity_records.csv.",
    )
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()

    sparsity_path = resolve_optional_file(
        args.sparsity_records,
        args.faithfulness_run,
        "sparsity_records.csv",
    )
    specificity_path = resolve_optional_file(
        args.class_specificity_records,
        args.class_specificity_run,
        "class_specificity_records.csv",
    )
    if sparsity_path is None and specificity_path is None:
        parser.error(
            "Provide --faithfulness-run/--sparsity-records and/or "
            "--class-specificity-run/--class-specificity-records."
        )

    run_dir = create_unique_run_dir(args.output_root, "explanation_metric_tables")
    config = {
        "sparsity_records": str(sparsity_path.resolve()) if sparsity_path else None,
        "class_specificity_records": (
            str(specificity_path.resolve()) if specificity_path else None
        ),
        "generated_files": [],
    }

    if sparsity_path is not None:
        if not sparsity_path.exists():
            raise FileNotFoundError(f"Sparsity records not found: {sparsity_path}")
        sparsity_summary = summarize_sparsity_by_method(pd.read_csv(sparsity_path))
        sparsity_summary.to_csv(run_dir / "sparsity_summary_by_method.csv", index=False)
        write_markdown_table(
            sparsity_summary,
            run_dir / "sparsity_summary_by_method.md",
        )
        config["generated_files"].extend(
            ["sparsity_summary_by_method.csv", "sparsity_summary_by_method.md"]
        )

    if specificity_path is not None:
        if not specificity_path.exists():
            raise FileNotFoundError(
                f"Class-specificity records not found: {specificity_path}"
            )
        specificity_summary = summarize_class_specificity_by_method(
            pd.read_csv(specificity_path)
        )
        specificity_summary.to_csv(
            run_dir / "class_specificity_summary_by_method.csv",
            index=False,
        )
        write_markdown_table(
            specificity_summary,
            run_dir / "class_specificity_summary_by_method.md",
        )
        config["generated_files"].extend(
            [
                "class_specificity_summary_by_method.csv",
                "class_specificity_summary_by_method.md",
            ]
        )

    write_json(run_dir / "config.json", config)
    print("Saved summary tables to:", run_dir)


if __name__ == "__main__":
    main()
