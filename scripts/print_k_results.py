#!/usr/bin/env python3
"""Print K=3, 5, and 10 metrics for the selected EdgeCode configuration.

Run this from the extracted EdgeCode project folder:
    python scripts/print_k_results.py

For a quick run's output folder:
    python scripts/print_k_results.py --output-dir outputs/development20_quick
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show K=3, 5, and 10 metrics for the selected configuration."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/development20"),
        help="Tuning output folder (default: outputs/development20)",
    )
    args = parser.parse_args()

    config_path = args.output_dir / "selected_config.json"
    metrics_path = args.output_dir / "development_tuning.csv"

    if not config_path.is_file() or not metrics_path.is_file():
        parser.error(
            f"Could not find the tuning outputs in {args.output_dir}. "
            "Run this script from the extracted project folder, or pass the "
            "correct folder with --output-dir."
        )

    with config_path.open(encoding="utf-8") as file:
        selected = json.load(file)

    with metrics_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    ast_size = int(selected["ast_max_lines"])
    bm25_weight = float(selected["bm25_weight"])
    matching_rows = [
        row
        for row in rows
        if int(row["ast_max_lines"]) == ast_size
        and abs(float(row["bm25_weight"]) - bm25_weight) < 1e-9
    ]

    if len(matching_rows) != 1:
        parser.error(
            f"Expected one metrics row for AST max lines={ast_size} and "
            f"BM25 weight={bm25_weight}; found {len(matching_rows)}."
        )

    row = matching_rows[0]
    print(
        f"Selected configuration: AST max lines={ast_size}, "
        f"BM25 weight={bm25_weight:g}, "
        f"dense weight={float(selected['dense_weight']):g}"
    )
    print("Metrics are averaged over the development questions.")

    for k in (3, 5, 10):
        print(
            f"K={k}: "
            f"MRR={float(row[f'mrr_at_{k}']):.4f}, "
            f"Recall={float(row[f'recall_at_{k}']):.4f}, "
            f"Precision={float(row[f'precision_at_{k}']):.4f}, "
            f"Hit rate={float(row[f'hit_rate_at_{k}']):.4f}"
        )


if __name__ == "__main__":
    main()
