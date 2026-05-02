from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml_svg_project.config import CONFIGS_DIR
from ml_svg_project.dataset_specs import (
    dataset_identities_from_specs,
    dataset_names_from_specs,
    parse_dataset_specs,
)
from ml_svg_project.io_utils import load_yaml
from ml_svg_project.paths import ensure_project_dirs
from ml_svg_project.preprocessing import (
    ProcessedSvg,
    clean_svg,
    combined_dataset_name,
    estimate_token_count,
    infer_id_column,
    infer_svg_column,
    persist_dataset_snapshot,
    split_records,
    summarize_records,
    validate_svg,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and preprocess SVG datasets.")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIGS_DIR / "preprocessing" / "default.yaml",
        help="Path to preprocessing config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val/test splitting.",
    )
    return parser.parse_args()


def load_hf_dataset(name: str, streaming: bool = False) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required. Install dependencies before running preprocessing."
        ) from exc
    return load_dataset(name, streaming=streaming)


def flatten_dataset_rows(dataset: Any, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    remaining = max_rows
    if hasattr(dataset, "items"):
        for split_name, split_dataset in dataset.items():
            for index, row in enumerate(split_dataset):
                enriched = dict(row)
                enriched["_source_split"] = str(split_name)
                enriched["_source_index"] = index
                rows.append(enriched)
                if remaining is not None:
                    remaining -= 1
                    if remaining <= 0:
                        return rows
        return rows
    for index, row in enumerate(dataset):
        enriched = dict(row)
        enriched["_source_split"] = "train"
        enriched["_source_index"] = index
        rows.append(enriched)
        if remaining is not None:
            remaining -= 1
            if remaining <= 0:
                return rows
    return rows


def process_dataset(config: dict[str, Any], seed: int) -> dict[str, Path]:
    dataset_cfg = config["dataset"]
    dataset_specs = parse_dataset_specs(dataset_cfg)
    dataset_names = dataset_names_from_specs(dataset_specs)
    cleaning_cfg = config["cleaning"]
    filtering_cfg = config["filtering"]
    validation_cfg = config["validation"]

    total_seen = 0
    rejection_reasons: Counter[str] = Counter()
    processed_records: list[ProcessedSvg] = []
    all_raw_rows: list[dict[str, Any]] = []

    for dataset_spec in dataset_specs:
        raw_dataset = load_hf_dataset(dataset_spec.name, streaming=dataset_spec.streaming)
        raw_rows = flatten_dataset_rows(raw_dataset, max_rows=dataset_spec.max_rows)
        if not raw_rows:
            raise RuntimeError(f"No rows found in dataset: {dataset_spec.name}")

        svg_column = infer_svg_column(raw_rows[0])
        id_column = infer_id_column(raw_rows[0], svg_column)

        for row in raw_rows:
            row["_dataset_name"] = dataset_spec.name
            all_raw_rows.append(row)
            total_seen += 1
            raw_svg = row.get(svg_column)
            if not isinstance(raw_svg, str):
                rejection_reasons["missing_svg_text"] += 1
                continue

            cleaned_svg = clean_svg(raw_svg, cleaning_cfg)
            if len(cleaned_svg) < int(filtering_cfg["min_chars"]):
                rejection_reasons["too_short"] += 1
                continue

            estimated_tokens = estimate_token_count(cleaned_svg)
            if estimated_tokens > int(filtering_cfg["max_tokens_estimate"]):
                rejection_reasons["too_long_estimated_tokens"] += 1
                continue

            is_valid, reason = validate_svg(cleaned_svg, validation_cfg)
            if not is_valid:
                rejection_reasons[reason or "invalid_svg"] += 1
                continue

            source_value = row.get(id_column) if id_column else None
            source_id = (
                str(source_value) if source_value is not None else str(row["_source_index"])
            )
            processed_records.append(
                ProcessedSvg(
                    dataset_name=dataset_spec.name,
                    source_id=source_id,
                    source_split=str(row["_source_split"]),
                    svg=cleaned_svg,
                    char_length=len(cleaned_svg),
                    estimated_tokens=estimated_tokens,
                )
            )

    split_records_map = split_records(processed_records, config["splits"], seed=seed)
    summary = summarize_records(
        dataset_names=dataset_names,
        total_seen=total_seen,
        rejection_reasons=dict(rejection_reasons),
        records=processed_records,
        split_records_map=split_records_map,
    )
    output_name = combined_dataset_name(dataset_identities_from_specs(dataset_specs))
    paths = persist_dataset_snapshot(
        dataset_name=output_name,
        raw_rows=all_raw_rows,
        processed_records=processed_records,
        split_records_map=split_records_map,
        summary=summary,
    )
    print(f"Processed datasets: {', '.join(dataset_names)}")
    print(f"Rows seen: {summary['total_seen']}")
    print(f"Rows kept: {summary['total_kept']}")
    print(f"Processed output: {paths['processed_dir']}")
    print(f"Summary: {paths['summary_json']}")
    return paths


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    config = load_yaml(args.config)
    process_dataset(config, seed=args.seed)


if __name__ == "__main__":
    main()
