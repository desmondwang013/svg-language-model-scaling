from __future__ import annotations

import math
import random
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lxml import etree

from .config import DATA_DIR
from .io_utils import write_json, write_jsonl


SVG_TOKEN_PATTERN = re.compile(
    r"[A-Za-z_:/-]+|[-+]?(?:\d+\.\d+|\d+)|#[0-9A-Fa-f]+|[^\s]"
)
COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)
METADATA_PATTERN = re.compile(r"<metadata\b.*?</metadata>", re.DOTALL | re.IGNORECASE)
DECIMAL_PATTERN = re.compile(r"(?<![\w-])[-+]?\d*\.\d+(?![\w-])")


@dataclass
class ProcessedSvg:
    dataset_name: str
    source_id: str
    source_split: str
    svg: str
    char_length: int
    estimated_tokens: int


def slugify_dataset_name(name: str) -> str:
    return name.replace("/", "__").replace("-", "_")


def combined_dataset_name(dataset_names: list[str]) -> str:
    if not dataset_names:
        raise ValueError("At least one dataset name is required.")
    if len(dataset_names) == 1:
        return dataset_names[0]
    return "__plus__".join(slugify_dataset_name(name) for name in dataset_names)


def dataset_paths(dataset_name: str) -> dict[str, Path]:
    slug = slugify_dataset_name(dataset_name)
    base = DATA_DIR
    return {
        "raw_dir": base / "raw" / slug,
        "processed_dir": base / "processed" / slug,
    }


def infer_svg_column(row: dict[str, Any]) -> str:
    for key, value in row.items():
        if isinstance(value, str) and "<svg" in value.lower():
            return key
    raise ValueError("Could not infer SVG column from dataset row.")


def infer_id_column(row: dict[str, Any], svg_column: str) -> str | None:
    preferred = ("id", "name", "key", "filename", "file_name", "slug")
    for key in preferred:
        value = row.get(key)
        if isinstance(value, (str, int)):
            return key
    for key, value in row.items():
        if key != svg_column and isinstance(value, (str, int)):
            return key
    return None


def estimate_token_count(svg: str) -> int:
    return len(SVG_TOKEN_PATTERN.findall(svg))


def round_decimal_match(match: re.Match[str], decimals: int) -> str:
    number = float(match.group(0))
    rounded = round(number, decimals)
    if rounded.is_integer():
        return str(int(rounded))
    return f"{rounded:.{decimals}f}".rstrip("0").rstrip(".")


def clean_svg(svg: str, cleaning_cfg: dict[str, Any]) -> str:
    cleaned = svg.strip()
    if cleaning_cfg.get("strip_comments", False):
        cleaned = COMMENT_PATTERN.sub("", cleaned)
    if cleaning_cfg.get("strip_metadata", False):
        cleaned = METADATA_PATTERN.sub("", cleaned)
    decimals = cleaning_cfg.get("round_coordinates_decimals")
    if isinstance(decimals, int) and decimals >= 0:
        cleaned = DECIMAL_PATTERN.sub(lambda m: round_decimal_match(m, decimals), cleaned)
    if cleaning_cfg.get("collapse_whitespace", False):
        cleaned = re.sub(r">\s+<", "><", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def validate_svg(svg: str, validation_cfg: dict[str, Any]) -> tuple[bool, str | None]:
    if validation_cfg.get("require_xml_parse", False):
        try:
            root = etree.fromstring(svg.encode("utf-8"))
        except etree.XMLSyntaxError as exc:
            return False, f"xml_parse_error:{exc}"
    else:
        root = None

    if validation_cfg.get("require_svg_root", False):
        if root is None:
            try:
                root = etree.fromstring(svg.encode("utf-8"))
            except etree.XMLSyntaxError as exc:
                return False, f"xml_parse_error:{exc}"
        tag = root.tag.lower()
        if not tag.endswith("svg"):
            return False, f"unexpected_root:{root.tag}"

    return True, None


def histogram(values: list[int], bins: int = 10) -> list[dict[str, int]]:
    if not values:
        return []
    lower = min(values)
    upper = max(values)
    if lower == upper:
        return [{"start": lower, "end": upper, "count": len(values)}]
    width = max(1, math.ceil((upper - lower + 1) / bins))
    buckets: list[dict[str, int]] = []
    for idx in range(bins):
        start = lower + idx * width
        end = start + width - 1
        buckets.append({"start": start, "end": end, "count": 0})
    for value in values:
        bucket_idx = min((value - lower) // width, bins - 1)
        buckets[bucket_idx]["count"] += 1
    return [bucket for bucket in buckets if bucket["count"] > 0]


def compute_length_stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "histogram": histogram(values),
    }


def split_records(
    records: list[ProcessedSvg],
    splits_cfg: dict[str, Any],
    seed: int = 42,
) -> dict[str, list[ProcessedSvg]]:
    train_ratio = float(splits_cfg["train"])
    val_ratio = float(splits_cfg["val"])
    test_ratio = float(splits_cfg["test"])
    total_ratio = train_ratio + val_ratio + test_ratio
    if not math.isclose(total_ratio, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("Train/val/test ratios must sum to 1.")

    shuffled = records[:]
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    splits = {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }
    if not splits["val"] and splits["test"] and len(shuffled) >= 2:
        splits["val"].append(splits["test"].pop(0))
    if not splits["test"] and splits["val"] and len(shuffled) >= 2:
        splits["test"].append(splits["val"].pop())
    return splits


def to_serializable_rows(records: list[ProcessedSvg]) -> list[dict[str, Any]]:
    return [
        {
            "dataset_name": record.dataset_name,
            "source_id": record.source_id,
            "source_split": record.source_split,
            "svg": record.svg,
            "char_length": record.char_length,
            "estimated_tokens": record.estimated_tokens,
        }
        for record in records
    ]


def summarize_records(
    dataset_names: list[str],
    total_seen: int,
    rejection_reasons: dict[str, int],
    records: list[ProcessedSvg],
    split_records_map: dict[str, list[ProcessedSvg]],
) -> dict[str, Any]:
    char_lengths = [record.char_length for record in records]
    token_lengths = [record.estimated_tokens for record in records]
    split_summary: dict[str, Any] = {}
    for split_name, split_records_list in split_records_map.items():
        split_summary[split_name] = {
            "count": len(split_records_list),
            "estimated_tokens_total": sum(r.estimated_tokens for r in split_records_list),
            "char_length": compute_length_stats([r.char_length for r in split_records_list]),
            "estimated_tokens": compute_length_stats([r.estimated_tokens for r in split_records_list]),
        }

    return {
        "dataset_names": dataset_names,
        "total_seen": total_seen,
        "total_kept": len(records),
        "rejection_reasons": rejection_reasons,
        "overall": {
            "char_length": compute_length_stats(char_lengths),
            "estimated_tokens": compute_length_stats(token_lengths),
        },
        "splits": split_summary,
    }


def persist_dataset_snapshot(
    dataset_name: str,
    raw_rows: list[dict[str, Any]],
    processed_records: list[ProcessedSvg],
    split_records_map: dict[str, list[ProcessedSvg]],
    summary: dict[str, Any],
) -> dict[str, Path]:
    paths = dataset_paths(dataset_name)
    raw_dir = paths["raw_dir"]
    processed_dir = paths["processed_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(raw_dir / "raw.jsonl", raw_rows)
    write_json(processed_dir / "summary.json", summary)
    write_jsonl(processed_dir / "all.jsonl", to_serializable_rows(processed_records))

    for split_name, split_list in split_records_map.items():
        write_jsonl(processed_dir / f"{split_name}.jsonl", to_serializable_rows(split_list))

    return {
        "raw_jsonl": raw_dir / "raw.jsonl",
        "processed_dir": processed_dir,
        "summary_json": processed_dir / "summary.json",
    }
