from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml_svg_project.inference import evaluate_svg_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a directory of generated SVG samples.")
    parser.add_argument("--samples-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_paths = sorted(args.samples_dir.glob("*.svg"))
    if not sample_paths:
        raise ValueError(f"No SVG samples found in {args.samples_dir}")

    rows: list[dict[str, object]] = []
    xml_valid_count = 0
    render_success_count = 0
    structural_valid_count = 0
    svg_root_count = 0
    valid_attribute_value_count = 0
    for sample_path in sample_paths:
        svg_text = sample_path.read_text(encoding="utf-8")
        result = evaluate_svg_text(svg_text)
        if result["valid_xml"]:
            xml_valid_count += 1
        if result["structural_valid"]:
            structural_valid_count += 1
        if result["structural_checks"]["has_svg_root"]:
            svg_root_count += 1
        if result["structural_checks"]["valid_attribute_values"]:
            valid_attribute_value_count += 1
        if result["rendered_png"]:
            render_success_count += 1
        rows.append({"sample_path": str(sample_path), **result})

    total = len(rows)
    summary = {
        "samples_dir": str(args.samples_dir),
        "sample_count": total,
        "xml_valid_count": xml_valid_count,
        "xml_valid_rate": xml_valid_count / total,
        "structural_valid_count": structural_valid_count,
        "structural_valid_rate": structural_valid_count / total,
        "svg_root_count": svg_root_count,
        "svg_root_rate": svg_root_count / total,
        "valid_attribute_value_count": valid_attribute_value_count,
        "valid_attribute_value_rate": valid_attribute_value_count / total,
        "render_success_count": render_success_count,
        "render_success_rate": render_success_count / total,
        "samples": rows,
    }
    summary_path = args.samples_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Evaluation summary: {summary_path}")


if __name__ == "__main__":
    main()
