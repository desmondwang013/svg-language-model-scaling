from __future__ import annotations

import argparse
import io
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml_svg_project.inference import render_svg_to_png
from ml_svg_project.io_utils import read_jsonl


def pick_examples(rows: list[dict[str, object]], field: str, count: int) -> list[dict[str, object]]:
    sorted_rows = sorted(rows, key=lambda row: int(row[field]))
    if count >= len(sorted_rows):
        return sorted_rows
    indices = [round(i * (len(sorted_rows) - 1) / max(1, count - 1)) for i in range(count)]
    picked: list[dict[str, object]] = []
    seen: set[int] = set()
    for idx in indices:
        if idx not in seen:
            picked.append(sorted_rows[idx])
            seen.add(idx)
    return picked


def tile_images(images: list[Image.Image], labels: list[str], cell_size: int = 256, cols: int = 3) -> Image.Image:
    rows = math.ceil(len(images) / cols)
    canvas = Image.new("RGB", (cols * cell_size, rows * (cell_size + 30)), color="white")
    draw = ImageDraw.Draw(canvas)
    for idx, (image, label) in enumerate(zip(images, labels)):
        x = (idx % cols) * cell_size
        y = (idx // cols) * (cell_size + 30)
        thumb = image.copy()
        thumb.thumbnail((cell_size - 20, cell_size - 40))
        paste_x = x + (cell_size - thumb.width) // 2
        paste_y = y + 10 + (cell_size - 40 - thumb.height) // 2
        canvas.paste(thumb, (paste_x, paste_y))
        draw.text((x + 8, y + cell_size), label, fill="black")
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render dataset SVG examples at different complexity levels.")
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--output-name", type=str, required=True)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--field", type=str, default="estimated_tokens")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.split_path)
    examples = pick_examples(rows, args.field, args.count)

    output_dir = PROJECT_ROOT / "outputs" / "figures" / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    rendered_images: list[Image.Image] = []
    labels: list[str] = []
    manifest_rows: list[dict[str, object]] = []

    for idx, row in enumerate(examples):
        svg_text = str(row["svg"])
        png_bytes = render_svg_to_png(svg_text)
        image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        rendered_images.append(image)
        label = f"id={row['source_id']} tok≈{row.get('estimated_tokens', '?')}"
        labels.append(label)
        svg_path = output_dir / f"example_{idx:02d}.svg"
        png_path = output_dir / f"example_{idx:02d}.png"
        svg_path.write_text(svg_text, encoding="utf-8")
        png_path.write_bytes(png_bytes)
        manifest_rows.append(
            {
                "index": idx,
                "source_id": row["source_id"],
                "estimated_tokens": row.get("estimated_tokens"),
                "char_length": row.get("char_length"),
                "svg_path": str(svg_path),
                "png_path": str(png_path),
            }
        )

    grid = tile_images(rendered_images, labels, cols=3)
    grid_path = output_dir / "dataset_examples_grid.png"
    grid.save(grid_path)

    summary = {
        "split_path": str(args.split_path),
        "field": args.field,
        "count": len(manifest_rows),
        "examples": manifest_rows,
        "grid_path": str(grid_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Dataset example summary: {summary_path}")
    print(f"Grid: {grid_path}")


if __name__ == "__main__":
    main()
