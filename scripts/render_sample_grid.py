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


def placeholder_image(size: int, text: str) -> Image.Image:
    image = Image.new("RGB", (size, size), color=(245, 245, 245))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, size - 1, size - 1), outline=(180, 180, 180))
    draw.multiline_text((16, size // 2 - 20), text, fill=(80, 80, 80))
    return image


def tile_images(images: list[Image.Image], labels: list[str], cell_size: int = 256, cols: int = 4) -> Image.Image:
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
    parser = argparse.ArgumentParser(description="Render generated SVG samples to a visual grid.")
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--output-name", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_paths = sorted(args.samples_dir.glob("*.svg"))
    if not sample_paths:
        raise ValueError(f"No SVG samples found in {args.samples_dir}")

    output_dir = PROJECT_ROOT / "outputs" / "figures" / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    images: list[Image.Image] = []
    labels: list[str] = []
    rows: list[dict[str, object]] = []
    for sample_path in sample_paths:
        svg_text = sample_path.read_text(encoding="utf-8")
        png_path = output_dir / f"{sample_path.stem}.png"
        status = "rendered"
        try:
            png_bytes = render_svg_to_png(svg_text)
            png_path.write_bytes(png_bytes)
            image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        except Exception:
            status = "invalid"
            image = placeholder_image(256, "Invalid SVG")
        images.append(image)
        labels.append(f"{sample_path.stem} [{status}]")
        rows.append(
            {
                "sample_path": str(sample_path),
                "png_path": str(png_path) if status == "rendered" else None,
                "status": status,
            }
        )

    grid = tile_images(images, labels)
    grid_path = output_dir / "sample_grid.png"
    grid.save(grid_path)

    summary = {
        "samples_dir": str(args.samples_dir),
        "grid_path": str(grid_path),
        "samples": rows,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Sample grid summary: {summary_path}")
    print(f"Grid: {grid_path}")


if __name__ == "__main__":
    main()
