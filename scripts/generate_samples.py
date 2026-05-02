from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml_svg_project.config import OUTPUTS_DIR
from ml_svg_project.inference import (
    decode_token_ids,
    evaluate_svg_text,
    generate_token_ids,
    load_model_checkpoint,
    load_tokenizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SVG samples from a trained checkpoint.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--output-name", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--prefix-text", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, _, _ = load_model_checkpoint(args.model_path)
    tokenizer = load_tokenizer(args.tokenizer_path)

    output_dir = OUTPUTS_DIR / "generation" / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for index in range(args.num_samples):
        token_ids = generate_token_ids(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            prefix_text=args.prefix_text,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        svg_text = decode_token_ids(tokenizer, token_ids, skip_special_tokens=True)
        evaluation = evaluate_svg_text(svg_text)
        sample_name = f"sample_{index:03d}.svg"
        sample_path = output_dir / sample_name
        sample_path.write_text(svg_text, encoding="utf-8")
        row = {
            "index": index,
            "sample_path": str(sample_path),
            "token_count": len(token_ids),
            "prefix_text": args.prefix_text,
            **evaluation,
        }
        rows.append(row)

    summary = {
        "output_name": args.output_name,
        "model_path": str(args.model_path),
        "tokenizer_path": str(args.tokenizer_path),
        "num_samples": args.num_samples,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "prefix_text": args.prefix_text,
        "samples": rows,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Generation summary: {summary_path}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
