from __future__ import annotations

import argparse
import json
import re
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


PATH_ALLOWED_PATTERN = re.compile(r"[^MmLlHhVvCcSsQqTtAaZz0-9,.\- ]+")
WHITESPACE_PATTERN = re.compile(r"\s+")
PATH_TOKEN_PATTERN = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]|[+-]?(?:\d+(?:\.\d+)?|\.\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SVG samples with a forced structural shell.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--output-name", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument(
        "--mode",
        type=str,
        default="path_attr",
        choices=["path_attr", "svg_root"],
        help="How to force the generated content into a valid SVG shell.",
    )
    return parser.parse_args()


def sanitize_path_payload(text: str) -> str:
    cleaned = text.replace('"', " ").replace("'", " ").replace("<", " ").replace(">", " ")
    cleaned = PATH_ALLOWED_PATTERN.sub(" ", cleaned)
    tokens = PATH_TOKEN_PATTERN.findall(cleaned)
    if not tokens:
        return "M12 12"
    normalized_tokens: list[str] = []
    last_was_command = False
    for token in tokens:
        if token and token[0] in "MmLlHhVvCcSsQqTtAaZz":
            normalized_tokens.append(token[0])
            last_was_command = True
        else:
            try:
                value = float(token)
            except ValueError:
                continue
            if abs(value) > 1000:
                value = max(min(value, 1000.0), -1000.0)
            rendered = f"{value:.2f}".rstrip("0").rstrip(".")
            if rendered == "-0":
                rendered = "0"
            normalized_tokens.append(rendered)
            last_was_command = False
    if not normalized_tokens:
        return "M12 12"
    if normalized_tokens[0] not in list("Mm"):
        normalized_tokens = ["M", "12", "12"] + normalized_tokens
    if len(normalized_tokens) == 1 and normalized_tokens[0] in list("MmLlHhVvCcSsQqTtAaZz"):
        normalized_tokens.extend(["12", "12"])
    return " ".join(normalized_tokens)


def force_svg_shell(raw_text: str, mode: str) -> tuple[str, dict[str, str]]:
    if mode == "path_attr":
        payload = sanitize_path_payload(raw_text)
        svg_text = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
            f'<path fill="none" stroke="black" stroke-width="0.3" d="{payload}"/>'
            "</svg>"
        )
        return svg_text, {"sanitized_payload": payload}

    stripped = raw_text.strip()
    if stripped.lower().startswith("<svg"):
        lower = stripped.lower()
        start = stripped.find(">")
        end = lower.rfind("</svg>")
        if start != -1:
            inner = stripped[start + 1 : end if end != -1 else None].strip()
        else:
            inner = stripped
    else:
        inner = stripped
    svg_text = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
        f"{inner}"
        "</svg>"
    )
    return svg_text, {"wrapped_inner": inner[:500]}


def generation_prefix(mode: str) -> str:
    if mode == "path_attr":
        return "M"
    return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'


def main() -> None:
    args = parse_args()
    model, _, _ = load_model_checkpoint(args.model_path)
    tokenizer = load_tokenizer(args.tokenizer_path)

    output_dir = OUTPUTS_DIR / "generation" / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    prefix_text = generation_prefix(args.mode)

    for index in range(args.num_samples):
        token_ids = generate_token_ids(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            prefix_text=prefix_text,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        raw_text = decode_token_ids(tokenizer, token_ids, skip_special_tokens=True)
        forced_svg_text, extras = force_svg_shell(raw_text, args.mode)
        evaluation = evaluate_svg_text(forced_svg_text)

        sample_stem = f"sample_{index:03d}"
        sample_path = output_dir / f"{sample_stem}.svg"
        raw_path = output_dir / f"{sample_stem}.raw.txt"
        sample_path.write_text(forced_svg_text, encoding="utf-8")
        raw_path.write_text(raw_text, encoding="utf-8")

        row = {
            "index": index,
            "sample_path": str(sample_path),
            "raw_path": str(raw_path),
            "token_count": len(token_ids),
            "generation_prefix": prefix_text,
            "mode": args.mode,
            "raw_preview": raw_text[:300],
            **extras,
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
        "mode": args.mode,
        "generation_prefix": prefix_text,
        "samples": rows,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Forced generation summary: {summary_path}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
