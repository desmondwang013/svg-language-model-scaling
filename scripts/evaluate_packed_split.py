from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml_svg_project.inference import load_model_checkpoint
from ml_svg_project.io_utils import read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate full perplexity over a packed token split.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--tokens-path", type=Path, required=True)
    parser.add_argument("--meta-path", type=Path, required=True)
    parser.add_argument("--batch-size-sequences", type=int, default=8)
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, _, device = load_model_checkpoint(args.model_path)
    meta = read_json(args.meta_path)
    tokens = np.memmap(args.tokens_path, dtype=np.dtype(meta["dtype"]), mode="r")
    context_length = model.cfg.context_length
    batch_size = max(1, args.batch_size_sequences)

    total_loss = 0.0
    total_tokens = 0
    starts = list(range(0, max(0, len(tokens) - 1), context_length))
    batch_x: list[np.ndarray] = []
    batch_y: list[np.ndarray] = []
    batch_lengths: list[int] = []

    def flush_batch() -> None:
        nonlocal total_loss, total_tokens, batch_x, batch_y, batch_lengths
        if not batch_x:
            return
        max_len = max(len(x) for x in batch_x)
        x_array = np.zeros((len(batch_x), max_len), dtype=np.int64)
        y_array = np.zeros((len(batch_y), max_len), dtype=np.int64)
        mask = np.zeros((len(batch_x), max_len), dtype=np.float32)
        for idx, (x_ids, y_ids, length) in enumerate(zip(batch_x, batch_y, batch_lengths)):
            x_array[idx, :length] = x_ids
            y_array[idx, :length] = y_ids
            mask[idx, :length] = 1.0
        x = torch.from_numpy(x_array).to(device)
        y = torch.from_numpy(y_array).to(device)
        with torch.no_grad():
            logits, _ = model(x, None)
            losses = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                reduction="none",
            ).view(len(batch_x), max_len)
            masked = losses * torch.from_numpy(mask).to(device)
            total_loss += float(masked.sum().item())
            total_tokens += int(mask.sum())
        batch_x = []
        batch_y = []
        batch_lengths = []

    for start in starts:
        end = min(start + context_length, len(tokens) - 1)
        x_ids = np.asarray(tokens[start:end], dtype=np.int64)
        y_ids = np.asarray(tokens[start + 1 : end + 1], dtype=np.int64)
        length = len(x_ids)
        if length == 0:
            continue
        batch_x.append(x_ids)
        batch_y.append(y_ids)
        batch_lengths.append(length)
        if len(batch_x) >= batch_size:
            flush_batch()
    flush_batch()

    mean_nll = total_loss / max(1, total_tokens)
    payload = {
        "model_path": str(args.model_path),
        "tokens_path": str(args.tokens_path),
        "meta_path": str(args.meta_path),
        "batch_size_sequences": batch_size,
        "context_length": context_length,
        "total_tokens_scored": total_tokens,
        "mean_nll": mean_nll,
        "perplexity": float(np.exp(mean_nll)),
    }
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
