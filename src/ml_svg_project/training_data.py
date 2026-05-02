from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import ARTIFACTS_DIR


@dataclass
class PackedSplitArtifacts:
    tokens_path: Path
    metadata_path: Path


def packed_split_dir(encoded_dir: Path) -> Path:
    return ARTIFACTS_DIR / "packed" / encoded_dir.name


def _iter_token_ids(jsonl_path: Path) -> Iterator[list[int]]:
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            yield row["token_ids"]


def pack_encoded_split(jsonl_path: Path, output_dir: Path, split_name: str) -> PackedSplitArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokens_path = output_dir / f"{split_name}.npy"
    metadata_path = output_dir / f"{split_name}_meta.json"

    total_tokens = 0
    max_token = 0
    for token_ids in _iter_token_ids(jsonl_path):
        total_tokens += len(token_ids)
        if token_ids:
            max_token = max(max_token, max(token_ids))

    dtype = np.uint16 if max_token < 65535 else np.uint32
    packed = np.memmap(tokens_path, dtype=dtype, mode="w+", shape=(total_tokens,))
    offset = 0
    for token_ids in _iter_token_ids(jsonl_path):
        length = len(token_ids)
        packed[offset : offset + length] = np.asarray(token_ids, dtype=dtype)
        offset += length
    packed.flush()

    metadata = {
        "split_name": split_name,
        "total_tokens": total_tokens,
        "dtype": np.dtype(dtype).name,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return PackedSplitArtifacts(tokens_path=tokens_path, metadata_path=metadata_path)


class PackedTokenDataset:
    def __init__(self, tokens_path: Path, dtype: str) -> None:
        self.tokens_path = tokens_path
        self.dtype = np.dtype(dtype)
        # Load fully into RAM — eliminates random-access page faults during shuffled training.
        self.tokens = np.fromfile(tokens_path, dtype=self.dtype)

    def __len__(self) -> int:
        return int(self.tokens.shape[0])

    def sequence_start_positions(self, context_length: int) -> np.ndarray:
        if len(self) <= context_length:
            raise ValueError("Packed dataset is shorter than the context length.")
        # Use non-overlapping chunks so one epoch corresponds to one pass over the packed stream.
        return np.arange(0, len(self) - context_length, context_length, dtype=np.int64)

    def num_sequences(self, context_length: int) -> int:
        return int(self.sequence_start_positions(context_length).shape[0])

    def iter_sequence_batches(
        self,
        batch_size: int,
        context_length: int,
        rng: np.random.Generator | None = None,
        shuffle: bool = False,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        starts = self.sequence_start_positions(context_length)
        if shuffle:
            if rng is None:
                raise ValueError("rng is required when shuffle=True.")
            starts = starts.copy()
            rng.shuffle(starts)

        offsets = np.arange(context_length, dtype=np.int64)
        for batch_start in range(0, len(starts), batch_size):
            batch_positions = starts[batch_start : batch_start + batch_size]
            idx = batch_positions[:, np.newaxis] + offsets[np.newaxis, :]
            x = self.tokens[idx].astype(np.int64)
            y = self.tokens[idx + 1].astype(np.int64)
            yield x, y

    def sample_batch(
        self,
        batch_size: int,
        context_length: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(self) <= context_length:
            raise ValueError("Packed dataset is shorter than the context length.")
        starts = rng.integers(0, len(self) - context_length - 1, size=batch_size)
        offsets = np.arange(context_length, dtype=np.int64)
        idx = starts[:, np.newaxis] + offsets[np.newaxis, :]
        return self.tokens[idx].astype(np.int64), self.tokens[idx + 1].astype(np.int64)
