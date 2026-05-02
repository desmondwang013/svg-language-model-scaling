from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from .config import ARTIFACTS_DIR, DATA_DIR
from .io_utils import read_json, read_jsonl, write_json, write_jsonl
from .preprocessing import combined_dataset_name, histogram, slugify_dataset_name


def processed_dataset_dir(dataset_names: list[str]) -> Path:
    dataset_key = slugify_dataset_name(combined_dataset_name(dataset_names))
    return DATA_DIR / "processed" / dataset_key


def tokenizer_artifact_dir(dataset_names: list[str], tokenizer_cfg: dict[str, Any]) -> Path:
    dataset_key = slugify_dataset_name(combined_dataset_name(dataset_names))
    backend = tokenizer_cfg["backend"]
    vocab_size = tokenizer_cfg["vocab_size"]
    return ARTIFACTS_DIR / "tokenizer" / f"{dataset_key}__{backend}_{vocab_size}"


def encoded_artifact_dir(dataset_names: list[str], tokenizer_cfg: dict[str, Any]) -> Path:
    dataset_key = slugify_dataset_name(combined_dataset_name(dataset_names))
    backend = tokenizer_cfg["backend"]
    vocab_size = tokenizer_cfg["vocab_size"]
    return ARTIFACTS_DIR / "encoded" / f"{dataset_key}__{backend}_{vocab_size}"


def load_processed_splits(dataset_names: list[str]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    base_dir = processed_dataset_dir(dataset_names)
    summary = read_json(base_dir / "summary.json")
    splits = {
        "train": read_jsonl(base_dir / "train.jsonl"),
        "val": read_jsonl(base_dir / "val.jsonl"),
        "test": read_jsonl(base_dir / "test.jsonl"),
    }
    return splits, summary


def write_training_corpus(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(row["svg"])
            handle.write("\n")


def sentencepiece_special_tokens(tokenizer_cfg: dict[str, Any]) -> dict[str, int]:
    use_bos = bool(tokenizer_cfg.get("bos", True))
    use_eos = bool(tokenizer_cfg.get("eos", True))
    use_pad = bool(tokenizer_cfg.get("pad", False))
    return {
        "bos_id": 1 if use_bos else -1,
        "eos_id": 2 if use_eos else -1,
        "pad_id": 3 if use_pad else -1,
        "unk_id": 0,
    }


def train_sentencepiece_tokenizer(
    corpus_path: Path,
    output_prefix: Path,
    tokenizer_cfg: dict[str, Any],
) -> tuple[Path, Path]:
    import sentencepiece as spm

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    special = sentencepiece_special_tokens(tokenizer_cfg)
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(output_prefix),
        model_type=str(tokenizer_cfg["model_type"]),
        vocab_size=int(tokenizer_cfg["vocab_size"]),
        character_coverage=float(tokenizer_cfg.get("character_coverage", 1.0)),
        max_sentence_length=int(tokenizer_cfg.get("max_sentence_length", 4192)),
        normalization_rule_name="identity",
        split_by_whitespace=False,
        split_digits=False,
        byte_fallback=False,
        bos_id=special["bos_id"],
        eos_id=special["eos_id"],
        pad_id=special["pad_id"],
        unk_id=special["unk_id"],
    )
    return output_prefix.with_suffix(".model"), output_prefix.with_suffix(".vocab")


def train_hf_bpe_tokenizer(
    corpus_path: Path,
    output_dir: Path,
    tokenizer_cfg: dict[str, Any],
) -> Path:
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.processors import TemplateProcessing
    from tokenizers.trainers import BpeTrainer

    output_dir.mkdir(parents=True, exist_ok=True)
    special_tokens = ["<unk>"]
    if tokenizer_cfg.get("bos", True):
        special_tokens.append("<s>")
    if tokenizer_cfg.get("eos", True):
        special_tokens.append("</s>")
    if tokenizer_cfg.get("pad", False):
        special_tokens.append("<pad>")

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=int(tokenizer_cfg["vocab_size"]),
        special_tokens=special_tokens,
        show_progress=True,
    )
    tokenizer.train([str(corpus_path)], trainer)

    vocab = tokenizer.get_vocab()
    bos_token = "<s>" if "<s>" in vocab else None
    eos_token = "</s>" if "</s>" in vocab else None
    if bos_token and eos_token:
        tokenizer.post_processor = TemplateProcessing(
            single=f"{bos_token} $A {eos_token}",
            special_tokens=[
                (bos_token, vocab[bos_token]),
                (eos_token, vocab[eos_token]),
            ],
        )
    elif eos_token:
        tokenizer.post_processor = TemplateProcessing(
            single=f"$A {eos_token}",
            special_tokens=[(eos_token, vocab[eos_token])],
        )

    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    return tokenizer_path


def encode_with_sentencepiece(
    model_path: Path,
    rows: list[dict[str, Any]],
    add_bos: bool,
    add_eos: bool,
) -> list[dict[str, Any]]:
    import sentencepiece as spm

    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    encoded_rows: list[dict[str, Any]] = []
    for row in rows:
        token_ids = processor.encode(row["svg"], out_type=int, add_bos=add_bos, add_eos=add_eos)
        encoded_rows.append(
            {
                "dataset_name": row.get("dataset_name"),
                "source_id": row["source_id"],
                "source_split": row["source_split"],
                "char_length": row["char_length"],
                "estimated_tokens": row["estimated_tokens"],
                "token_count": len(token_ids),
                "token_ids": token_ids,
            }
        )
    return encoded_rows


def encode_with_hf_tokenizer(
    tokenizer_path: Path,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    encoded_rows: list[dict[str, Any]] = []
    for row in rows:
        encoded = tokenizer.encode(row["svg"])
        token_ids = encoded.ids
        encoded_rows.append(
            {
                "dataset_name": row.get("dataset_name"),
                "source_id": row["source_id"],
                "source_split": row["source_split"],
                "char_length": row["char_length"],
                "estimated_tokens": row["estimated_tokens"],
                "token_count": len(token_ids),
                "token_ids": token_ids,
            }
        )
    return encoded_rows


def length_stats(values: list[int]) -> dict[str, Any]:
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


def build_tokenization_summary(
    dataset_names: list[str],
    tokenizer_cfg: dict[str, Any],
    encoded_splits: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    split_summary: dict[str, Any] = {}
    for split_name, rows in encoded_splits.items():
        token_lengths = [row["token_count"] for row in rows]
        split_summary[split_name] = {
            "count": len(rows),
            "token_count_total": sum(token_lengths),
            "token_length": length_stats(token_lengths),
        }
    return {
        "dataset_names": dataset_names,
        "tokenizer": tokenizer_cfg,
        "splits": split_summary,
    }


def persist_encoded_splits(
    output_dir: Path,
    encoded_splits: dict[str, list[dict[str, Any]]],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rows in encoded_splits.items():
        write_jsonl(output_dir / f"{split_name}.jsonl", rows)
    write_json(output_dir / "summary.json", summary)
