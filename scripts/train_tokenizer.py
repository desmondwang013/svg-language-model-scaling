from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml_svg_project.config import CONFIGS_DIR
from ml_svg_project.dataset_specs import dataset_identities_from_specs, parse_dataset_specs
from ml_svg_project.io_utils import load_yaml
from ml_svg_project.paths import ensure_project_dirs
from ml_svg_project.tokenization import (
    build_tokenization_summary,
    encoded_artifact_dir,
    encode_with_hf_tokenizer,
    encode_with_sentencepiece,
    load_processed_splits,
    persist_encoded_splits,
    processed_dataset_dir,
    train_hf_bpe_tokenizer,
    tokenizer_artifact_dir,
    train_sentencepiece_tokenizer,
    write_training_corpus,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tokenizer and encode processed SVG splits.")
    parser.add_argument(
        "--preprocess-config",
        type=Path,
        default=CONFIGS_DIR / "preprocessing" / "default.yaml",
        help="Path to preprocessing config used to identify processed dataset artifacts.",
    )
    parser.add_argument(
        "--tokenizer-config",
        type=Path,
        default=CONFIGS_DIR / "tokenizer" / "default.yaml",
        help="Path to tokenizer config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    preprocess_cfg = load_yaml(args.preprocess_config)
    tokenizer_cfg = load_yaml(args.tokenizer_config)["tokenizer"]
    dataset_names = dataset_identities_from_specs(parse_dataset_specs(preprocess_cfg["dataset"]))

    processed_dir = processed_dataset_dir(dataset_names)
    splits, _summary = load_processed_splits(dataset_names)

    artifact_dir = tokenizer_artifact_dir(dataset_names, tokenizer_cfg)
    corpus_path = artifact_dir / "train_corpus.txt"
    write_training_corpus(splits["train"], corpus_path)

    backend = tokenizer_cfg["backend"]
    if backend == "sentencepiece":
        output_prefix = artifact_dir / "sentencepiece"
        model_path, vocab_path = train_sentencepiece_tokenizer(
            corpus_path, output_prefix, tokenizer_cfg
        )
        encoded_splits = {
            split_name: encode_with_sentencepiece(
                model_path=model_path,
                rows=rows,
                add_bos=bool(tokenizer_cfg.get("bos", True)),
                add_eos=bool(tokenizer_cfg.get("eos", True)),
            )
            for split_name, rows in splits.items()
        }
        artifact_label = [("Tokenizer model", model_path), ("Tokenizer vocab", vocab_path)]
    elif backend == "hf_tokenizers":
        tokenizer_path = train_hf_bpe_tokenizer(corpus_path, artifact_dir, tokenizer_cfg)
        encoded_splits = {
            split_name: encode_with_hf_tokenizer(tokenizer_path=tokenizer_path, rows=rows)
            for split_name, rows in splits.items()
        }
        artifact_label = [("Tokenizer file", tokenizer_path)]
    else:
        raise ValueError(f"Unsupported tokenizer backend: {backend}")

    summary = build_tokenization_summary(dataset_names, tokenizer_cfg, encoded_splits)
    encoded_dir = encoded_artifact_dir(dataset_names, tokenizer_cfg)
    persist_encoded_splits(encoded_dir, encoded_splits, summary)

    print(f"Processed dataset input: {processed_dir}")
    for label, path in artifact_label:
        print(f"{label}: {path}")
    print(f"Encoded output: {encoded_dir}")
    for split_name, split_summary in summary["splits"].items():
        print(
            f"{split_name}: {split_summary['count']} samples, "
            f"{split_summary['token_count_total']} total tokens"
        )


if __name__ == "__main__":
    main()
