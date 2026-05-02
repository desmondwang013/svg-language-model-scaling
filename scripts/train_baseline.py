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
from ml_svg_project.io_utils import load_yaml, read_json
from ml_svg_project.paths import ensure_project_dirs
from ml_svg_project.tokenization import encoded_artifact_dir
from ml_svg_project.training import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline decoder-only transformer.")
    parser.add_argument(
        "--preprocess-config",
        type=Path,
        default=CONFIGS_DIR / "preprocessing" / "icons_plus_fonts_75k.yaml",
    )
    parser.add_argument(
        "--tokenizer-config",
        type=Path,
        default=CONFIGS_DIR / "tokenizer" / "hf_bpe_4096.yaml",
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=CONFIGS_DIR / "training" / "tiny.yaml",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="baseline_tiny",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0,
        help="Optional optimizer LR override. Leave 0 to use config value.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Optional training max_steps override. Leave 0 to use config value.",
    )
    parser.add_argument(
        "--batch-size-tokens",
        type=int,
        default=0,
        help="Optional training batch_size_tokens override. Leave 0 to use config value.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    preprocess_cfg = load_yaml(args.preprocess_config)
    tokenizer_cfg = load_yaml(args.tokenizer_config)["tokenizer"]
    training_cfg = load_yaml(args.training_config)

    dataset_ids = dataset_identities_from_specs(parse_dataset_specs(preprocess_cfg["dataset"]))
    encoded_dir = encoded_artifact_dir(dataset_ids, tokenizer_cfg)
    summary = read_json(encoded_dir / "summary.json")

    training_cfg["model"]["vocab_size"] = int(tokenizer_cfg["vocab_size"])
    if args.learning_rate > 0:
        training_cfg["optimizer"]["learning_rate"] = args.learning_rate
    if args.max_steps > 0:
        training_cfg.setdefault("training", {})
        training_cfg["training"]["max_steps"] = args.max_steps
    if args.batch_size_tokens > 0:
        training_cfg.setdefault("training", {})
        training_cfg["training"]["batch_size_tokens"] = args.batch_size_tokens
    artifacts = run_training(encoded_dir=encoded_dir, training_cfg=training_cfg, run_name=args.run_name)

    print(f"Encoded input: {encoded_dir}")
    print(f"Train tokens: {summary['splits']['train']['token_count_total']}")
    print(f"Run directory: {artifacts.run_dir}")
    print(f"Metrics: {artifacts.metrics_path}")
    print(f"Model: {artifacts.model_path}")


if __name__ == "__main__":
    main()
