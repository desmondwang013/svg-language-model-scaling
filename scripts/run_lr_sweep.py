from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml_svg_project.config import CONFIGS_DIR, OUTPUTS_DIR
from ml_svg_project.dataset_specs import dataset_identities_from_specs, parse_dataset_specs
from ml_svg_project.io_utils import load_yaml, read_json
from ml_svg_project.paths import ensure_project_dirs
from ml_svg_project.tokenization import encoded_artifact_dir
from ml_svg_project.training import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LR sweep on the smallest baseline model.")
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
        default=CONFIGS_DIR / "training" / "tiny_sweep.yaml",
    )
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="*",
        default=[1e-4, 2e-4, 3e-4, 5e-4, 8e-4, 1e-3],
    )
    parser.add_argument(
        "--sweep-name",
        type=str,
        default="tiny_lr_sweep",
    )
    parser.add_argument(
        "--batch-size-tokens",
        type=int,
        default=0,
        help="Optional batch_size_tokens override for all sweep runs. Leave 0 to use config value.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    preprocess_cfg = load_yaml(args.preprocess_config)
    tokenizer_cfg = load_yaml(args.tokenizer_config)["tokenizer"]
    base_training_cfg = load_yaml(args.training_config)

    dataset_ids = dataset_identities_from_specs(parse_dataset_specs(preprocess_cfg["dataset"]))
    encoded_dir = encoded_artifact_dir(dataset_ids, tokenizer_cfg)
    base_training_cfg["model"]["vocab_size"] = int(tokenizer_cfg["vocab_size"])

    sweep_dir = OUTPUTS_DIR / "lr_sweeps" / args.sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, float | str | int]] = []

    for lr in args.learning_rates:
        training_cfg = copy.deepcopy(base_training_cfg)
        training_cfg["optimizer"]["learning_rate"] = lr
        if args.batch_size_tokens > 0:
            training_cfg["training"]["batch_size_tokens"] = args.batch_size_tokens
        lr_label = f"{lr:.0e}".replace("+", "")
        run_name = f"{args.sweep_name}_{training_cfg['model']['name']}_{lr_label}"
        artifacts = run_training(encoded_dir=encoded_dir, training_cfg=training_cfg, run_name=run_name)
        summary = read_json(artifacts.summary_path)
        results.append(
            {
                "run_name": run_name,
                "learning_rate": lr,
                "final_val_loss": summary["final_val_loss"],
                "num_parameters": summary["num_parameters"],
            }
        )

    best = min(results, key=lambda row: float(row["final_val_loss"]))
    payload = {
        "sweep_name": args.sweep_name,
        "training_config": str(args.training_config),
        "results": results,
        "best_run": best,
    }
    summary_path = sweep_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Sweep summary: {summary_path}")
    print(f"Best LR: {best['learning_rate']}")
    print(f"Best final val loss: {best['final_val_loss']}")


if __name__ == "__main__":
    main()
