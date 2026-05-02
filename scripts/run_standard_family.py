from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml_svg_project.config import CONFIGS_DIR, OUTPUTS_DIR
from ml_svg_project.dataset_specs import dataset_identities_from_specs, parse_dataset_specs
from ml_svg_project.io_utils import load_yaml, read_json
from ml_svg_project.paths import ensure_project_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the standard-parameterization model family with one shared LR."
    )
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
        "--training-configs",
        type=Path,
        nargs="+",
        default=[
            CONFIGS_DIR / "training" / "tiny.yaml",
            CONFIGS_DIR / "training" / "small.yaml",
            CONFIGS_DIR / "training" / "medium.yaml",
            CONFIGS_DIR / "training" / "large.yaml",
            CONFIGS_DIR / "training" / "xl.yaml",
        ],
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--family-name",
        type=str,
        default="standard_family",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Optional override for quick validation runs. Leave 0 for config default.",
    )
    parser.add_argument(
        "--batch-size-tokens",
        type=int,
        default=0,
        help="Optional shared batch_size_tokens override for all models.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    preprocess_cfg = load_yaml(args.preprocess_config)
    tokenizer_cfg = load_yaml(args.tokenizer_config)["tokenizer"]

    family_dir = OUTPUTS_DIR / "standard_family" / args.family_name
    family_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    family_start = time.time()
    for config_path in args.training_configs:
        training_cfg = load_yaml(config_path)
        model_name = str(training_cfg["model"]["name"])
        run_name = f"{args.family_name}_{model_name}"
        print(
            f"\n{'#'*72}\n"
            f"# Launching {model_name} -> {run_name}\n"
            f"# Config: {config_path}\n"
            f"# LR: {args.learning_rate}\n"
            f"{'#'*72}",
            flush=True,
        )
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train_baseline.py"),
            "--preprocess-config",
            str(args.preprocess_config),
            "--tokenizer-config",
            str(args.tokenizer_config),
            "--training-config",
            str(config_path),
            "--run-name",
            run_name,
            "--learning-rate",
            str(args.learning_rate),
        ]
        if args.max_steps > 0:
            command.extend(["--max-steps", str(args.max_steps)])
        if args.batch_size_tokens > 0:
            command.extend(["--batch-size-tokens", str(args.batch_size_tokens)])
        model_start = time.time()
        subprocess.run(command, check=True)
        summary_path = OUTPUTS_DIR / "training_runs" / run_name / "summary.json"
        model_path = OUTPUTS_DIR / "training_runs" / run_name / "model.pt"
        summary = read_json(summary_path)
        result = {
            "model_name": model_name,
            "config_path": str(config_path),
            "run_name": run_name,
            "num_parameters": summary["num_parameters"],
            "final_val_loss": summary["final_val_loss"],
            "learning_rate": args.learning_rate,
            "summary_path": str(summary_path),
            "model_path": str(model_path),
            "elapsed_seconds": time.time() - model_start,
            "avg_tokens_per_second": summary.get("avg_tokens_per_second"),
            "peak_gpu_memory_mb": summary.get("peak_gpu_memory_mb"),
        }
        results.append(result)
        print(
            f"Completed {model_name}: val_loss={result['final_val_loss']:.4f}"
            f" | elapsed={result['elapsed_seconds']/60:.1f}min"
            f" | avg_tok_s={result['avg_tokens_per_second']:,.0f}"
            f" | peak_gpu_mb={result['peak_gpu_memory_mb']:.0f}",
            flush=True,
        )

    summary_path = family_dir / "summary.json"
    payload = {
        "family_name": args.family_name,
        "learning_rate": args.learning_rate,
        "total_elapsed_seconds": time.time() - family_start,
        "results": results,
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nFamily summary: {summary_path}")
    print(
        f"Total family elapsed: {payload['total_elapsed_seconds']/60:.1f}min",
        flush=True,
    )
    for result in results:
        print(
            f"{result['model_name']}: params={result['num_parameters']:,} "
            f"val_loss={result['final_val_loss']:.4f} "
            f"elapsed={result['elapsed_seconds']/60:.1f}min "
            f"tok/s={result['avg_tokens_per_second']:,.0f} "
            f"gpu_mb={result['peak_gpu_memory_mb']:.0f}"
        )


if __name__ == "__main__":
    main()
