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
from ml_svg_project.io_utils import read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble an LR sweep summary from completed run dirs.")
    parser.add_argument("--sweep-name", required=True, type=str)
    parser.add_argument("--run-prefix", required=True, type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = OUTPUTS_DIR / "training_runs"
    sweep_dir = OUTPUTS_DIR / "lr_sweeps" / args.sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    for run_dir in sorted(runs_dir.glob(f"{args.run_prefix}*")):
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = read_json(summary_path)
        results.append(
            {
                "run_name": summary["run_name"],
                "learning_rate": summary["learning_rate"],
                "final_val_loss": summary["final_val_loss"],
                "num_parameters": summary["num_parameters"],
            }
        )

    if not results:
        raise RuntimeError("No completed run summaries found for the requested sweep.")

    results.sort(key=lambda row: float(row["learning_rate"]))
    best = min(results, key=lambda row: float(row["final_val_loss"]))
    payload = {
        "sweep_name": args.sweep_name,
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
