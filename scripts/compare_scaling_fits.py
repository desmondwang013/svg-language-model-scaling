from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml_svg_project.config import OUTPUTS_DIR
from ml_svg_project.io_utils import read_json


def power_law(params: np.ndarray | float, a: float, alpha: float, c: float) -> np.ndarray:
    n = np.asarray(params, dtype=float)
    return a * np.power(n, -alpha) + c


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare standard and uP scaling fits.")
    parser.add_argument(
        "--standard-fit",
        type=Path,
        default=OUTPUTS_DIR / "scaling_fits" / "standard_scaling_step5_75k" / "fit_summary.json",
    )
    parser.add_argument(
        "--mup-fit",
        type=Path,
        default=OUTPUTS_DIR / "scaling_fits" / "mup_scaling_step6_75k" / "fit_summary.json",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="standard_vs_mup_step6_75k",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    standard = read_json(args.standard_fit)
    mup = read_json(args.mup_fit)

    standard_params = np.array([row["num_parameters"] for row in standard["results"]], dtype=float)
    standard_losses = np.array([row["final_val_loss"] for row in standard["results"]], dtype=float)
    mup_params = np.array([row["num_parameters"] for row in mup["results"]], dtype=float)
    mup_losses = np.array([row["final_val_loss"] for row in mup["results"]], dtype=float)

    std_coeffs = standard["parameters"]
    mup_coeffs = mup["parameters"]

    x_min = float(min(standard_params.min(), mup_params.min()))
    x_max = float(max(standard_params.max(), mup_params.max()))
    x_dense = np.logspace(np.log10(x_min), np.log10(x_max), 400)
    y_standard = power_law(
        x_dense,
        float(std_coeffs["a"]),
        float(std_coeffs["alpha"]),
        float(std_coeffs["c"]),
    )
    y_mup = power_law(
        x_dense,
        float(mup_coeffs["a"]),
        float(mup_coeffs["alpha"]),
        float(mup_coeffs["c"]),
    )

    output_dir = OUTPUTS_DIR / "scaling_fits" / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_summary = {
        "standard_fit": str(args.standard_fit),
        "mup_fit": str(args.mup_fit),
        "standard_alpha": float(std_coeffs["alpha"]),
        "mup_alpha": float(mup_coeffs["alpha"]),
        "standard_r2": float(standard["r2"]),
        "mup_r2": float(mup["r2"]),
        "standard_extrapolated_loss_10x": float(standard["extrapolated_loss_10x"]),
        "mup_extrapolated_loss_10x": float(mup["extrapolated_loss_10x"]),
    }
    summary_path = output_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps(comparison_summary, indent=2), encoding="utf-8")

    plt.figure(figsize=(8, 5))
    plt.scatter(standard_params, standard_losses, color="#1f77b4", label="Standard observed")
    plt.plot(x_dense, y_standard, color="#1f77b4", linestyle="-", label="Standard fit")
    plt.scatter(mup_params, mup_losses, color="#d62728", label="uP observed")
    plt.plot(x_dense, y_mup, color="#d62728", linestyle="-", label="uP fit")
    plt.xscale("log")
    plt.xlabel("Number of Parameters")
    plt.ylabel("Validation Loss")
    plt.title("Standard vs uP Scaling Comparison")
    plt.legend()
    plt.tight_layout()
    plot_path = output_dir / "comparison_curve.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"Comparison summary: {summary_path}")
    print(f"Comparison plot: {plot_path}")


if __name__ == "__main__":
    main()
