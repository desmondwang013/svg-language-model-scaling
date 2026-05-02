from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


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
    parser = argparse.ArgumentParser(description="Fit scaling curve for standard-family results.")
    parser.add_argument(
        "--family-summary",
        type=Path,
        default=OUTPUTS_DIR / "standard_family" / "standard_family_step4_final_75k" / "summary.json",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="standard_scaling_step5",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    family = read_json(args.family_summary)
    results = family["results"]
    params = np.array([row["num_parameters"] for row in results], dtype=float)
    losses = np.array([row["final_val_loss"] for row in results], dtype=float)
    labels = [row["model_name"] for row in results]

    initial_c = float(losses.min() * 0.9)
    initial_a = float((losses.max() - initial_c) * (params.min() ** 0.5))
    initial_alpha = 0.5

    fitted, covariance = curve_fit(
        power_law,
        params,
        losses,
        p0=(initial_a, initial_alpha, initial_c),
        bounds=([0.0, 0.0, 0.0], [np.inf, 5.0, np.inf]),
        maxfev=100000,
    )
    a, alpha, c = [float(x) for x in fitted]
    predicted = power_law(params, a, alpha, c)
    residuals = losses - predicted
    sse = float(np.sum(residuals**2))
    sst = float(np.sum((losses - losses.mean()) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else 1.0
    max_trained_parameters = float(params.max())
    extrapolated_parameters_10x = float(max_trained_parameters * 10.0)
    extrapolated_loss_10x = float(power_law(extrapolated_parameters_10x, a, alpha, c))
    rng = np.random.default_rng(42)
    sampled_params = rng.multivariate_normal(mean=np.asarray(fitted), cov=np.asarray(covariance), size=5000)
    valid_samples = sampled_params[
        (sampled_params[:, 0] > 0.0) & (sampled_params[:, 1] > 0.0) & (sampled_params[:, 2] >= 0.0)
    ]
    if len(valid_samples) > 0:
        extrapolated_samples = power_law(
            extrapolated_parameters_10x,
            valid_samples[:, 0],
            valid_samples[:, 1],
            valid_samples[:, 2],
        )
        extrapolated_ci = {
            "p2_5": float(np.quantile(extrapolated_samples, 0.025)),
            "p50": float(np.quantile(extrapolated_samples, 0.50)),
            "p97_5": float(np.quantile(extrapolated_samples, 0.975)),
            "sample_count": int(len(extrapolated_samples)),
        }
    else:
        extrapolated_ci = None

    output_dir = OUTPUTS_DIR / "scaling_fits" / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    fit_summary = {
        "family_summary": str(args.family_summary),
        "family_name": family["family_name"],
        "learning_rate": family["learning_rate"],
        "fit_function": "L = a * N^-alpha + c",
        "parameters": {
            "a": a,
            "alpha": alpha,
            "c": c,
        },
        "covariance": np.asarray(covariance).tolist(),
        "r2": r2,
        "results": results,
        "predicted_losses": predicted.tolist(),
        "max_trained_parameters": max_trained_parameters,
        "extrapolated_parameters_10x": extrapolated_parameters_10x,
        "extrapolated_loss_10x": extrapolated_loss_10x,
        "extrapolated_loss_10x_ci": extrapolated_ci,
    }
    summary_path = output_dir / "fit_summary.json"
    summary_path.write_text(json.dumps(fit_summary, indent=2), encoding="utf-8")

    x_dense = np.logspace(np.log10(params.min()), np.log10(params.max()), 300)
    y_dense = power_law(x_dense, a, alpha, c)

    plt.figure(figsize=(8, 5))
    plt.scatter(params, losses, color="#1f77b4", label="Observed")
    plt.plot(x_dense, y_dense, color="#d62728", label="Power-law fit")
    for x, y, label in zip(params, losses, labels):
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)
    plt.xscale("log")
    plt.xlabel("Number of Parameters")
    plt.ylabel("Validation Loss")
    title = str(family["family_name"]).replace("_", " ")
    plt.title(f"Scaling Curve: {title}")
    plt.legend()
    plt.tight_layout()
    plot_path = output_dir / "scaling_curve.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"Fit summary: {summary_path}")
    print(f"Plot: {plot_path}")
    print(f"alpha: {alpha}")
    print(f"r2: {r2}")


if __name__ == "__main__":
    main()
