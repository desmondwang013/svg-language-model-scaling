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
from ml_svg_project.inference import load_model_checkpoint, load_tokenizer, score_split_jsonl
from ml_svg_project.io_utils import read_json


def fmt_int(value: int) -> str:
    return f"{value:,}"


def fmt_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a report-facing experiment summary bundle.")
    parser.add_argument(
        "--output-name",
        type=str,
        default="report_summary_h100_final",
    )
    parser.add_argument(
        "--val-limit",
        type=int,
        default=25,
    )
    return parser.parse_args()


def model_table_markdown(title: str, rows: list[dict[str, object]]) -> list[str]:
    lines = [f"### {title}", "", "| Model | Parameters | Final Val Loss |", "| --- | ---: | ---: |"]
    for row in rows:
        lines.append(
            f"| {row['model_name']} | {fmt_int(int(row['num_parameters']))} | {fmt_float(float(row['final_val_loss']))} |"
        )
    lines.append("")
    return lines


def main() -> None:
    args = parse_args()
    output_dir = OUTPUTS_DIR / "report_summary" / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_summary = read_json(
        PROJECT_ROOT
        / "data/processed/starvector__svg_icons_simple__plus__starvector__svg_fonts_simple__maxrows_75000__stream/summary.json"
    )
    encoded_summary = read_json(
        PROJECT_ROOT
        / "artifacts/encoded/starvector__svg_icons_simple__plus__starvector__svg_fonts_simple__maxrows_75000__stream__hf_tokenizers_4096/summary.json"
    )
    standard_family = read_json(
        PROJECT_ROOT / "outputs/standard_family/standard_family_h100_b196608/summary.json"
    )
    mup_family = read_json(PROJECT_ROOT / "outputs/standard_family/mup_family_h100_b196608/summary.json")
    standard_fit = read_json(
        PROJECT_ROOT / "outputs/scaling_fits/standard_scaling_h100_b196608/fit_summary.json"
    )
    mup_fit = read_json(PROJECT_ROOT / "outputs/scaling_fits/mup_scaling_h100_b196608/fit_summary.json")
    comparison = read_json(
        PROJECT_ROOT / "outputs/scaling_fits/standard_vs_mup_h100_b196608/comparison_summary.json"
    )
    xl_best = read_json(PROJECT_ROOT / "outputs/training_runs/xl_best_step7/summary.json")
    unconditional_eval = read_json(
        PROJECT_ROOT / "outputs/generation/xl_best_unconditional_step7/evaluation_summary.json"
    )
    prefix_eval = read_json(
        PROJECT_ROOT / "outputs/generation/xl_best_prefix_step7/evaluation_summary.json"
    )
    full_test_path = PROJECT_ROOT / "outputs/report_summary/report_summary_step7/test_perplexity_full.json"
    full_test_scoring = read_json(full_test_path) if full_test_path.exists() else None
    generation_eval_paths = [
        PROJECT_ROOT / "outputs/generation/xl_best_uncond_t05/evaluation_summary.json",
        PROJECT_ROOT / "outputs/generation/xl_best_uncond_t08/evaluation_summary.json",
        PROJECT_ROOT / "outputs/generation/xl_best_uncond_t10/evaluation_summary.json",
        PROJECT_ROOT / "outputs/generation/xl_best_prefix_t05/evaluation_summary.json",
        PROJECT_ROOT / "outputs/generation/xl_best_prefix_t08/evaluation_summary.json",
        PROJECT_ROOT / "outputs/generation/xl_best_prefix_t10/evaluation_summary.json",
    ]
    extra_generation_evals = [read_json(path) for path in generation_eval_paths if path.exists()]

    model, _, _ = load_model_checkpoint(
        PROJECT_ROOT / "outputs/training_runs/xl_best_step7/model.pt"
    )
    tokenizer = load_tokenizer(
        PROJECT_ROOT
        / "artifacts/tokenizer/starvector__svg_icons_simple__plus__starvector__svg_fonts_simple__maxrows_75000__stream__hf_tokenizers_4096/tokenizer.json"
    )
    val_scoring = score_split_jsonl(
        model,
        tokenizer,
        PROJECT_ROOT
        / "data/processed/starvector__svg_icons_simple__plus__starvector__svg_fonts_simple__maxrows_75000__stream/val.jsonl",
        limit=args.val_limit,
    )

    payload = {
        "dataset": {
            "kept_samples": processed_summary["total_kept"],
            "train_samples": processed_summary["splits"]["train"]["count"],
            "val_samples": processed_summary["splits"]["val"]["count"],
            "test_samples": processed_summary["splits"]["test"]["count"],
            "train_tokens": encoded_summary["splits"]["train"]["token_count_total"],
            "val_tokens": encoded_summary["splits"]["val"]["token_count_total"],
            "test_tokens": encoded_summary["splits"]["test"]["token_count_total"],
            "vocab_size": encoded_summary["tokenizer"]["vocab_size"],
        },
        "standard_family": standard_family,
        "mup_family": mup_family,
        "standard_fit": standard_fit,
        "mup_fit": mup_fit,
        "comparison": comparison,
        "xl_best": xl_best,
        "xl_best_val_scoring": val_scoring,
        "xl_best_test_scoring": full_test_scoring,
        "xl_best_unconditional_eval": unconditional_eval,
        "xl_best_prefix_eval": prefix_eval,
        "temperature_generation_evals": extra_generation_evals,
    }
    json_path = output_dir / "report_summary.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Experiment Summary")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- Kept samples: `{fmt_int(processed_summary['total_kept'])}`")
    lines.append(f"- Train/val/test samples: `{fmt_int(processed_summary['splits']['train']['count'])}` / `{fmt_int(processed_summary['splits']['val']['count'])}` / `{fmt_int(processed_summary['splits']['test']['count'])}`")
    lines.append(f"- Train/val/test tokens: `{fmt_int(encoded_summary['splits']['train']['token_count_total'])}` / `{fmt_int(encoded_summary['splits']['val']['token_count_total'])}` / `{fmt_int(encoded_summary['splits']['test']['token_count_total'])}`")
    lines.append(f"- Tokenizer: `HF BPE`, vocab `{fmt_int(encoded_summary['tokenizer']['vocab_size'])}`")
    lines.append("")
    lines.extend(model_table_markdown("Standard Parameterization", standard_family["results"]))
    lines.extend(model_table_markdown("uP", mup_family["results"]))
    lines.append("## Scaling Fits")
    lines.append("")
    lines.append(f"- Standard alpha: `{fmt_float(float(standard_fit['parameters']['alpha']), 5)}`")
    lines.append(f"- Standard R^2: `{fmt_float(float(standard_fit['r2']), 4)}`")
    lines.append(f"- Standard 10x extrapolated loss: `{fmt_float(float(standard_fit['extrapolated_loss_10x']), 4)}`")
    if standard_fit.get("extrapolated_loss_10x_ci"):
        lines.append(
            f"- Standard 10x 95% CI: `[{fmt_float(float(standard_fit['extrapolated_loss_10x_ci']['p2_5']), 4)}, {fmt_float(float(standard_fit['extrapolated_loss_10x_ci']['p97_5']), 4)}]`"
        )
    lines.append(f"- uP alpha: `{fmt_float(float(mup_fit['parameters']['alpha']), 5)}`")
    lines.append(f"- uP R^2: `{fmt_float(float(mup_fit['r2']), 4)}`")
    lines.append(f"- uP 10x extrapolated loss: `{fmt_float(float(mup_fit['extrapolated_loss_10x']), 4)}`")
    if mup_fit.get("extrapolated_loss_10x_ci"):
        lines.append(
            f"- uP 10x 95% CI: `[{fmt_float(float(mup_fit['extrapolated_loss_10x_ci']['p2_5']), 4)}, {fmt_float(float(mup_fit['extrapolated_loss_10x_ci']['p97_5']), 4)}]`"
        )
    lines.append("")
    lines.append("## Best Model")
    lines.append("")
    lines.append(f"- Run: `xl_best_step7`")
    lines.append(f"- Parameters: `{fmt_int(int(xl_best['num_parameters']))}`")
    lines.append(f"- Final validation loss: `{fmt_float(float(xl_best['final_val_loss']), 4)}`")
    lines.append(f"- Validation NLL on `{args.val_limit}` held-out samples: `{fmt_float(float(val_scoring['mean_nll']), 4)}`")
    lines.append(f"- Validation perplexity on `{args.val_limit}` held-out samples: `{fmt_float(float(val_scoring['perplexity']), 4)}`")
    if full_test_scoring is not None:
        lines.append(f"- Test-set NLL: `{fmt_float(float(full_test_scoring['mean_nll']), 4)}`")
        lines.append(f"- Test-set perplexity: `{fmt_float(float(full_test_scoring['perplexity']), 4)}`")
    lines.append("")
    lines.append("## Generation Evaluation")
    lines.append("")
    lines.append(f"- Unconditional XML-valid rate: `{fmt_float(float(unconditional_eval['xml_valid_rate']) * 100.0, 1)}%`")
    lines.append(f"- Unconditional structural-valid rate: `{fmt_float(float(unconditional_eval['structural_valid_rate']) * 100.0, 1)}%`")
    lines.append(f"- Unconditional render-success rate: `{fmt_float(float(unconditional_eval['render_success_rate']) * 100.0, 1)}%`")
    lines.append(f"- Prefix-conditioned XML-valid rate: `{fmt_float(float(prefix_eval['xml_valid_rate']) * 100.0, 1)}%`")
    lines.append(f"- Prefix-conditioned structural-valid rate: `{fmt_float(float(prefix_eval['structural_valid_rate']) * 100.0, 1)}%`")
    lines.append(f"- Prefix-conditioned render-success rate: `{fmt_float(float(prefix_eval['render_success_rate']) * 100.0, 1)}%`")
    if extra_generation_evals:
        total_temp_samples = sum(int(row["sample_count"]) for row in extra_generation_evals)
        lines.append(f"- Additional temperature-study samples saved: `{total_temp_samples}`")
    lines.append("")
    lines.append("## Main Takeaway")
    lines.append("")
    lines.append("- Under the exact 1-epoch H100 protocol, standard parameterization outperformed `uP` on observed loss, fitted scaling exponent, and 10x extrapolated loss.")
    lines.append("- The extrapolation study is included directly in the scaling-fit outputs through the 10x loss estimates and their uncertainty intervals.")
    lines.append("- The longer `xl_best_step7` run remains the best available generation model in this repo and substantially improved held-out perplexity, but generated SVGs were still not XML-valid or renderable.")
    lines.append("")

    md_path = output_dir / "Experiment Summary.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    csv_lines = ["family,model_name,num_parameters,final_val_loss"]
    for row in standard_family["results"]:
        csv_lines.append(f"standard,{row['model_name']},{row['num_parameters']},{row['final_val_loss']}")
    for row in mup_family["results"]:
        csv_lines.append(f"mup,{row['model_name']},{row['num_parameters']},{row['final_val_loss']}")
    csv_path = output_dir / "model_results.csv"
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    print(f"Report summary JSON: {json_path}")
    print(f"Report summary Markdown: {md_path}")
    print(f"Model results CSV: {csv_path}")


if __name__ == "__main__":
    main()
