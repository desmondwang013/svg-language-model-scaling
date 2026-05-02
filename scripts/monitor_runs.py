from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor training runs using run_status.json and metrics.jsonl."
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "training_runs",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*",
        help="Glob pattern for run directory names.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Refresh until interrupted.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=30.0,
        help="Refresh interval in watch mode.",
    )
    parser.add_argument(
        "--stale-minutes",
        type=float,
        default=10.0,
        help="Mark a run STALE if metrics have not changed for this many minutes.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_last_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None

    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        pos = handle.tell()
        buf = bytearray()
        while pos > 0:
            pos -= 1
            handle.seek(pos)
            char = handle.read(1)
            if char == b"\n" and buf:
                break
            if char != b"\n":
                buf.extend(char)

    if not buf:
        return None
    return json.loads(buf[::-1].decode("utf-8"))


def fmt_float(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def fmt_int(value: float | int | None) -> str:
    return "-" if value is None else f"{value:,.0f}"


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    return f"{minutes/60:.1f}h"


def collect_rows(runs_dir: Path, pattern: str, stale_minutes: float) -> list[dict[str, Any]]:
    now = time.time()
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(runs_dir.glob(pattern)):
        if not run_dir.is_dir():
            continue

        status_path = run_dir / "run_status.json"
        summary_path = run_dir / "summary.json"
        metrics_path = run_dir / "metrics.jsonl"

        if not status_path.exists() and not summary_path.exists() and not metrics_path.exists():
            continue

        status = read_json(status_path) if status_path.exists() else {}
        summary = read_json(summary_path) if summary_path.exists() else None
        metric = read_last_jsonl(metrics_path)

        run_name = str(status.get("run_name") or (summary or {}).get("run_name") or run_dir.name)
        total_steps = int(status.get("total_steps") or (summary or {}).get("total_steps") or 0)
        current_step = (int(metric["step"]) + 1) if metric and "step" in metric else 0
        pct = (100.0 * current_step / total_steps) if total_steps > 0 and current_step > 0 else None

        if summary is not None:
            state = "DONE"
        elif metric is None:
            state = "PENDING"
        else:
            age_sec = now - metrics_path.stat().st_mtime
            state = "STALE" if age_sec > stale_minutes * 60 else "RUNNING"

        elapsed = (
            float(metric["elapsed_seconds"])
            if metric and "elapsed_seconds" in metric
            else (float(summary["total_elapsed_seconds"]) if summary else None)
        )
        tok_s = (
            float(metric["tokens_per_second"])
            if metric and "tokens_per_second" in metric
            else (float(summary["avg_tokens_per_second"]) if summary else None)
        )
        eta = None
        if state == "RUNNING" and elapsed is not None and current_step > 0 and total_steps > current_step:
            eta = (total_steps - current_step) * (elapsed / current_step)

        rows.append(
            {
                "state": state,
                "run_name": run_name,
                "step": current_step,
                "total_steps": total_steps,
                "pct": pct,
                "train_loss": float(metric["train_loss"]) if metric and "train_loss" in metric else None,
                "val_loss": (
                    float(metric["val_loss"])
                    if metric and "val_loss" in metric
                    else (float(summary["final_val_loss"]) if summary else None)
                ),
                "tok_s": tok_s,
                "eta": eta,
                "gpu_mb": float(summary["peak_gpu_memory_mb"]) if summary else None,
            }
        )
    return rows


def print_rows(rows: list[dict[str, Any]]) -> None:
    print(
        f"{'STATE':<8} {'RUN':<42} {'STEP':<18} {'TRAIN':>10} {'VAL':>10} "
        f"{'TOK/S':>12} {'ETA':>8} {'GPU MB':>8}"
    )
    print("-" * 126)
    for row in rows:
        step_str = "-"
        if row["total_steps"] > 0:
            pct_str = f"{row['pct']:.1f}%" if row["pct"] is not None else "-"
            step_str = f"{row['step']}/{row['total_steps']} {pct_str}"
        print(
            f"{row['state']:<8} "
            f"{row['run_name'][:42]:<42} "
            f"{step_str[:18]:<18} "
            f"{fmt_float(row['train_loss']):>10} "
            f"{fmt_float(row['val_loss']):>10} "
            f"{fmt_int(row['tok_s']):>12} "
            f"{fmt_duration(row['eta']):>8} "
            f"{fmt_int(row['gpu_mb']):>8}"
        )


def main() -> None:
    args = parse_args()
    while True:
        rows = collect_rows(args.runs_dir, args.pattern, args.stale_minutes)
        print(
            f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"runs_dir={args.runs_dir} pattern={args.pattern}"
        )
        if rows:
            print_rows(rows)
        else:
            print("No matching runs found.")

        if not args.watch:
            break
        time.sleep(max(1.0, args.interval_seconds))


if __name__ == "__main__":
    main()
