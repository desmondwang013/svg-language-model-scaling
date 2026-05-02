from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import OUTPUTS_DIR
from .io_utils import read_json
from .model import DecoderOnlyTransformer, TransformerConfig
from .training_data import PackedTokenDataset, pack_encoded_split, packed_split_dir


@dataclass
class TrainArtifacts:
    run_dir: Path
    metrics_path: Path
    model_path: Path
    summary_path: Path


def build_model_config(training_cfg: dict[str, Any], vocab_size: int) -> TransformerConfig:
    model_cfg = training_cfg["model"]
    return TransformerConfig(
        vocab_size=vocab_size,
        context_length=int(model_cfg["context_length"]),
        d_model=int(model_cfg["d_model"]),
        n_layers=int(model_cfg["n_layers"]),
        n_heads=int(model_cfg["n_heads"]),
        d_ff=int(model_cfg["d_ff"]),
        dropout=float(model_cfg.get("dropout", 0.1)),
        use_mup=bool(model_cfg.get("use_mup", False)),
    )


def build_aux_model_config(
    target_cfg: dict[str, Any],
    vocab_size: int,
    overrides: dict[str, Any],
) -> TransformerConfig:
    model_cfg = dict(target_cfg["model"])
    model_cfg.update(overrides)
    wrapped = {"model": model_cfg}
    return build_model_config(wrapped, vocab_size=vocab_size)


def maybe_apply_mup_shapes(
    model: DecoderOnlyTransformer,
    training_cfg: dict[str, Any],
    vocab_size: int,
) -> None:
    model_cfg = training_cfg["model"]
    if not bool(model_cfg.get("use_mup", False)):
        return
    if "mup" not in training_cfg:
        raise ValueError("mup configuration is required when use_mup is enabled.")

    import mup

    mup_cfg = training_cfg["mup"]
    base_cfg = build_aux_model_config(training_cfg, vocab_size, mup_cfg["base"])
    delta_cfg = build_aux_model_config(training_cfg, vocab_size, mup_cfg["delta"])
    base_model = DecoderOnlyTransformer(base_cfg)
    delta_model = DecoderOnlyTransformer(delta_cfg)
    mup.set_base_shapes(model, base_model, delta=delta_model)


def cosine_lr(step: int, total_steps: int, warmup_steps: int, max_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return 0.1 * max_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * 0.9 * max_lr


def ensure_packed_splits(encoded_dir: Path) -> Path:
    packed_dir = packed_split_dir(encoded_dir)
    packed_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        tokens_path = packed_dir / f"{split_name}.npy"
        meta_path = packed_dir / f"{split_name}_meta.json"
        if tokens_path.exists() and meta_path.exists():
            continue
        pack_encoded_split(encoded_dir / f"{split_name}.jsonl", packed_dir, split_name)
    return packed_dir


def load_packed_dataset(packed_dir: Path, split_name: str) -> PackedTokenDataset:
    metadata = read_json(packed_dir / f"{split_name}_meta.json")
    return PackedTokenDataset(packed_dir / f"{split_name}.npy", metadata["dtype"])


def run_training(
    encoded_dir: Path,
    training_cfg: dict[str, Any],
    run_name: str,
) -> TrainArtifacts:
    encoded_summary = read_json(encoded_dir / "summary.json")
    vocab_size = int(training_cfg.get("vocab_size_override") or training_cfg["model"]["vocab_size"])
    if vocab_size <= 0:
        raise ValueError("Vocab size must be set in the training config.")

    packed_dir = ensure_packed_splits(encoded_dir)
    train_data = load_packed_dataset(packed_dir, "train")
    val_data = load_packed_dataset(packed_dir, "val")

    model_cfg = build_model_config(training_cfg, vocab_size=vocab_size)
    train_cfg = training_cfg["training"]
    optim_cfg = training_cfg["optimizer"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if use_amp:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = DecoderOnlyTransformer(model_cfg)
    maybe_apply_mup_shapes(model, training_cfg, vocab_size=vocab_size)
    model = model.to(device)

    optimizer_name = str(optim_cfg.get("name", "adamw")).lower()
    is_standard_adamw = optimizer_name == "adamw"
    optimizer_cls = torch.optim.AdamW
    if not is_standard_adamw:
        import mup
        optimizer_cls = mup.MuAdamW
    optimizer_kwargs: dict = {
        "lr": float(optim_cfg["learning_rate"]),
        "betas": tuple(float(x) for x in optim_cfg["betas"]),
        "weight_decay": float(optim_cfg["weight_decay"]),
    }
    if is_standard_adamw and use_amp:
        optimizer_kwargs["fused"] = True
    optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)

    batch_size = max(1, int(train_cfg["batch_size_tokens"]) // model_cfg.context_length)
    epoch_count = max(1, int(train_cfg.get("epochs", 1)))
    train_sequences_per_epoch = max(1, train_data.num_sequences(model_cfg.context_length))
    total_steps = max(1, math.ceil(train_sequences_per_epoch / max(1, batch_size)) * epoch_count)
    max_steps = int(train_cfg.get("max_steps", 0))
    if max_steps > 0:
        total_steps = min(total_steps, max_steps)
    warmup_steps = int(train_cfg["warmup_steps"])
    grad_clip_norm = float(train_cfg["grad_clip_norm"])
    seed = int(train_cfg["seed"])
    eval_steps = int(train_cfg.get("eval_steps", 100))
    val_batches = int(train_cfg.get("val_batches", 8))

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    run_dir = OUTPUTS_DIR / "training_runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    model_path = run_dir / "model.pt"
    summary_path = run_dir / "summary.json"

    metrics_file = metrics_path.open("w", encoding="utf-8")

    # Write status file so monitor_runs.py can compute ETA without knowing total_steps
    run_status = {
        "run_name": run_name,
        "total_steps": total_steps,
        "num_parameters": model.num_parameters(),
        "model_name": str(training_cfg["model"].get("name", "?")),
        "batch_size_tokens": batch_size * model_cfg.context_length,
        "device": str(device),
    }
    (run_dir / "run_status.json").write_text(json.dumps(run_status, indent=2), encoding="utf-8")

    print(
        f"\n{'='*64}\n"
        f"  Run   : {run_name}\n"
        f"  Model : {run_status['model_name']} | {model.num_parameters():,} params | {device}\n"
        f"  Steps : {total_steps} | batch={run_status['batch_size_tokens']:,} tokens"
        f" | epochs={epoch_count}\n"
        f"{'='*64}",
        flush=True,
    )

    print_every = int(train_cfg.get("print_steps", 50))
    start_time = time.time()

    def log_metric(payload: dict[str, Any]) -> None:
        metrics_file.write(json.dumps(payload))
        metrics_file.write("\n")
        metrics_file.flush()

    def evaluate() -> float:
        model.eval()
        losses: list[float] = []
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            if val_batches <= 0:
                iterator = val_data.iter_sequence_batches(
                    batch_size=batch_size,
                    context_length=model_cfg.context_length,
                    shuffle=False,
                )
                for x_np, y_np in iterator:
                    x = torch.from_numpy(x_np).to(device, non_blocking=True)
                    y = torch.from_numpy(y_np).to(device, non_blocking=True)
                    _, loss = model(x, y)
                    losses.append(float(loss.item()))
            else:
                for _ in range(val_batches):
                    x_np, y_np = val_data.sample_batch(batch_size, model_cfg.context_length, rng)
                    x = torch.from_numpy(x_np).to(device, non_blocking=True)
                    y = torch.from_numpy(y_np).to(device, non_blocking=True)
                    _, loss = model(x, y)
                    losses.append(float(loss.item()))
        model.train()
        return float(sum(losses) / len(losses))

    model.train()
    tokens_seen_total = 0
    step = 0
    for _epoch in range(epoch_count):
        epoch_iterator = train_data.iter_sequence_batches(
            batch_size=batch_size,
            context_length=model_cfg.context_length,
            rng=rng,
            shuffle=True,
        )
        for x_np, y_np in epoch_iterator:
            if step >= total_steps:
                break
            lr = cosine_lr(step, total_steps, warmup_steps, float(optim_cfg["learning_rate"]))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            x = torch.from_numpy(x_np).to(device, non_blocking=True)
            y = torch.from_numpy(y_np).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            elapsed = time.time() - start_time
            batch_tokens = int(x_np.shape[0]) * model_cfg.context_length
            tokens_seen_total += batch_tokens
            metric = {
                "step": step,
                "train_loss": float(loss.item()),
                "learning_rate": lr,
                "tokens_seen": tokens_seen_total,
                "elapsed_seconds": elapsed,
                "tokens_per_second": tokens_seen_total / max(elapsed, 1e-8),
            }
            if step == 0 or (step + 1) % eval_steps == 0 or step == total_steps - 1:
                metric["val_loss"] = evaluate()
            log_metric(metric)

            if step % print_every == 0 or "val_loss" in metric:
                pct = 100.0 * (step + 1) / total_steps
                eta_sec = (total_steps - step - 1) * elapsed / max(step + 1, 1)
                val_str = f" | val={metric['val_loss']:.4f}" if "val_loss" in metric else ""
                print(
                    f"  step {step+1:>5}/{total_steps}"
                    f" ({pct:5.1f}%)"
                    f" | loss={metric['train_loss']:.4f}{val_str}"
                    f" | lr={lr:.2e}"
                    f" | {tokens_seen_total / elapsed:>9,.0f} tok/s"
                    f" | ETA {eta_sec/60:.1f}min",
                    flush=True,
                )

            step += 1
        if step >= total_steps:
            break

    final_val_loss = evaluate()
    total_elapsed = time.time() - start_time
    print(
        f"\n  {'─'*60}\n"
        f"  DONE  : {run_name}\n"
        f"  final val_loss={final_val_loss:.4f}"
        f" | elapsed={total_elapsed/60:.1f}min"
        f" | avg {tokens_seen_total/total_elapsed:,.0f} tok/s\n"
        f"  {'─'*60}",
        flush=True,
    )
    epoch_token_budget = train_sequences_per_epoch * model_cfg.context_length * epoch_count
    peak_gpu_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda"
        else 0.0
    )
    summary_payload = {
        "run_name": run_name,
        "device": str(device),
        "parameterization": "mup" if model_cfg.use_mup else "standard",
        "num_parameters": model.num_parameters(),
        "total_steps": total_steps,
        "epochs": epoch_count,
        "steps_per_epoch": math.ceil(train_sequences_per_epoch / max(1, batch_size)),
        "batch_size_sequences": batch_size,
        "batch_size_tokens": batch_size * model_cfg.context_length,
        "context_length": model_cfg.context_length,
        "learning_rate": float(optim_cfg["learning_rate"]),
        "final_val_loss": final_val_loss,
        "train_tokens_total": len(train_data),
        "tokens_per_epoch": train_sequences_per_epoch * model_cfg.context_length,
        "epoch_fraction_completed": tokens_seen_total / max(1, epoch_token_budget),
        "strict_epoch_traversal": max_steps <= 0,
        "total_elapsed_seconds": total_elapsed,
        "avg_tokens_per_second": tokens_seen_total / max(total_elapsed, 1e-8),
        "peak_gpu_memory_mb": peak_gpu_mb,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_cfg.__dict__,
            "training_config": training_cfg,
            "encoded_summary": encoded_summary,
            "num_parameters": model.num_parameters(),
            "final_val_loss": final_val_loss,
        },
        model_path,
    )
    metrics_file.close()
    return TrainArtifacts(
        run_dir=run_dir,
        metrics_path=metrics_path,
        model_path=model_path,
        summary_path=summary_path,
    )
