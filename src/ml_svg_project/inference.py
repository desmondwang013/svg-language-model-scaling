from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import cairosvg
import numpy as np
import torch
from lxml import etree
from tokenizers import Tokenizer

from .io_utils import read_jsonl
from .model import DecoderOnlyTransformer, TransformerConfig


def load_tokenizer(tokenizer_path: Path) -> Tokenizer:
    return Tokenizer.from_file(str(tokenizer_path))


def load_model_checkpoint(model_path: Path, device: torch.device | None = None) -> tuple[DecoderOnlyTransformer, dict[str, Any], torch.device]:
    resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=resolved_device)
    model_cfg = TransformerConfig(**checkpoint["model_config"])
    model = DecoderOnlyTransformer(model_cfg)
    state_dict = dict(checkpoint["model_state_dict"])
    stale_keys = [key for key in state_dict if key.endswith(".attn.causal_mask")]
    for key in stale_keys:
        state_dict.pop(key, None)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(resolved_device)
    model.eval()
    return model, checkpoint, resolved_device


def get_special_token_id(tokenizer: Tokenizer, token: str) -> int | None:
    token_id = tokenizer.token_to_id(token)
    return int(token_id) if token_id is not None else None


def encode_text(tokenizer: Tokenizer, text: str, add_special_tokens: bool = False) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=add_special_tokens).ids)


def decode_token_ids(tokenizer: Tokenizer, token_ids: list[int], skip_special_tokens: bool = True) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


def generate_token_ids(
    model: DecoderOnlyTransformer,
    tokenizer: Tokenizer,
    max_new_tokens: int,
    prefix_text: str = "",
    temperature: float = 1.0,
    top_k: int = 40,
    top_p: float = 0.0,
) -> list[int]:
    bos_id = get_special_token_id(tokenizer, "<s>")
    eos_id = get_special_token_id(tokenizer, "</s>")

    prefix_ids = encode_text(tokenizer, prefix_text, add_special_tokens=False)
    if not prefix_ids and bos_id is not None:
        prefix_ids = [bos_id]
    if not prefix_ids:
        raise ValueError("Unable to start generation because no prefix tokens or BOS token are available.")

    device = next(model.parameters()).device
    generated = list(prefix_ids)

    for _ in range(max_new_tokens):
        context = generated[-model.cfg.context_length :]
        x = torch.tensor([context], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _ = model(x, None)
        next_logits = logits[0, -1, :]
        if temperature <= 0:
            next_token = int(torch.argmax(next_logits).item())
        else:
            scaled = next_logits / temperature
            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
                sorted_probs = torch.softmax(sorted_logits, dim=-1)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                keep_mask = cumulative <= top_p
                keep_mask[0] = True
                kept_logits = sorted_logits[keep_mask]
                kept_indices = sorted_indices[keep_mask]
                probs = torch.softmax(kept_logits, dim=-1)
                chosen = int(torch.multinomial(probs, num_samples=1).item())
                next_token = int(kept_indices[chosen].item())
            elif top_k > 0:
                top_values, top_indices = torch.topk(scaled, k=min(top_k, scaled.size(-1)))
                probs = torch.softmax(top_values, dim=-1)
                chosen = int(torch.multinomial(probs, num_samples=1).item())
                next_token = int(top_indices[chosen].item())
            else:
                probs = torch.softmax(scaled, dim=-1)
                next_token = int(torch.multinomial(probs, num_samples=1).item())
        generated.append(next_token)
        if eos_id is not None and next_token == eos_id:
            break
    return generated


def score_text_nll(
    model: DecoderOnlyTransformer,
    tokenizer: Tokenizer,
    text: str,
) -> dict[str, float]:
    token_ids = encode_text(tokenizer, text, add_special_tokens=True)
    if len(token_ids) < 2:
        raise ValueError("Need at least two tokens to score sequence likelihood.")
    device = next(model.parameters()).device
    max_len = model.cfg.context_length
    losses: list[float] = []
    token_total = 0
    with torch.no_grad():
        start = 0
        while start < len(token_ids) - 1:
            end = min(start + max_len, len(token_ids) - 1)
            x_ids = token_ids[start:end]
            y_ids = token_ids[start + 1 : end + 1]
            x = torch.tensor([x_ids], dtype=torch.long, device=device)
            y = torch.tensor([y_ids], dtype=torch.long, device=device)
            _, loss = model(x, y)
            chunk_len = len(y_ids)
            losses.append(float(loss.item()) * chunk_len)
            token_total += chunk_len
            start = end
    nll = float(sum(losses) / max(1, token_total))
    return {
        "nll": nll,
        "perplexity": float(math.exp(nll)),
        "token_count": float(token_total + 1),
    }


def score_split_jsonl(
    model: DecoderOnlyTransformer,
    tokenizer: Tokenizer,
    split_path: Path,
    limit: int = 100,
) -> dict[str, float]:
    rows = read_jsonl(split_path)
    if limit > 0:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"No rows found in split file: {split_path}")
    losses: list[float] = []
    token_counts: list[int] = []
    for row in rows:
        scored = score_text_nll(model, tokenizer, row["svg"])
        losses.append(scored["nll"])
        token_counts.append(int(scored["token_count"]))
    mean_nll = float(sum(losses) / len(losses))
    return {
        "samples_scored": float(len(rows)),
        "mean_nll": mean_nll,
        "perplexity": float(math.exp(mean_nll)),
        "mean_token_count": float(sum(token_counts) / len(token_counts)),
    }


def is_valid_xml(svg_text: str) -> tuple[bool, str | None]:
    try:
        root = etree.fromstring(svg_text.encode("utf-8"))
    except etree.XMLSyntaxError as exc:
        return False, f"xml_parse_error:{exc}"
    tag = root.tag.lower()
    if not tag.endswith("svg"):
        return False, "root_not_svg"
    return True, None


NUMERIC_ATTRIBUTE_NAMES = {
    "x",
    "y",
    "cx",
    "cy",
    "r",
    "rx",
    "ry",
    "x1",
    "x2",
    "y1",
    "y2",
    "width",
    "height",
    "stroke-width",
    "opacity",
    "stroke-opacity",
    "fill-opacity",
}
NUMERIC_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:px|pt|pc|mm|cm|in|em|rem|%)?$")
COLOR_PATTERN = re.compile(r"^(?:#[0-9A-Fa-f]{3,8}|[A-Za-z]+|none|currentColor)$")


def validate_structural_rules(root: etree._Element) -> tuple[bool, dict[str, bool], list[str]]:
    checks = {
        "has_svg_root": bool(root.tag.lower().endswith("svg")),
        "closed_tags": True,
        "valid_attribute_values": True,
    }
    reasons: list[str] = []
    for element in root.iter():
        for attr_name, attr_value in element.attrib.items():
            name = attr_name.lower()
            value = attr_value.strip()
            if not value:
                checks["valid_attribute_values"] = False
                reasons.append(f"empty_attribute:{attr_name}")
                continue
            if name == "viewbox":
                parts = value.replace(",", " ").split()
                if len(parts) != 4:
                    checks["valid_attribute_values"] = False
                    reasons.append("invalid_viewBox_arity")
                    continue
                try:
                    [float(part) for part in parts]
                except ValueError:
                    checks["valid_attribute_values"] = False
                    reasons.append("invalid_viewBox_numeric")
                continue
            if name in {"fill", "stroke"}:
                if not COLOR_PATTERN.match(value):
                    checks["valid_attribute_values"] = False
                    reasons.append(f"invalid_color:{attr_name}")
                continue
            if name in NUMERIC_ATTRIBUTE_NAMES and not NUMERIC_PATTERN.match(value):
                checks["valid_attribute_values"] = False
                reasons.append(f"invalid_numeric:{attr_name}")
                continue
            if name == "d" and not value:
                checks["valid_attribute_values"] = False
                reasons.append("empty_path_d")
    structural_valid = all(checks.values())
    return structural_valid, checks, reasons


def render_svg_to_png(svg_text: str) -> bytes:
    return cairosvg.svg2png(bytestring=svg_text.encode("utf-8"))


def evaluate_svg_text(svg_text: str) -> dict[str, Any]:
    valid_xml, xml_reason = is_valid_xml(svg_text)
    structural_valid = False
    structural_checks = {
        "has_svg_root": False,
        "closed_tags": False,
        "valid_attribute_values": False,
    }
    structural_reasons: list[str] = []
    rendered = False
    render_reason: str | None = None
    png_size: int | None = None
    if valid_xml:
        root = etree.fromstring(svg_text.encode("utf-8"))
        structural_valid, structural_checks, structural_reasons = validate_structural_rules(root)
        try:
            png_bytes = render_svg_to_png(svg_text)
            rendered = True
            png_size = len(png_bytes)
        except Exception as exc:  # pragma: no cover - renderer exceptions are library-defined
            render_reason = f"render_error:{exc}"
    else:
        render_reason = "skipped_due_to_invalid_xml"
    return {
        "valid_xml": valid_xml,
        "xml_reason": xml_reason,
        "structural_valid": structural_valid,
        "structural_checks": structural_checks,
        "structural_reasons": structural_reasons,
        "rendered_png": rendered,
        "render_reason": render_reason,
        "png_size_bytes": png_size,
    }
