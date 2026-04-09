"""
Output formatting and filtering for the live IDS CLI.

Two output modes:
    - text  (default): one-line-per-flow human readable, optional ANSI colour
    - jsonl (--json):  one compact JSON object per line, suitable for piping

Filtering uses the BENIGN_LABEL constant for exact-string matching against the
prediction. The label string is the CIC-IDS2017 relabelling produced by
training/preprocess.py — NOT 'BENIGN'.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone


# Exact CIC-IDS2017 relabelling for the benign class. Set in
# training/preprocess.py:LABEL_GROUPS where 'BENIGN' is mapped to
# 'Normal Traffic'. The model's label_mapping pickle uses this string.
BENIGN_LABEL = "Normal Traffic"


# ANSI escape codes (no external dependency).
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_DIM = "\033[2m"
_RESET = "\033[0m"


@dataclass
class FlowSummary:
    """Lightweight 5-tuple + timing snapshot used by the formatter."""
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    proto: str
    duration_s: float
    packets: int
    end_time: float  # epoch seconds


def _iso_utc(epoch_s: float) -> str:
    return (
        datetime.fromtimestamp(epoch_s, tz=timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    )


def _top_n_probs(probabilities: dict[str, float], n: int = 2) -> list[tuple[str, float]]:
    return sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)[:n]


def _color_for(prediction: str, confidence: float, no_color: bool) -> tuple[str, str]:
    """Return (prefix, suffix) ANSI codes for a class label."""
    if no_color:
        return ("", "")
    if prediction == BENIGN_LABEL:
        return (_GREEN, _RESET)
    if confidence < 0.7:
        return (_YELLOW, _RESET)
    return (_RED, _RESET)


def format_text(flow: FlowSummary, prediction: dict, no_color: bool = False) -> str:
    """Render one flow + prediction as a two-line human-readable record."""
    pred_label = prediction.get("prediction", "?")
    confidence = float(prediction.get("confidence", 0.0))
    probs = prediction.get("probabilities", {}) or {}

    pre, post = _color_for(pred_label, confidence, no_color)
    dim_pre = "" if no_color else _DIM
    dim_post = "" if no_color else _RESET

    top = _top_n_probs(probs, 2)
    top_str = ", ".join(f"{lbl}:{p:.3f}" for lbl, p in top)

    return (
        f"{dim_pre}{_iso_utc(flow.end_time)}{dim_post}  "
        f"{flow.src_ip}:{flow.src_port} → {flow.dst_ip}:{flow.dst_port}  "
        f"{flow.proto}  flow={flow.duration_s:.3f}s pkts={flow.packets}\n"
        f"  prediction={pre}{pred_label}{post}  "
        f"confidence={confidence:.3f}  top2=[{top_str}]"
    )


def format_jsonl(flow: FlowSummary, prediction: dict) -> str:
    """Render one flow + prediction as a single compact JSON line."""
    obj = {
        "ts": _iso_utc(flow.end_time),
        "src_ip": flow.src_ip,
        "dst_ip": flow.dst_ip,
        "src_port": flow.src_port,
        "dst_port": flow.dst_port,
        "proto": flow.proto,
        "duration_s": round(flow.duration_s, 6),
        "packets": flow.packets,
        "prediction": prediction.get("prediction"),
        "confidence": prediction.get("confidence"),
        "probabilities": prediction.get("probabilities"),
    }
    return json.dumps(obj, separators=(",", ":"))


def should_emit(prediction: dict, only_non_benign: bool, min_confidence: float) -> bool:
    """Filter rule applied before printing a record."""
    if only_non_benign and prediction.get("prediction") == BENIGN_LABEL:
        return False
    if float(prediction.get("confidence", 0.0)) < min_confidence:
        return False
    return True


def stdout_supports_color() -> bool:
    return sys.stdout.isatty()
