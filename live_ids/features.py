"""
Feature mapping layer for the live IDS CLI.

Transforms a completed bidirectional flow object (from either nfstream or the
scapy fallback aggregator in capture.py) into a ``dict[str, float]`` whose keys
exactly match the 40 feature names the trained XGBoost model expects, as stored
in ``models/xgb_ids_model_feature_names.pkl``.

Status legend for FEATURE_SOURCE:
    V — verified: direct nfstream field whose meaning matches CIC-IDS2017
    A — approximate: derived value where nfstream's definition may differ
    S — synthesized: computed in capture.ActiveIdleEnricher (or scapy
        fallback) from raw per-packet state stored in ``flow.udps.*``

The flow object passed to ``flow_to_feature_dict`` must expose the nfstream
NFlow-style attribute names listed in FEATURE_SOURCE plus a ``udps`` namespace
with the synthesized fields. The scapy fallback in capture.py builds an
adapter that satisfies this contract so the mapping table stays single-sourced.
"""

from __future__ import annotations

import math


# Active/Idle period split threshold (CICFlowMeter default), in microseconds.
# capture.py overrides this per-flow when --active-gap is set.
DEFAULT_ACTIVE_GAP_US = 5_000_000


# The exact 40 feature names the model expects, in model order.
# Loaded from models/xgb_ids_model_feature_names.pkl. Do not reorder.
FEATURE_NAMES: tuple[str, ...] = (
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Length of Fwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd Header Length",
    "Bwd Header Length",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Variance",
    "FIN Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Max",
    "Active Min",
)


# Mapping table: {feature_name: (source_expr, status)}.
# source_expr is informational text shown by `validate`; status is V/A/S.
FEATURE_SOURCE: dict[str, tuple[str, str]] = {
    "Destination Port":             ("flow.dst_port", "V"),
    "Flow Duration":                ("ms_to_us(bidirectional_duration_ms)", "V"),
    "Total Fwd Packets":            ("flow.src2dst_packets", "V"),
    "Total Length of Fwd Packets":  ("flow.src2dst_bytes", "V"),
    "Fwd Packet Length Max":        ("flow.src2dst_max_ps", "V"),
    "Fwd Packet Length Min":        ("flow.src2dst_min_ps", "V"),
    "Fwd Packet Length Mean":       ("flow.src2dst_mean_ps", "V"),
    "Bwd Packet Length Max":        ("flow.dst2src_max_ps", "V"),
    "Bwd Packet Length Min":        ("flow.dst2src_min_ps", "V"),
    "Flow Bytes/s":                 ("bidirectional_bytes / duration_s", "V"),
    "Flow Packets/s":               ("bidirectional_packets / duration_s", "V"),
    "Flow IAT Mean":                ("ms_to_us(bidirectional_mean_piat_ms)", "V"),
    "Flow IAT Std":                 ("ms_to_us(bidirectional_stddev_piat_ms)", "V"),
    "Flow IAT Max":                 ("ms_to_us(bidirectional_max_piat_ms)", "V"),
    "Flow IAT Min":                 ("ms_to_us(bidirectional_min_piat_ms)", "V"),
    "Fwd IAT Mean":                 ("ms_to_us(src2dst_mean_piat_ms)", "V"),
    "Fwd IAT Std":                  ("ms_to_us(src2dst_stddev_piat_ms)", "V"),
    "Fwd IAT Min":                  ("ms_to_us(src2dst_min_piat_ms)", "V"),
    "Bwd IAT Total":                ("ms_to_us(dst2src_last - dst2src_first)", "V"),
    "Bwd IAT Mean":                 ("ms_to_us(dst2src_mean_piat_ms)", "V"),
    "Bwd IAT Std":                  ("ms_to_us(dst2src_stddev_piat_ms)", "V"),
    "Bwd IAT Max":                  ("ms_to_us(dst2src_max_piat_ms)", "V"),
    "Bwd IAT Min":                  ("ms_to_us(dst2src_min_piat_ms)", "V"),
    "Fwd Header Length":            ("udps.fwd_header_bytes", "S"),
    "Bwd Header Length":            ("udps.bwd_header_bytes", "S"),
    "Bwd Packets/s":                ("dst2src_packets / duration_s", "V"),
    "Min Packet Length":            ("min(src2dst_min_ps, dst2src_min_ps)", "A"),
    "Max Packet Length":            ("max(src2dst_max_ps, dst2src_max_ps)", "A"),
    "Packet Length Mean":           ("flow.bidirectional_mean_ps", "V"),
    "Packet Length Variance":       ("var_pop(udps.fwd_pkt_lengths + bwd)", "S"),
    "FIN Flag Count":                ("udps.fin_count", "S"),
    "PSH Flag Count":                ("flow.bidirectional_psh_packets", "V"),
    "ACK Flag Count":                ("flow.bidirectional_ack_packets", "V"),
    "Init_Win_bytes_forward":       ("udps.init_win_fwd (-1 if absent)", "S"),
    "Init_Win_bytes_backward":      ("udps.init_win_bwd (-1 if absent)", "S"),
    "act_data_pkt_fwd":             ("udps.fwd_act_data_pkts", "S"),
    "min_seg_size_forward":         ("udps.fwd_min_tcp_header (0 for UDP)", "S"),
    "Active Mean":                  ("active_stats(udps.timestamps_us)[0]", "S"),
    "Active Max":                   ("active_stats(udps.timestamps_us)[1]", "S"),
    "Active Min":                   ("active_stats(udps.timestamps_us)[2]", "S"),
}


def _finite(x, default: float = 0.0) -> float:
    """Coerce ``x`` to a finite float; fall back to ``default`` on NaN/inf/None."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(v):
        return float(default)
    return v


def _ms_to_us(x) -> float:
    """Convert nfstream's millisecond duration/IAT to CICFlowMeter microseconds."""
    return _finite(x) * 1000.0


def _variance_population(xs) -> float:
    """Population variance of a sequence (matches CICFlowMeter's definition)."""
    n = len(xs)
    if n == 0:
        return 0.0
    mean = sum(xs) / n
    return sum((x - mean) ** 2 for x in xs) / n


def _active_stats(timestamps_us, gap_us: int = DEFAULT_ACTIVE_GAP_US):
    """
    CICFlowMeter Active period statistics.

    Walks the sorted per-packet timestamp list (microseconds) and splits the
    flow on any gap larger than ``gap_us``. Each contiguous "active" run
    contributes a duration. Returns ``(mean, max, min)`` of those durations
    in microseconds. Single-period flows yield ``mean == max == min == flow_dur``.
    Zero/one packet flows yield ``(0, 0, 0)``.
    """
    n = len(timestamps_us)
    if n < 2:
        return (0.0, 0.0, 0.0)
    ts = sorted(timestamps_us)
    durations = []
    active_start = ts[0]
    last = ts[0]
    for t in ts[1:]:
        if t - last > gap_us:
            durations.append(last - active_start)
            active_start = t
        last = t
    durations.append(last - active_start)
    if not durations:
        return (0.0, 0.0, 0.0)
    return (
        sum(durations) / len(durations),
        float(max(durations)),
        float(min(durations)),
    )


def _attr(flow, name: str, default=0):
    """Tolerant getattr for nfstream attribute drift across versions."""
    return getattr(flow, name, default)


def _udps(flow, name: str, default=0):
    """Read a field from ``flow.udps``; tolerate missing attribute / namespace."""
    udps = getattr(flow, "udps", None)
    if udps is None:
        return default
    return getattr(udps, name, default)


def _min_ignoring_zero(*xs) -> float:
    """min() that ignores values that are 0 (e.g. one-direction flows)."""
    pos = [x for x in xs if x and x > 0]
    if not pos:
        return 0.0
    return float(min(pos))


def flow_to_feature_dict(flow, active_gap_us: int = DEFAULT_ACTIVE_GAP_US) -> dict[str, float]:
    """
    Convert a completed bidirectional flow into the 40-key feature dict the
    IDS model expects. ``flow`` may be either an nfstream NFlow with a
    populated ``udps`` namespace (see capture.ActiveIdleEnricher) or the
    fallback adapter built by capture._FallbackFlowBuilder.

    All values are coerced to finite floats. Missing nfstream fields fall back
    to 0.0 via the ``_attr`` helper, so partial flows still produce a complete
    dict (the API server's NaN clamp would do the same, but we prefer to send
    clean values so ``--debug`` can detect logic bugs in this layer).
    """
    duration_us = _ms_to_us(_attr(flow, "bidirectional_duration_ms"))
    duration_s = duration_us / 1_000_000.0

    bidir_packets = _attr(flow, "bidirectional_packets", 0)
    bidir_bytes = _attr(flow, "bidirectional_bytes", 0)
    src2dst_packets = _attr(flow, "src2dst_packets", 0)
    dst2src_packets = _attr(flow, "dst2src_packets", 0)

    # Rate features: clamp to 0 for single-packet (zero duration) flows.
    if bidir_packets < 2 or duration_s <= 0:
        flow_bytes_s = 0.0
        flow_pkts_s = 0.0
        bwd_pkts_s = 0.0
    else:
        flow_bytes_s = _finite(bidir_bytes / duration_s)
        flow_pkts_s = _finite(bidir_packets / duration_s)
        bwd_pkts_s = _finite(dst2src_packets / duration_s)

    # Bwd IAT total: span from first to last backward packet, in microseconds.
    bwd_first_ms = _attr(flow, "dst2src_first_seen_ms", 0)
    bwd_last_ms = _attr(flow, "dst2src_last_seen_ms", 0)
    bwd_iat_total = _ms_to_us(bwd_last_ms - bwd_first_ms) if bwd_last_ms and bwd_first_ms else 0.0

    # Synthesized fields populated by ActiveIdleEnricher (nfstream path) or by
    # the scapy fallback adapter.
    fwd_pkt_lengths = _udps(flow, "fwd_pkt_lengths", []) or []
    bwd_pkt_lengths = _udps(flow, "bwd_pkt_lengths", []) or []
    timestamps_us = _udps(flow, "timestamps_us", []) or []

    # CICFlowMeter Packet Length features are computed from TCP payload sizes,
    # not wire packet sizes. The enricher (and scapy fallback) store payload
    # bytes in fwd_pkt_lengths/bwd_pkt_lengths so we derive directly here.
    all_lengths = list(fwd_pkt_lengths) + list(bwd_pkt_lengths)
    fwd_len_max = max(fwd_pkt_lengths) if fwd_pkt_lengths else 0.0
    fwd_len_min = min(fwd_pkt_lengths) if fwd_pkt_lengths else 0.0
    fwd_len_mean = (sum(fwd_pkt_lengths) / len(fwd_pkt_lengths)) if fwd_pkt_lengths else 0.0
    fwd_len_total = sum(fwd_pkt_lengths)
    bwd_len_max = max(bwd_pkt_lengths) if bwd_pkt_lengths else 0.0
    bwd_len_min = min(bwd_pkt_lengths) if bwd_pkt_lengths else 0.0
    pkt_len_min = min(all_lengths) if all_lengths else 0.0
    pkt_len_max = max(all_lengths) if all_lengths else 0.0
    pkt_len_mean = (sum(all_lengths) / len(all_lengths)) if all_lengths else 0.0
    pkt_length_variance = _variance_population(all_lengths)
    active_mean, active_max, active_min = _active_stats(timestamps_us, active_gap_us)

    # Init_Win: prefer nfstream NFlow's native fields, fall back to udps.
    init_win_fwd = _attr(flow, "src2dst_init_window_size", None)
    if init_win_fwd is None or init_win_fwd == 0:
        init_win_fwd = _udps(flow, "init_win_fwd", -1)
    init_win_bwd = _attr(flow, "dst2src_init_window_size", None)
    if init_win_bwd is None or init_win_bwd == 0:
        init_win_bwd = _udps(flow, "init_win_bwd", -1)

    # Some nfstream releases expose bidirectional_fin_packets directly; if not,
    # fall back to the value the NFPlugin counted.
    fin_count = _attr(flow, "bidirectional_fin_packets", None)
    if fin_count is None:
        fin_count = _udps(flow, "fin_count", 0)

    out: dict[str, float] = {
        "Destination Port":             _finite(_attr(flow, "dst_port")),
        "Flow Duration":                duration_us,
        "Total Fwd Packets":            _finite(src2dst_packets),
        "Total Length of Fwd Packets":  _finite(fwd_len_total),
        "Fwd Packet Length Max":        _finite(fwd_len_max),
        "Fwd Packet Length Min":        _finite(fwd_len_min),
        "Fwd Packet Length Mean":       _finite(fwd_len_mean),
        "Bwd Packet Length Max":        _finite(bwd_len_max),
        "Bwd Packet Length Min":        _finite(bwd_len_min),
        "Flow Bytes/s":                 flow_bytes_s,
        "Flow Packets/s":               flow_pkts_s,
        "Flow IAT Mean":                _ms_to_us(_attr(flow, "bidirectional_mean_piat_ms")),
        "Flow IAT Std":                 _ms_to_us(_attr(flow, "bidirectional_stddev_piat_ms")),
        "Flow IAT Max":                 _ms_to_us(_attr(flow, "bidirectional_max_piat_ms")),
        "Flow IAT Min":                 _ms_to_us(_attr(flow, "bidirectional_min_piat_ms")),
        "Fwd IAT Mean":                 _ms_to_us(_attr(flow, "src2dst_mean_piat_ms")),
        "Fwd IAT Std":                  _ms_to_us(_attr(flow, "src2dst_stddev_piat_ms")),
        "Fwd IAT Min":                  _ms_to_us(_attr(flow, "src2dst_min_piat_ms")),
        "Bwd IAT Total":                _finite(bwd_iat_total),
        "Bwd IAT Mean":                 _ms_to_us(_attr(flow, "dst2src_mean_piat_ms")),
        "Bwd IAT Std":                  _ms_to_us(_attr(flow, "dst2src_stddev_piat_ms")),
        "Bwd IAT Max":                  _ms_to_us(_attr(flow, "dst2src_max_piat_ms")),
        "Bwd IAT Min":                  _ms_to_us(_attr(flow, "dst2src_min_piat_ms")),
        "Fwd Header Length":            _finite(_udps(flow, "fwd_header_bytes", 0)),
        "Bwd Header Length":            _finite(_udps(flow, "bwd_header_bytes", 0)),
        "Bwd Packets/s":                bwd_pkts_s,
        "Min Packet Length":            _finite(pkt_len_min),
        "Max Packet Length":            _finite(pkt_len_max),
        "Packet Length Mean":           _finite(pkt_len_mean),
        "Packet Length Variance":       _finite(pkt_length_variance),
        "FIN Flag Count":                _finite(fin_count),
        "PSH Flag Count":                _finite(_attr(flow, "bidirectional_psh_packets")),
        "ACK Flag Count":                _finite(_attr(flow, "bidirectional_ack_packets")),
        # Init_Win_bytes_*: -1 sentinel preserved (CICFlowMeter convention).
        "Init_Win_bytes_forward":       float(init_win_fwd),
        "Init_Win_bytes_backward":      float(init_win_bwd),
        "act_data_pkt_fwd":             _finite(_udps(flow, "fwd_act_data_pkts", 0)),
        "min_seg_size_forward":         _finite(_udps(flow, "fwd_min_tcp_header", 0)),
        "Active Mean":                  _finite(active_mean),
        "Active Max":                   _finite(active_max),
        "Active Min":                   _finite(active_min),
    }

    return out


def synthesize_sample_flow() -> dict[str, float]:
    """
    Build a deterministic sample feature dict used by ``validate`` (offline
    mode). Values are not realistic — they only need to satisfy "40 finite
    floats with the right keys" so the mapping layer can be smoke-tested
    without a packet source.
    """

    class _DummyUDPS:
        timestamps_us = [0, 1_000_000, 2_000_000, 3_000_000, 4_000_000]
        fwd_pkt_lengths = [60, 120, 180]
        bwd_pkt_lengths = [80, 100]
        fwd_header_bytes = 60
        bwd_header_bytes = 40
        init_win_fwd = 65535
        init_win_bwd = 65535
        fwd_act_data_pkts = 2
        fwd_min_tcp_header = 20
        fin_count = 1

    class _DummyFlow:
        dst_port = 443
        bidirectional_duration_ms = 4.0
        bidirectional_packets = 5
        bidirectional_bytes = 540
        src2dst_packets = 3
        dst2src_packets = 2
        src2dst_bytes = 360
        src2dst_max_ps = 180
        src2dst_min_ps = 60
        src2dst_mean_ps = 120.0
        dst2src_max_ps = 100
        dst2src_min_ps = 80
        bidirectional_mean_piat_ms = 1.0
        bidirectional_stddev_piat_ms = 0.0
        bidirectional_max_piat_ms = 1.0
        bidirectional_min_piat_ms = 1.0
        src2dst_mean_piat_ms = 1.0
        src2dst_stddev_piat_ms = 0.0
        src2dst_min_piat_ms = 1.0
        dst2src_first_seen_ms = 0.0
        dst2src_last_seen_ms = 4.0
        dst2src_mean_piat_ms = 2.0
        dst2src_stddev_piat_ms = 0.0
        dst2src_max_piat_ms = 2.0
        dst2src_min_piat_ms = 2.0
        bidirectional_mean_ps = 108.0
        bidirectional_psh_packets = 1
        bidirectional_ack_packets = 4
        bidirectional_fin_packets = 1
        udps = _DummyUDPS()

    return flow_to_feature_dict(_DummyFlow())
