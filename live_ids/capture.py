"""
Packet capture and flow assembly.

Two backends, one public iterator interface:

    iter_flows(source, opts, engine='auto') -> Iterator[FlowRecord]

* nfstream (preferred): bidirectional flow assembly + most CICFlowMeter
  statistics computed in libpcap C code. We attach an ``ActiveIdleEnricher``
  NFPlugin to fill in the synthesized features (Active/Idle, Init_Win_bytes,
  per-direction header bytes, act_data_pkt_fwd, min_seg_size_forward, FIN
  count). Marked **S** in features.py:FEATURE_SOURCE.

* scapy fallback: triggered if ``import nfstream`` raises (e.g. numpy ABI
  break) or if the user passes ``--engine scapy``. Hand-rolled flow table
  keyed on the canonical bidirectional 5-tuple. Builds a small adapter
  object whose attribute names match nfstream's NFlow contract so the same
  features.flow_to_feature_dict() consumes both backends — single source
  of truth for the feature mapping.

Active/Idle accuracy degrades on flows with more than MAX_TS_PER_FLOW packets
because we cap the per-flow timestamp list to keep memory bounded. Documented
limit; bump if your traffic regularly exceeds it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .output import FlowSummary


log = logging.getLogger(__name__)


# Per-flow timestamp list cap. Beyond this, Active Mean/Max/Min become
# approximate (we still see total duration but lose fine-grained gap analysis).
MAX_TS_PER_FLOW = 8192


@dataclass
class CaptureOpts:
    bpf_filter: str | None = None
    snaplen: int = 1600
    promiscuous: bool = False
    idle_timeout: int = 15        # seconds
    active_timeout: int = 120     # seconds (hard cap)
    active_gap_us: int = 5_000_000  # microseconds (CICFlowMeter Active/Idle split)


@dataclass
class FlowRecord:
    """A completed flow yielded to cli.py for feature extraction + prediction."""
    flow: Any              # nfstream NFlow or _FallbackFlow
    summary: FlowSummary
    engine: str            # 'nfstream' or 'scapy'


# ---------------------------------------------------------------------------
# nfstream backend
# ---------------------------------------------------------------------------

def _try_import_nfstream():
    try:
        import nfstream  # type: ignore
        return nfstream
    except Exception as e:  # pragma: no cover - import-time fallback
        log.info("nfstream import failed (%s); will use scapy fallback", e)
        return None


def _build_active_idle_enricher(nfstream_mod):
    """
    Define ActiveIdleEnricher subclassing nfstream.NFPlugin at runtime so the
    module imports cleanly even when nfstream is absent.
    """
    NFPlugin = nfstream_mod.NFPlugin

    class ActiveIdleEnricher(NFPlugin):
        """
        Per-packet hook stashing state needed for the synthesized features.
        Runs inside nfstream's libpcap loop — no double-parse.
        """

        def on_init(self, packet, flow):
            flow.udps.timestamps_us = []
            flow.udps.fwd_pkt_lengths = []
            flow.udps.bwd_pkt_lengths = []
            flow.udps.fwd_header_bytes = 0
            flow.udps.bwd_header_bytes = 0
            flow.udps.fwd_min_tcp_header = 0   # 0 sentinel = no TCP fwd pkt seen
            flow.udps.fwd_act_data_pkts = 0
            flow.udps.init_win_fwd = -1        # CICFlowMeter convention
            flow.udps.init_win_bwd = -1
            flow.udps.fin_count = 0
            self._account(packet, flow)

        def on_update(self, packet, flow):
            self._account(packet, flow)

        @staticmethod
        def _account(packet, flow):
            # Timestamp (nfstream uses ms since epoch). Cap the list to bound
            # memory; flows beyond the cap get approximate Active/Idle stats.
            ts_ms = getattr(packet, "time", None)
            if ts_ms is not None and len(flow.udps.timestamps_us) < MAX_TS_PER_FLOW:
                flow.udps.timestamps_us.append(int(ts_ms) * 1000)

            ip_size = int(getattr(packet, "ip_size", 0) or 0)
            payload_size = int(getattr(packet, "payload_size", 0) or 0)
            header_bytes = max(ip_size - payload_size, 0)
            direction = int(getattr(packet, "direction", 0) or 0)
            protocol = int(getattr(packet, "protocol", 0) or 0)

            # tcp_window: nfstream may or may not expose this depending on
            # version. -1 sentinel preserved if absent.
            tcp_window = getattr(packet, "tcp_window", None)
            if tcp_window is None:
                tcp_window = getattr(packet, "window", None)

            is_tcp = protocol == 6  # IPPROTO_TCP
            # IPv4 baseline: assume 20-byte IP header (no options).
            tcp_header_len = max(header_bytes - 20, 0) if is_tcp else 0

            if direction == 0:  # forward (src2dst)
                if len(flow.udps.fwd_pkt_lengths) < MAX_TS_PER_FLOW:
                    flow.udps.fwd_pkt_lengths.append(payload_size)
                flow.udps.fwd_header_bytes += header_bytes
                if is_tcp:
                    if payload_size > 0:
                        flow.udps.fwd_act_data_pkts += 1
                    if (
                        flow.udps.fwd_min_tcp_header == 0
                        or tcp_header_len < flow.udps.fwd_min_tcp_header
                    ) and tcp_header_len > 0:
                        flow.udps.fwd_min_tcp_header = tcp_header_len
                    if (
                        flow.udps.init_win_fwd == -1
                        and tcp_window is not None
                        and getattr(packet, "syn", 0)
                    ):
                        flow.udps.init_win_fwd = int(tcp_window)
            else:  # backward (dst2src)
                if len(flow.udps.bwd_pkt_lengths) < MAX_TS_PER_FLOW:
                    flow.udps.bwd_pkt_lengths.append(payload_size)
                flow.udps.bwd_header_bytes += header_bytes
                if is_tcp and (
                    flow.udps.init_win_bwd == -1
                    and tcp_window is not None
                    and getattr(packet, "syn", 0)
                ):
                    flow.udps.init_win_bwd = int(tcp_window)

            if getattr(packet, "fin", 0):
                flow.udps.fin_count += 1

    return ActiveIdleEnricher


def _proto_name(proto: int) -> str:
    return {6: "TCP", 17: "UDP", 1: "ICMP"}.get(int(proto or 0), str(proto))


def _nfstream_summary(flow) -> FlowSummary:
    duration_ms = float(getattr(flow, "bidirectional_duration_ms", 0) or 0)
    end_ms = float(getattr(flow, "bidirectional_last_seen_ms", 0) or 0)
    return FlowSummary(
        src_ip=str(getattr(flow, "src_ip", "?")),
        dst_ip=str(getattr(flow, "dst_ip", "?")),
        src_port=int(getattr(flow, "src_port", 0) or 0),
        dst_port=int(getattr(flow, "dst_port", 0) or 0),
        proto=_proto_name(getattr(flow, "protocol", 0)),
        duration_s=duration_ms / 1000.0,
        packets=int(getattr(flow, "bidirectional_packets", 0) or 0),
        end_time=(end_ms / 1000.0) if end_ms else time.time(),
    )


def _iter_nfstream(source, opts: CaptureOpts) -> Iterator[FlowRecord]:
    nfstream_mod = _try_import_nfstream()
    if nfstream_mod is None:
        raise RuntimeError("nfstream is not importable")

    NFStreamer = nfstream_mod.NFStreamer
    enricher_cls = _build_active_idle_enricher(nfstream_mod)

    streamer_kwargs = dict(
        source=source,
        decode_tunnels=False,
        statistical_analysis=True,
        n_dissections=0,
        idle_timeout=opts.idle_timeout,
        active_timeout=opts.active_timeout,
        snapshot_length=opts.snaplen,
        accounting_mode=0,
        udps=enricher_cls(),
    )
    if opts.bpf_filter:
        streamer_kwargs["bpf_filter"] = opts.bpf_filter
    if opts.promiscuous:
        streamer_kwargs["promiscuous_mode"] = True

    streamer = NFStreamer(**streamer_kwargs)
    for flow in streamer:
        yield FlowRecord(flow=flow, summary=_nfstream_summary(flow), engine="nfstream")


# ---------------------------------------------------------------------------
# scapy fallback backend
# ---------------------------------------------------------------------------

def _try_import_scapy():
    try:
        from scapy.all import (  # type: ignore
            IP, TCP, UDP, AsyncSniffer, PcapReader,
        )
        return {"IP": IP, "TCP": TCP, "UDP": UDP,
                "AsyncSniffer": AsyncSniffer, "PcapReader": PcapReader}
    except Exception as e:  # pragma: no cover
        log.error("scapy import failed: %s", e)
        return None


@dataclass
class _FallbackUDPS:
    timestamps_us: list = field(default_factory=list)
    fwd_pkt_lengths: list = field(default_factory=list)
    bwd_pkt_lengths: list = field(default_factory=list)
    fwd_header_bytes: int = 0
    bwd_header_bytes: int = 0
    fwd_min_tcp_header: int = 0
    fwd_act_data_pkts: int = 0
    init_win_fwd: int = -1
    init_win_bwd: int = -1
    fin_count: int = 0


@dataclass
class _FallbackFlow:
    """
    Adapter that exposes nfstream NFlow attribute names so features.py can
    consume it without branching on backend. Stores raw per-direction
    statistics during accumulation; finalize() computes the aggregates.
    """
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int

    # nfstream-shaped fields populated by finalize()
    bidirectional_duration_ms: float = 0.0
    bidirectional_packets: int = 0
    bidirectional_bytes: int = 0
    bidirectional_first_seen_ms: float = 0.0
    bidirectional_last_seen_ms: float = 0.0
    bidirectional_mean_ps: float = 0.0
    bidirectional_psh_packets: int = 0
    bidirectional_ack_packets: int = 0
    bidirectional_fin_packets: int = 0
    bidirectional_mean_piat_ms: float = 0.0
    bidirectional_stddev_piat_ms: float = 0.0
    bidirectional_max_piat_ms: float = 0.0
    bidirectional_min_piat_ms: float = 0.0

    src2dst_packets: int = 0
    src2dst_bytes: int = 0
    src2dst_max_ps: int = 0
    src2dst_min_ps: int = 0
    src2dst_mean_ps: float = 0.0
    src2dst_mean_piat_ms: float = 0.0
    src2dst_stddev_piat_ms: float = 0.0
    src2dst_min_piat_ms: float = 0.0

    dst2src_packets: int = 0
    dst2src_bytes: int = 0
    dst2src_max_ps: int = 0
    dst2src_min_ps: int = 0
    dst2src_first_seen_ms: float = 0.0
    dst2src_last_seen_ms: float = 0.0
    dst2src_mean_piat_ms: float = 0.0
    dst2src_stddev_piat_ms: float = 0.0
    dst2src_max_piat_ms: float = 0.0
    dst2src_min_piat_ms: float = 0.0

    udps: _FallbackUDPS = field(default_factory=_FallbackUDPS)

    # Internal accumulators (not on nfstream contract)
    _all_ts_ms: list = field(default_factory=list)
    _fwd_ts_ms: list = field(default_factory=list)
    _bwd_ts_ms: list = field(default_factory=list)
    _fwd_sizes: list = field(default_factory=list)
    _bwd_sizes: list = field(default_factory=list)
    _last_seen: float = 0.0
    _first_seen: float = 0.0


def _stats(xs: list[float]) -> tuple[float, float, float, float]:
    """Return (mean, stddev_pop, max, min) of a sequence; (0,0,0,0) if empty."""
    n = len(xs)
    if n == 0:
        return (0.0, 0.0, 0.0, 0.0)
    if n == 1:
        return (float(xs[0]), 0.0, float(xs[0]), float(xs[0]))
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    return (mean, var ** 0.5, float(max(xs)), float(min(xs)))


def _piats(ts_ms: list[float]) -> list[float]:
    """Inter-arrival times (ms) from a sorted timestamp list."""
    if len(ts_ms) < 2:
        return []
    return [ts_ms[i] - ts_ms[i - 1] for i in range(1, len(ts_ms))]


def _finalize_fallback_flow(f: _FallbackFlow) -> None:
    """Compute aggregate stats from accumulators after the flow is closed."""
    f._all_ts_ms.sort()
    f._fwd_ts_ms.sort()
    f._bwd_ts_ms.sort()

    f.bidirectional_packets = len(f._all_ts_ms)
    f.bidirectional_first_seen_ms = f._all_ts_ms[0] if f._all_ts_ms else 0.0
    f.bidirectional_last_seen_ms = f._all_ts_ms[-1] if f._all_ts_ms else 0.0
    f.bidirectional_duration_ms = (
        f.bidirectional_last_seen_ms - f.bidirectional_first_seen_ms
    )
    f.bidirectional_bytes = sum(f._fwd_sizes) + sum(f._bwd_sizes)

    all_sizes = f._fwd_sizes + f._bwd_sizes
    f.bidirectional_mean_ps = (sum(all_sizes) / len(all_sizes)) if all_sizes else 0.0

    fwd_mean, _, fwd_max, fwd_min = _stats([float(x) for x in f._fwd_sizes])
    f.src2dst_packets = len(f._fwd_sizes)
    f.src2dst_bytes = sum(f._fwd_sizes)
    f.src2dst_max_ps = int(fwd_max)
    f.src2dst_min_ps = int(fwd_min) if f._fwd_sizes else 0
    f.src2dst_mean_ps = fwd_mean

    bwd_mean, _, bwd_max, bwd_min = _stats([float(x) for x in f._bwd_sizes])
    f.dst2src_packets = len(f._bwd_sizes)
    f.dst2src_bytes = sum(f._bwd_sizes)
    f.dst2src_max_ps = int(bwd_max)
    f.dst2src_min_ps = int(bwd_min) if f._bwd_sizes else 0

    if f._bwd_ts_ms:
        f.dst2src_first_seen_ms = f._bwd_ts_ms[0]
        f.dst2src_last_seen_ms = f._bwd_ts_ms[-1]

    bidir_piat = _piats(f._all_ts_ms)
    fwd_piat = _piats(f._fwd_ts_ms)
    bwd_piat = _piats(f._bwd_ts_ms)

    bp_mean, bp_std, bp_max, bp_min = _stats(bidir_piat)
    f.bidirectional_mean_piat_ms = bp_mean
    f.bidirectional_stddev_piat_ms = bp_std
    f.bidirectional_max_piat_ms = bp_max
    f.bidirectional_min_piat_ms = bp_min

    fp_mean, fp_std, fp_max, fp_min = _stats(fwd_piat)
    f.src2dst_mean_piat_ms = fp_mean
    f.src2dst_stddev_piat_ms = fp_std
    f.src2dst_min_piat_ms = fp_min

    dp_mean, dp_std, dp_max, dp_min = _stats(bwd_piat)
    f.dst2src_mean_piat_ms = dp_mean
    f.dst2src_stddev_piat_ms = dp_std
    f.dst2src_max_piat_ms = dp_max
    f.dst2src_min_piat_ms = dp_min

    # Populate udps fields used by features.py
    f.udps.timestamps_us = [int(t * 1000) for t in f._all_ts_ms]
    f.udps.fwd_pkt_lengths = list(f._fwd_sizes)
    f.udps.bwd_pkt_lengths = list(f._bwd_sizes)


def _flow_key(src_ip, dst_ip, src_port, dst_port, proto):
    """Canonical bidirectional key — same flow regardless of direction."""
    a = (src_ip, src_port)
    b = (dst_ip, dst_port)
    if a < b:
        return (a, b, proto)
    return (b, a, proto)


def _iter_scapy(source, opts: CaptureOpts) -> Iterator[FlowRecord]:
    mods = _try_import_scapy()
    if mods is None:
        raise RuntimeError("scapy is not importable")

    IP = mods["IP"]
    TCP = mods["TCP"]
    UDP = mods["UDP"]
    PcapReader = mods["PcapReader"]
    AsyncSniffer = mods["AsyncSniffer"]

    flows: dict[tuple, _FallbackFlow] = {}
    flow_initiators: dict[tuple, tuple] = {}  # key -> (src_ip, src_port) of first packet

    def process_packet(pkt) -> None:
        if IP not in pkt:
            return
        ip = pkt[IP]
        proto = int(ip.proto)

        if TCP in pkt:
            l4 = pkt[TCP]
            sport, dport = int(l4.sport), int(l4.dport)
            tcp_header = int(l4.dataofs) * 4 if l4.dataofs else 20
            payload_len = max(int(ip.len) - int(ip.ihl) * 4 - tcp_header, 0)
            tcp_window = int(l4.window) if l4.window is not None else None
            flags = int(l4.flags) if l4.flags is not None else 0
            is_syn = bool(flags & 0x02)
            is_fin = bool(flags & 0x01)
            is_psh = bool(flags & 0x08)
            is_ack = bool(flags & 0x10)
            is_rst = bool(flags & 0x04)
        elif UDP in pkt:
            l4 = pkt[UDP]
            sport, dport = int(l4.sport), int(l4.dport)
            tcp_header = 0
            payload_len = max(int(l4.len) - 8, 0)
            tcp_window = None
            is_syn = is_fin = is_psh = is_ack = is_rst = False
        else:
            return

        key = _flow_key(ip.src, ip.dst, sport, dport, proto)

        # Direction normalization: the first packet defines the initiator.
        if key not in flows:
            flow_initiators[key] = (ip.src, sport)
            flows[key] = _FallbackFlow(
                src_ip=ip.src, dst_ip=ip.dst,
                src_port=sport, dst_port=dport,
                protocol=proto,
            )

        f = flows[key]
        initiator = flow_initiators[key]
        is_forward = (ip.src, sport) == initiator

        ip_size = int(ip.len)
        ip_header = int(ip.ihl) * 4
        header_bytes = ip_header + tcp_header
        ts_ms = float(pkt.time) * 1000.0

        if not f._first_seen:
            f._first_seen = ts_ms
        f._last_seen = ts_ms
        f._all_ts_ms.append(ts_ms)

        if is_forward:
            f._fwd_ts_ms.append(ts_ms)
            f._fwd_sizes.append(payload_len)
            f.udps.fwd_header_bytes += header_bytes
            if proto == 6:  # TCP
                if payload_len > 0:
                    f.udps.fwd_act_data_pkts += 1
                if (
                    f.udps.fwd_min_tcp_header == 0
                    or tcp_header < f.udps.fwd_min_tcp_header
                ) and tcp_header > 0:
                    f.udps.fwd_min_tcp_header = tcp_header
                if f.udps.init_win_fwd == -1 and is_syn and tcp_window is not None:
                    f.udps.init_win_fwd = tcp_window
        else:
            f._bwd_ts_ms.append(ts_ms)
            f._bwd_sizes.append(payload_len)
            f.udps.bwd_header_bytes += header_bytes
            if proto == 6 and f.udps.init_win_bwd == -1 and is_syn and tcp_window is not None:
                f.udps.init_win_bwd = tcp_window

        if is_psh:
            f.bidirectional_psh_packets += 1
        if is_ack:
            f.bidirectional_ack_packets += 1
        if is_fin:
            f.udps.fin_count += 1
            f.bidirectional_fin_packets += 1
        # Mark as ready-to-close on RST or FIN handshake completion (best
        # effort — for live streams the idle timeout is the real backstop).
        f._closed_on_reset = bool(is_rst) or getattr(f, "_closed_on_reset", False)

    def expire_idle(now_ms: float) -> list[tuple]:
        idle_us_threshold_ms = opts.idle_timeout * 1000
        active_us_threshold_ms = opts.active_timeout * 1000
        to_close: list[tuple] = []
        for k, f in flows.items():
            if not f._last_seen:
                continue
            if (now_ms - f._last_seen) >= idle_us_threshold_ms:
                to_close.append(k)
            elif (f._last_seen - f._first_seen) >= active_us_threshold_ms:
                to_close.append(k)
            elif getattr(f, "_closed_on_reset", False):
                to_close.append(k)
        return to_close

    def emit(key: tuple) -> FlowRecord:
        f = flows.pop(key)
        flow_initiators.pop(key, None)
        _finalize_fallback_flow(f)
        return FlowRecord(
            flow=f,
            summary=FlowSummary(
                src_ip=f.src_ip,
                dst_ip=f.dst_ip,
                src_port=f.src_port,
                dst_port=f.dst_port,
                proto=_proto_name(f.protocol),
                duration_s=f.bidirectional_duration_ms / 1000.0,
                packets=f.bidirectional_packets,
                end_time=(f.bidirectional_last_seen_ms or time.time() * 1000) / 1000.0,
            ),
            engine="scapy",
        )

    # Source dispatch: PCAP file vs live interface
    is_pcap = isinstance(source, (str, Path)) and Path(str(source)).is_file()

    if is_pcap:
        with PcapReader(str(source)) as reader:
            for pkt in reader:
                process_packet(pkt)
                # Periodically check for idle expiry to keep memory bounded.
                if pkt.time and len(flows) > 0:
                    now_ms = float(pkt.time) * 1000.0
                    for k in expire_idle(now_ms):
                        yield emit(k)
        # Flush remaining flows at EOF.
        for k in list(flows.keys()):
            yield emit(k)
        return

    # Live capture: AsyncSniffer feeds packets via prn.
    pending: list[FlowRecord] = []

    def _on_packet(pkt):
        process_packet(pkt)

    sniffer_kwargs = dict(iface=source, store=False, prn=_on_packet)
    if opts.bpf_filter:
        sniffer_kwargs["filter"] = opts.bpf_filter
    if opts.promiscuous:
        sniffer_kwargs["promisc"] = True

    sniffer = AsyncSniffer(**sniffer_kwargs)
    sniffer.start()
    try:
        while True:
            time.sleep(0.5)
            now_ms = time.time() * 1000.0
            for k in expire_idle(now_ms):
                pending.append(emit(k))
            while pending:
                yield pending.pop(0)
    except (KeyboardInterrupt, GeneratorExit):
        sniffer.stop()
        # Flush all remaining flows on shutdown.
        for k in list(flows.keys()):
            yield emit(k)


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def detect_engine() -> str:
    """Return 'nfstream' if importable, else 'scapy'."""
    if _try_import_nfstream() is not None:
        return "nfstream"
    if _try_import_scapy() is not None:
        return "scapy"
    return "none"


def iter_flows(source, opts: CaptureOpts, engine: str = "auto") -> Iterator[FlowRecord]:
    """
    Yield FlowRecord objects from either a NIC name or a PCAP file.

    Engine selection:
        'auto' (default) — nfstream if importable, else scapy
        'nfstream' — force nfstream (raises if absent)
        'scapy' — force the pure-Python fallback
    """
    if engine == "auto":
        engine = detect_engine()
        if engine == "none":
            raise RuntimeError(
                "Neither nfstream nor scapy is importable. "
                "Run `pip install -r requirements.txt`."
            )

    if engine == "nfstream":
        yield from _iter_nfstream(source, opts)
    elif engine == "scapy":
        yield from _iter_scapy(source, opts)
    else:
        raise ValueError(f"Unknown engine: {engine!r}")
