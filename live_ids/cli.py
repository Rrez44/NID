"""
Command-line entry point for the live IDS.

Subcommands:
    capture   Capture live / PCAP traffic → flow features → POST /predict
    validate  Assert that the CLI's FEATURE_NAMES match the trained model's
              feature pickle on disk, and that flow_to_feature_dict() produces
              a clean 40-key dict for a synthetic (or real PCAP) flow
    info      Print capture engine + API /health + /classes

The CLI is a pure client of the API in api/serve.py — it never imports or
loads the XGBoost model itself. Model artifacts are only read (via joblib)
to self-check the feature name list in `validate`.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import signal
import sys
from pathlib import Path

from . import __version__
from . import capture as capture_mod
from . import features as features_mod
from . import output as output_mod

# client module is imported lazily inside subcommands that need it, so that
# `validate` (which doesn't touch the API) works in environments where httpx
# is not installed.


log = logging.getLogger("live_ids")


# Exit codes (documented in the plan)
EXIT_OK = 0
EXIT_API = 2
EXIT_PERM = 3
EXIT_SOURCE = 4
EXIT_VALIDATE = 5
EXIT_SIGINT = 130


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="live_ids",
        description="Live-capture IDS CLI — packets → flows → POST /predict",
    )
    p.add_argument("--version", action="version", version=f"live_ids {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    # -- capture -----------------------------------------------------------
    cap = sub.add_parser("capture", help="Capture + classify network flows")
    src = cap.add_mutually_exclusive_group(required=True)
    src.add_argument("-i", "--interface", help="Network interface for live capture")
    src.add_argument("--pcap", help="Read from a PCAP/PCAPNG file instead of a NIC")

    cap.add_argument("--engine", choices=("auto", "nfstream", "scapy"),
                     default="auto", help="Capture backend selection")

    cap.add_argument("--api-url",
                     default=os.environ.get("IDS_API_URL", "http://127.0.0.1:8000"),
                     help="Base URL of the IDS API (default: $IDS_API_URL or "
                          "http://127.0.0.1:8000)")
    cap.add_argument("--api-timeout", type=float, default=5.0,
                     help="HTTP request timeout in seconds")
    cap.add_argument("--api-retries", type=int, default=2,
                     help="Retry count on 5xx / network errors")
    cap.add_argument("--no-health-check", action="store_true",
                     help="Skip the startup /health probe")

    cap.add_argument("--idle-timeout", type=int, default=15,
                     help="Flow idle timeout in seconds")
    cap.add_argument("--active-timeout", type=int, default=120,
                     help="Flow active (hard-cap) timeout in seconds")
    cap.add_argument("--active-gap", type=float, default=5.0,
                     help="CICFlowMeter Active/Idle split threshold in seconds")
    cap.add_argument("--bpf-filter", default=None,
                     help="BPF capture filter expression")
    cap.add_argument("--snaplen", type=int, default=1600,
                     help="Capture snap length in bytes")
    cap.add_argument("--promiscuous", action="store_true",
                     help="Enable promiscuous mode (live only)")

    cap.add_argument("--min-confidence", type=float, default=0.0,
                     help="Hide predictions below this confidence [0..1]")
    cap.add_argument("--only-non-benign", action="store_true",
                     help=f"Hide flows predicted as '{output_mod.BENIGN_LABEL}'")
    cap.add_argument("--json", dest="json_out", action="store_true",
                     help="Emit jsonl instead of human-readable text")
    cap.add_argument("--no-color", action="store_true",
                     help="Disable ANSI colour output")
    cap.add_argument("--dry-run", action="store_true",
                     help="Compute features but skip the API POST")
    cap.add_argument("--max-flows", type=int, default=None,
                     help="Exit after processing N flows (test convenience)")
    cap.add_argument("--debug", action="store_true",
                     help="Dump feature dict on 422 and enable DEBUG logs")
    cap.add_argument("-v", "--verbose", action="store_true",
                     help="Enable INFO logs")

    # -- validate ----------------------------------------------------------
    val = sub.add_parser("validate", help="Self-check feature mapping vs model")
    val.add_argument("--pcap", default=None,
                     help="Run real capture → features on a small PCAP too")
    val.add_argument("--feature-names", default=None,
                     help="Override path to feature names pickle")
    val.add_argument("--strict", action="store_true",
                     help="Fail if any feature has status 'A' or 'S'")
    val.add_argument("-v", "--verbose", action="store_true", help="Enable INFO logs")

    # -- info --------------------------------------------------------------
    info = sub.add_parser("info", help="Print engine + API health + classes")
    info.add_argument("--api-url",
                      default=os.environ.get("IDS_API_URL", "http://127.0.0.1:8000"),
                      help="Base URL of the IDS API")
    info.add_argument("--api-timeout", type=float, default=5.0)

    return p


# ---------------------------------------------------------------------------
# Subcommand: capture
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool, debug: bool) -> None:
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def _cmd_capture(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose, args.debug)

    from .client import (  # lazy: requires httpx
        APIBadRequest,
        APIModelNotLoaded,
        APIUnexpectedError,
        APIUnreachable,
        IDSClient,
    )

    source = args.interface or args.pcap
    if args.pcap and not Path(args.pcap).is_file():
        print(f"error: pcap file not found: {args.pcap}", file=sys.stderr)
        return EXIT_SOURCE

    opts = capture_mod.CaptureOpts(
        bpf_filter=args.bpf_filter,
        snaplen=args.snaplen,
        promiscuous=args.promiscuous,
        idle_timeout=args.idle_timeout,
        active_timeout=args.active_timeout,
        active_gap_us=int(args.active_gap * 1_000_000),
    )

    no_color = args.no_color or not output_mod.stdout_supports_color()

    # API client + health check
    client: IDSClient | None = None
    if not args.dry_run:
        client = IDSClient(args.api_url, timeout=args.api_timeout, retries=args.api_retries)
        if not args.no_health_check:
            try:
                h = client.health()
                if not h.get("model_loaded"):
                    print(
                        "error: API is up but model not loaded — run `make train` "
                        "or check models/ directory",
                        file=sys.stderr,
                    )
                    return EXIT_API
                try:
                    class_list = client.classes()
                    if output_mod.BENIGN_LABEL not in class_list:
                        print(
                            f"warning: '{output_mod.BENIGN_LABEL}' not in /classes "
                            f"({class_list!r}); --only-non-benign filter may be wrong",
                            file=sys.stderr,
                        )
                except APIModelNotLoaded:
                    print("error: model not loaded", file=sys.stderr)
                    return EXIT_API
            except APIUnreachable as e:
                print(
                    f"error: {e}\nIs the API running? Try `make serve`.",
                    file=sys.stderr,
                )
                return EXIT_API
            except APIUnexpectedError as e:
                print(f"error: unexpected API response: {e}", file=sys.stderr)
                return EXIT_API

    # Graceful SIGINT: the inner generator flushes remaining flows on
    # GeneratorExit (scapy path) or nfstream streaming stop.
    def _sigint(signum, frame):
        print("\nshutting down...", file=sys.stderr)
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, _sigint)

    emitted = 0
    try:
        for record in capture_mod.iter_flows(source, opts, engine=args.engine):
            try:
                feats = features_mod.flow_to_feature_dict(
                    record.flow, active_gap_us=opts.active_gap_us
                )
            except Exception as e:
                log.warning("feature extraction failed for flow: %s", e)
                continue

            # Sanity: all 40 keys finite
            if args.debug:
                missing_local = set(features_mod.FEATURE_NAMES) - set(feats.keys())
                if missing_local:
                    log.debug("local missing keys: %s", sorted(missing_local))

            if args.dry_run:
                if args.json_out:
                    import json as _json
                    print(_json.dumps({"summary": record.summary.__dict__,
                                       "features": feats},
                                      separators=(",", ":"), default=str))
                else:
                    print(f"-- flow {record.summary.src_ip}:{record.summary.src_port}"
                          f" → {record.summary.dst_ip}:{record.summary.dst_port}"
                          f" ({record.summary.proto}) --")
                    for k in features_mod.FEATURE_NAMES:
                        print(f"  {k}: {feats[k]}")
                emitted += 1
                if args.max_flows and emitted >= args.max_flows:
                    break
                continue

            try:
                assert client is not None
                prediction = client.predict(feats)
            except APIUnreachable as e:
                log.warning("API unreachable, skipping flow: %s", e)
                continue
            except APIModelNotLoaded:
                print("error: model unloaded mid-stream", file=sys.stderr)
                return EXIT_API
            except APIBadRequest as e:
                if args.debug:
                    print(f"DEBUG 422: {e}", file=sys.stderr)
                    print("sent features:", file=sys.stderr)
                    for k in features_mod.FEATURE_NAMES:
                        print(f"  {k}: {feats.get(k, '<missing>')}", file=sys.stderr)
                    if e.missing:
                        print(f"server-reported missing: {e.missing}", file=sys.stderr)
                else:
                    log.warning("422 from API: %s", e)
                continue
            except APIUnexpectedError as e:
                log.warning("unexpected API error: %s", e)
                continue

            if not output_mod.should_emit(
                prediction, args.only_non_benign, args.min_confidence
            ):
                continue

            if args.json_out:
                print(output_mod.format_jsonl(record.summary, prediction))
            else:
                print(output_mod.format_text(record.summary, prediction, no_color=no_color))

            emitted += 1
            if args.max_flows and emitted >= args.max_flows:
                break

    except PermissionError as e:
        print(
            f"error: permission denied on capture ({e}). Run with sudo, or: "
            f"sudo setcap cap_net_raw,cap_net_admin=eip $(readlink -f $(which python))",
            file=sys.stderr,
        )
        return EXIT_PERM
    except KeyboardInterrupt:
        return EXIT_SIGINT
    finally:
        if client is not None:
            client.close()

    return EXIT_OK


# ---------------------------------------------------------------------------
# Subcommand: validate
# ---------------------------------------------------------------------------

def _cmd_validate(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose, debug=False)

    # Resolve feature names pickle path via config.py (single source of truth).
    try:
        # Ensure repo root is on sys.path so `config` imports even when the
        # CLI is invoked via `python -m live_ids` from any cwd.
        _repo_root = Path(__file__).resolve().parent.parent
        if str(_repo_root) not in sys.path:
            sys.path.insert(0, str(_repo_root))
        import config  # type: ignore
    except Exception as e:
        print(f"error: cannot import repo config.py: {e}", file=sys.stderr)
        return EXIT_VALIDATE

    pickle_path = Path(args.feature_names or str(config.DEFAULT_FEATURE_NAMES))
    if not pickle_path.exists():
        print(
            f"error: feature names pickle not found at {pickle_path}\n"
            f"Run `make train` to produce model artifacts.",
            file=sys.stderr,
        )
        return EXIT_VALIDATE

    try:
        import joblib  # lazy import
        expected = list(joblib.load(pickle_path))
    except Exception as e:
        print(f"error: failed to load {pickle_path}: {e}", file=sys.stderr)
        return EXIT_VALIDATE

    cli_names = list(features_mod.FEATURE_NAMES)

    # Set equality check
    expected_set = set(expected)
    cli_set = set(cli_names)
    added = sorted(cli_set - expected_set)
    removed = sorted(expected_set - cli_set)
    if added or removed:
        print("FAIL: feature name mismatch between CLI and model artifact")
        if removed:
            print(f"  missing from CLI (in model but not in features.py): {removed}")
        if added:
            print(f"  extra in CLI (not expected by model): {added}")
        return EXIT_VALIDATE

    if tuple(expected) != tuple(cli_names):
        print("warning: feature ORDER differs from model pickle (API tolerates this)")

    # Synthetic feature dict sanity
    sample = features_mod.synthesize_sample_flow()
    if len(sample) != 40:
        print(f"FAIL: synthetic sample has {len(sample)} keys, expected 40")
        return EXIT_VALIDATE

    for k in cli_names:
        if k not in sample:
            print(f"FAIL: synthetic sample missing key {k!r}")
            return EXIT_VALIDATE
        v = sample[k]
        if not isinstance(v, float):
            print(f"FAIL: {k!r} value {v!r} is not a float")
            return EXIT_VALIDATE
        if not math.isfinite(v):
            print(f"FAIL: {k!r} value {v!r} is not finite")
            return EXIT_VALIDATE

    # Count statuses and apply --strict if requested
    counts = {"V": 0, "A": 0, "S": 0}
    for name in cli_names:
        status = features_mod.FEATURE_SOURCE.get(name, (None, "?"))[1]
        counts[status] = counts.get(status, 0) + 1

    if args.strict:
        non_verified = [
            n for n in cli_names
            if features_mod.FEATURE_SOURCE.get(n, (None, "?"))[1] != "V"
        ]
        if non_verified:
            print(
                f"FAIL (--strict): {len(non_verified)} non-verified features: "
                f"{non_verified}"
            )
            return EXIT_VALIDATE

    print(
        f"OK: 40/40 features map to model artifact at {pickle_path}\n"
        f"    verified={counts.get('V', 0)} "
        f"approximate={counts.get('A', 0)} "
        f"synthesized={counts.get('S', 0)}"
    )

    # Optional end-to-end on a real PCAP
    if args.pcap:
        pcap = Path(args.pcap)
        if not pcap.is_file():
            print(f"FAIL: pcap not found: {pcap}", file=sys.stderr)
            return EXIT_VALIDATE
        print(f"Running end-to-end check on {pcap}...")
        opts = capture_mod.CaptureOpts()
        checked = 0
        try:
            for record in capture_mod.iter_flows(str(pcap), opts, engine="auto"):
                feats = features_mod.flow_to_feature_dict(record.flow)
                missing = [k for k in cli_names if k not in feats]
                if missing:
                    print(f"FAIL: flow missing keys: {missing}")
                    return EXIT_VALIDATE
                for k, v in feats.items():
                    if not isinstance(v, float) or not math.isfinite(v):
                        print(f"FAIL: {k!r} non-finite: {v!r}")
                        return EXIT_VALIDATE
                checked += 1
                if checked >= 10:
                    break
        except Exception as e:
            print(f"FAIL: capture pipeline raised: {e}", file=sys.stderr)
            return EXIT_VALIDATE
        print(f"OK: {checked} flows checked, all 40 keys present and finite")

    return EXIT_OK


# ---------------------------------------------------------------------------
# Subcommand: info
# ---------------------------------------------------------------------------

def _cmd_info(args: argparse.Namespace) -> int:
    _setup_logging(verbose=False, debug=False)
    engine = capture_mod.detect_engine()
    print(f"live_ids {__version__}")
    print(f"capture engine: {engine}")
    print(f"api url: {args.api_url}")

    try:
        from .client import (  # lazy: requires httpx
            APIModelNotLoaded,
            APIUnexpectedError,
            APIUnreachable,
            IDSClient,
        )
    except ModuleNotFoundError as e:
        print(f"api: client unavailable ({e}); install httpx")
        return EXIT_OK

    try:
        with IDSClient(args.api_url, timeout=args.api_timeout) as c:
            try:
                h = c.health()
                print(f"api /health: {h}")
            except APIUnreachable as e:
                print(f"api /health: unreachable ({e})")
                return EXIT_OK
            try:
                cls = c.classes()
                print(f"api /classes: {cls}")
            except (APIModelNotLoaded, APIUnexpectedError) as e:
                print(f"api /classes: error ({e})")
    except Exception as e:  # defensive
        print(f"api: {e}")
    return EXIT_OK


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "capture":
        return _cmd_capture(args)
    if args.command == "validate":
        return _cmd_validate(args)
    if args.command == "info":
        return _cmd_info(args)

    parser.print_help()
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
