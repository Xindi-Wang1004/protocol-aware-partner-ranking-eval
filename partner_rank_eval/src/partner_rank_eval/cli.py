"""CLI: metrics | simulate | checklist."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .io import load_records
from .metrics import decompose, metrics_from_ranks
from .report import empty_checklist, fill_checklist
from .simulate import simulate_gallery_nesting, simulate_grid


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (bytes, str)):
        try:
            return _jsonable(obj.item())
        except Exception:
            return obj
    return obj


def _dump(obj, path: Path | None) -> None:
    text = json.dumps(_jsonable(obj), indent=2, allow_nan=False)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    else:
        print(text)


def cmd_metrics(args: argparse.Namespace) -> None:
    if args.K > args.T:
        raise ValueError(f"identity requires K <= T (got K={args.K}, T={args.T})")
    recs = load_records(args.rankings)
    m = metrics_from_ranks(recs, T=args.T, K=args.K)
    extra = {}
    if args.protocol:
        extra["protocol"] = args.protocol
    if args.gallery:
        extra["evaluation_gallery"] = args.gallery
    payload = {"metrics": m, "decomposition": decompose(m)}
    if args.checklist:
        payload["checklist"] = fill_checklist(m, extra=extra)
    _dump(payload, Path(args.out) if args.out else None)


def cmd_simulate(args: argparse.Namespace) -> None:
    grid = simulate_grid(n_queries=args.n, gallery_size=args.gallery, seed=args.seed)
    nest = simulate_gallery_nesting(
        n_queries=args.n, full_gallery=args.gallery, seed=args.seed
    )
    payload = {
        "identity_grid": grid,
        "gallery_nesting": nest,
        "note": (
            "Hit@K = Recall@T × oracle@K holds by construction when K ≤ T "
            "on the same query set. Nesting shows headline Hit@K is "
            "monotonically non-decreasing as the evaluation gallery shrinks "
            "(equivalently, non-increasing as the gallery grows)."
        ),
    }
    _dump(payload, Path(args.out) if args.out else None)


def cmd_checklist(args: argparse.Namespace) -> None:
    if args.metrics:
        m = json.loads(Path(args.metrics).read_text())
        if "metrics" in m:
            m = m["metrics"]
        doc = fill_checklist(m)
    else:
        doc = empty_checklist()
    _dump(doc, Path(args.out) if args.out else None)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="partner-rank-eval",
        description="Protocol-aware partner-ranking evaluation and reporting",
    )
    sub = p.add_subparsers(dest="cmd", required=False)

    m = sub.add_parser("metrics", help="Compute Recall@T / oracle@K / Hit@K and the identity")
    m.add_argument("--rankings", required=True, help="CSV/JSONL of per-query ranks")
    m.add_argument("--T", type=int, default=20)
    m.add_argument("--K", type=int, default=10)
    m.add_argument("--protocol", default=None)
    m.add_argument("--gallery", default=None)
    m.add_argument("--checklist", action="store_true")
    m.add_argument("--out", default=None)
    m.set_defaults(func=cmd_metrics)

    s = sub.add_parser("simulate", help="Identity grid + gallery-nesting monotonicity")
    s.add_argument("--n", type=int, default=2000)
    s.add_argument("--gallery", type=int, default=1000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_simulate)

    c = sub.add_parser("checklist", help="Empty or filled reporting-checklist JSON template")
    c.add_argument("--metrics", default=None, help="metrics JSON from the metrics command")
    c.add_argument("--out", default=None)
    c.set_defaults(func=cmd_checklist)

    p.add_argument("--version", action="version", version=f"partner-rank-eval {__version__}")
    args = p.parse_args(argv)
    if not getattr(args, "cmd", None):
        p.error("a command is required: metrics, simulate, or checklist")
    try:
        args.func(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
