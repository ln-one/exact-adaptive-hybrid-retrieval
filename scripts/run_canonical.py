#!/usr/bin/env python3
"""Canonical ED-WRRF experiment entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from canonical_runner.runner import E1Config, run_e1

FROZEN_SYSTEM_COMMIT = "cf9d988386b9b63f5ba559deb76e0f66b55c0fde"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="experiment", required=True)
    e1 = subparsers.add_parser("e1", help="ordered parity against exhaustive exact channels")
    e1.add_argument("--artifact-root", type=Path, required=True)
    e1.add_argument("--dataset", required=True)
    e1.add_argument("--collection", required=True)
    e1.add_argument("--url", default="http://127.0.0.1:6333")
    e1.add_argument("--output", type=Path, required=True)
    e1.add_argument("--bench-repo", type=Path, default=Path(__file__).resolve().parents[1])
    e1.add_argument("--system-repo", type=Path, required=True)
    e1.add_argument("--system-commit", default=FROZEN_SYSTEM_COMMIT)
    e1.add_argument("--system-artifact", required=True)
    e1.add_argument("--hardware-profile", default="apple-m4-pro-24gb-v1")
    e1.add_argument("--dense-name", default="dense")
    e1.add_argument("--sparse-name", default="sparse")
    e1.add_argument("--limit", type=int, default=20)
    e1.add_argument("--rrf-k", type=int, default=60)
    e1.add_argument("--weights", nargs=2, type=float, default=(1.0, 1.0))
    e1.add_argument("--query-limit", type=int)
    e1.add_argument("--max-live-oracle-points", type=int, default=10_000)
    e1.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.experiment == "e1":
        summary = run_e1(
            E1Config(
                artifact_root=args.artifact_root,
                dataset=args.dataset,
                collection=args.collection,
                url=args.url,
                output=args.output,
                bench_repo=args.bench_repo,
                system_repo=args.system_repo,
                system_commit=args.system_commit,
                system_artifact=args.system_artifact,
                hardware_profile=args.hardware_profile,
                dense_name=args.dense_name,
                sparse_name=args.sparse_name,
                limit=args.limit,
                rrf_k=args.rrf_k,
                weights=tuple(args.weights),
                query_limit=args.query_limit,
                allow_dirty=args.allow_dirty,
                max_live_oracle_points=args.max_live_oracle_points,
            )
        )
        print(json.dumps(summary, sort_keys=True))
        if (
            summary["mismatchQueries"] > 0
            or summary["timeoutQueries"] > 0
            or summary["errorQueries"] > 0
        ):
            raise SystemExit(2)


if __name__ == "__main__":
    main()
