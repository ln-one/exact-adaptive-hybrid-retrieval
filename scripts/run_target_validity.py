#!/usr/bin/env python3
"""CLI for the recoverable target-validity retrieval campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from canonical_runner.target_validity import (
    DEFAULT_DEPTHS,
    TargetValidityConfig,
    run_target_validity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--system-repo", type=Path, required=True)
    parser.add_argument("--system-commit", required=True)
    parser.add_argument("--system-artifact", required=True)
    parser.add_argument("--system-binary", type=Path)
    parser.add_argument("--system-build-manifest", type=Path)
    parser.add_argument("--hardware-profile", default="apple-m4-pro-24gb-v1")
    parser.add_argument("--url", default="http://127.0.0.1:6333")
    parser.add_argument("--dense-name", default="dense")
    parser.add_argument("--sparse-name", default="sparse")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--weights", type=float, nargs=2, default=(1.0, 1.0))
    parser.add_argument("--depths", type=int, nargs="+", default=DEFAULT_DEPTHS)
    parser.add_argument("--request-timeout-seconds", type=float, default=3_600.0)
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bench_repo = Path(__file__).resolve().parents[1]
    summary = run_target_validity(
        TargetValidityConfig(
            artifact_root=args.artifact_root,
            dataset=args.dataset,
            collection=args.collection,
            output=args.output,
            bench_repo=bench_repo,
            system_repo=args.system_repo,
            system_commit=args.system_commit,
            system_artifact=args.system_artifact,
            hardware_profile=args.hardware_profile,
            system_binary=args.system_binary,
            system_build_manifest=args.system_build_manifest,
            url=args.url,
            dense_name=args.dense_name,
            sparse_name=args.sparse_name,
            limit=args.limit,
            certificate_limit=args.limit + 1,
            rrf_k=args.rrf_k,
            weights=tuple(args.weights),
            depths=tuple(args.depths),
            request_timeout_seconds=args.request_timeout_seconds,
            query_limit=args.query_limit,
            allow_dirty=args.allow_dirty,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["allQueriesComplete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
