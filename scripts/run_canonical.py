#!/usr/bin/env python3
"""Canonical ED-WRRF experiment entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from canonical_runner.e2 import E2Config, run_e2
from canonical_runner.e3 import DEFAULT_DEPTHS, E3Config, run_e3
from canonical_runner.e4 import DEFAULT_SEEDS, DEFAULT_SIZES, E4Config, run_e4
from canonical_runner.e5 import E5Config, run_e5
from canonical_runner.runner import E1Config, run_e1
from canonical_runner.synthetic import REGIMES

FROZEN_SYSTEM_COMMIT = "cf9d988386b9b63f5ba559deb76e0f66b55c0fde"
E2_SYSTEM_COMMIT = "70f4943d9604cd2b5fe2df60e93521015d87fa74"
E5_SYSTEM_COMMIT = "d4a1a59b19d8c3c869fe691e59c6349b5db987c1"


def add_common_arguments(
    parser: argparse.ArgumentParser, *, system_commit: str = FROZEN_SYSTEM_COMMIT
) -> None:
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:6333")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bench-repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--system-repo", type=Path, required=True)
    parser.add_argument("--system-commit", default=system_commit)
    parser.add_argument("--system-artifact", required=True)
    parser.add_argument("--hardware-profile", default="apple-m4-pro-24gb-v1")
    parser.add_argument("--dense-name", default="dense")
    parser.add_argument("--sparse-name", default="sparse")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--weights", nargs=2, type=float, default=(1.0, 1.0))
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--allow-dirty", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="experiment", required=True)
    e1 = subparsers.add_parser("e1", help="ordered parity against exhaustive exact channels")
    add_common_arguments(e1)
    e1.add_argument("--max-live-oracle-points", type=int, default=10_000)
    e2 = subparsers.add_parser(
        "e2", help="paired proof-driven stopping versus native-bulk exact exhaustion"
    )
    add_common_arguments(e2, system_commit=E2_SYSTEM_COMMIT)
    e2.add_argument("--system-binary", type=Path)
    e2.add_argument("--system-build-manifest", type=Path)
    e2.add_argument("--hardware-manifest", type=Path)
    e2.add_argument("--warmups", type=int, default=2)
    e2.add_argument("--repetitions", type=int, default=5)
    e2.add_argument("--request-timeout-seconds", type=float, default=120.0)
    e3 = subparsers.add_parser("e3", help="exact fixed-prefix WRRF information frontier")
    add_common_arguments(e3, system_commit=E2_SYSTEM_COMMIT)
    e3.add_argument("--system-binary", type=Path)
    e3.add_argument("--system-build-manifest", type=Path)
    e3.add_argument("--depths", nargs="+", type=int, default=DEFAULT_DEPTHS)
    e4 = subparsers.add_parser("e4", help="controlled synthetic rank-regime execution")
    e4.add_argument("--output", type=Path, required=True)
    e4.add_argument(
        "--bench-repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    e4.add_argument("--hardware-profile", default="apple-m4-pro-24gb-v1")
    e4.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    e4.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    e4.add_argument("--regimes", nargs="+", choices=REGIMES, default=REGIMES)
    e4.add_argument("--limit", type=int, default=20)
    e4.add_argument("--rrf-k", type=int, default=60)
    e4.add_argument("--weights", nargs=2, type=float, default=(1.0, 1.0))
    e4.add_argument("--batch-size", type=int, default=16)
    e4.add_argument("--allow-dirty", action="store_true")
    e5 = subparsers.add_parser("e5", help="proof-driven Dense/Sparse producer ablation")
    add_common_arguments(e5, system_commit=E5_SYSTEM_COMMIT)
    e5.add_argument("--system-binary", type=Path, required=True)
    e5.add_argument("--system-build-manifest", type=Path, required=True)
    e5.add_argument("--warmups", type=int, default=2)
    e5.add_argument("--repetitions", type=int, default=5)
    e5.add_argument("--process-starts", type=int, default=3)
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
    elif args.experiment == "e2":
        summary = run_e2(
            E2Config(
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
                hardware_manifest=args.hardware_manifest,
                system_binary=args.system_binary,
                system_build_manifest=args.system_build_manifest,
                dense_name=args.dense_name,
                sparse_name=args.sparse_name,
                limit=args.limit,
                rrf_k=args.rrf_k,
                weights=tuple(args.weights),
                warmups=args.warmups,
                repetitions=args.repetitions,
                request_timeout_seconds=args.request_timeout_seconds,
                query_limit=args.query_limit,
                allow_dirty=args.allow_dirty,
            )
        )
        print(json.dumps(summary, sort_keys=True))
        if (
            summary["mismatchQueries"] > 0
            or summary["timeoutQueries"] > 0
            or summary["errorQueries"] > 0
        ):
            raise SystemExit(2)
    elif args.experiment == "e3":
        summary = run_e3(
            E3Config(
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
                system_binary=args.system_binary,
                system_build_manifest=args.system_build_manifest,
                dense_name=args.dense_name,
                sparse_name=args.sparse_name,
                limit=args.limit,
                rrf_k=args.rrf_k,
                weights=tuple(args.weights),
                depths=tuple(args.depths),
                query_limit=args.query_limit,
                allow_dirty=args.allow_dirty,
            )
        )
        print(json.dumps(summary, sort_keys=True))
        if summary["timeoutQueries"] > 0 or summary["errorQueries"] > 0:
            raise SystemExit(2)
    elif args.experiment == "e4":
        summary = run_e4(
            E4Config(
                output=args.output,
                bench_repo=args.bench_repo,
                hardware_profile=args.hardware_profile,
                sizes=tuple(args.sizes),
                seeds=tuple(args.seeds),
                regimes=tuple(args.regimes),
                limit=args.limit,
                rrf_k=args.rrf_k,
                weights=tuple(args.weights),
                batch_size=args.batch_size,
                allow_dirty=args.allow_dirty,
            )
        )
        print(json.dumps(summary, sort_keys=True))
        if (
            summary["mismatchQueries"] > 0
            or summary["timeoutQueries"] > 0
            or summary["errorQueries"] > 0
        ):
            raise SystemExit(2)
    elif args.experiment == "e5":
        summary = run_e5(
            E5Config(
                artifact_root=args.artifact_root,
                dataset=args.dataset,
                collection=args.collection,
                output=args.output,
                bench_repo=args.bench_repo,
                system_repo=args.system_repo,
                system_commit=args.system_commit,
                system_artifact=args.system_artifact,
                hardware_profile=args.hardware_profile,
                system_binary=args.system_binary,
                system_build_manifest=args.system_build_manifest,
                dense_name=args.dense_name,
                sparse_name=args.sparse_name,
                limit=args.limit,
                rrf_k=args.rrf_k,
                weights=tuple(args.weights),
                warmups=args.warmups,
                repetitions=args.repetitions,
                process_starts=args.process_starts,
                query_limit=args.query_limit,
                allow_dirty=args.allow_dirty,
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
