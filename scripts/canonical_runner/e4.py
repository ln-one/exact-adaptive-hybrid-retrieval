"""E4 controlled synthetic rank-regime execution."""

from __future__ import annotations

import hashlib
import math
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .logs import AtomicJsonlWriter
from .provenance import (
    canonical_hash,
    git_is_dirty,
    git_revision,
    runner_source_sha256,
    runtime_metadata,
)
from .runner import SCHEMA, _mismatch, _sha256_output
from .synthetic import REGIMES, BalancedCertificate, generate_rankings

DEFAULT_SIZES = (100_000, 1_000_000, 5_000_000)
DEFAULT_SEEDS = (1_729, 2_027, 65_537)


@dataclass(frozen=True)
class E4Config:
    output: Path
    bench_repo: Path
    hardware_profile: str
    sizes: tuple[int, ...] = DEFAULT_SIZES
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    regimes: tuple[str, ...] = REGIMES
    limit: int = 20
    rrf_k: int = 60
    weights: tuple[float, float] = (1.0, 1.0)
    batch_size: int = 16
    allow_dirty: bool = False


def _validate(config: E4Config) -> None:
    if config.limit <= 0 or config.rrf_k <= 0 or config.batch_size <= 0:
        raise ValueError("limit, WRRF k, and batch size must be positive")
    if not config.sizes or any(size <= config.limit for size in config.sizes):
        raise ValueError("synthetic sizes must exceed the output limit")
    if tuple(sorted(set(config.sizes))) != config.sizes:
        raise ValueError("synthetic sizes must be unique and strictly increasing")
    if not config.seeds or len(set(config.seeds)) != len(config.seeds):
        raise ValueError("synthetic seeds must be non-empty and unique")
    if not config.regimes or len(set(config.regimes)) != len(config.regimes):
        raise ValueError("synthetic regimes must be non-empty and unique")
    if any(regime not in REGIMES for regime in config.regimes):
        raise ValueError("synthetic run contains an unsupported regime")
    if any(not math.isfinite(weight) or weight < 0 for weight in config.weights) or not any(
        weight > 0 for weight in config.weights
    ):
        raise ValueError("weights must be finite and non-negative with one positive value")


def _ranking_sha256(first: np.ndarray, second: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(first.tobytes())
    digest.update(second.tobytes())
    return digest.hexdigest()


def _run_record(config: E4Config, run_id: str, dirty: bool) -> dict[str, Any]:
    runtime = runtime_metadata(config.hardware_profile)
    return {
        "recordType": "run",
        "schema": SCHEMA,
        "runId": run_id,
        "experiment": "E4",
        "method": "balanced-fixed16-ed-wrrf-certificate",
        "dataset": "controlled-synthetic-rankings-v1",
        "split": "generated",
        "benchCommit": git_revision(config.bench_repo),
        "dirty": dirty,
        "runnerSourceSha256": runner_source_sha256(config.bench_repo),
        "buildProfile": "canonical-numpy-reference-v1",
        **runtime,
        "cacheState": "not-a-latency-experiment",
        "environmentAllowlist": {
            "python": runtime["python"],
            "numpy": np.__version__,
            "bitGenerator": "PCG64",
        },
        "parameters": {
            "limit": config.limit,
            "rrfK": config.rrf_k,
            "weights": list(config.weights),
            "batchSize": config.batch_size,
            "sizes": list(config.sizes),
            "seeds": list(config.seeds),
            "regimes": list(config.regimes),
            "schedule": "balanced-equal-depth",
            "latencyEligibility": "none-mechanism-and-work-ratio-only",
            "allTiedMeaning": "equal-raw-channel-scores-resolved-by-identity",
        },
        "startedAtUtc": datetime.now(UTC).isoformat(),
    }


def run_e4(config: E4Config) -> dict[str, Any]:
    _validate(config)
    dirty = git_is_dirty(config.bench_repo)
    if dirty and not config.allow_dirty:
        raise RuntimeError("canonical E4 refuses a dirty bench repository")

    run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:12]}"
    attempted = ok = mismatch_count = error_count = 0
    query_hashes: list[str] = []
    sequence = 0
    with AtomicJsonlWriter(config.output) as writer:
        writer.write(_run_record(config, run_id, dirty))
        for size in config.sizes:
            for regime in config.regimes:
                for seed in config.seeds:
                    attempted += 1
                    query_id = f"{regime}-n{size}-s{seed}"
                    started = time.perf_counter_ns()
                    try:
                        rankings = generate_rankings(size=size, seed=seed, regime=regime)
                        ranking_sha256 = _ranking_sha256(rankings.first, rankings.second)
                        certificate = BalancedCertificate(
                            rankings,
                            k=config.rrf_k,
                            weights=config.weights,
                            limit=config.limit,
                            batch_size=config.batch_size,
                        )
                        result = certificate.find_minimum_batch_depth()
                        actual = list(result.output)
                        oracle = list(certificate.oracle)
                        membership_mismatch, order_mismatch = _mismatch(oracle, actual)
                        mismatch = membership_mismatch or order_mismatch
                        status = "mismatch" if mismatch else "ok"
                        mismatch_count += mismatch
                        ok += not mismatch
                        record: dict[str, Any] = {
                            "recordType": "query",
                            "runId": run_id,
                            "queryId": query_id,
                            "sequence": sequence,
                            "status": status,
                            "regime": regime,
                            "size": size,
                            "seed": seed,
                            "rankingSha256": ranking_sha256,
                            "topWindowOverlap": rankings.top_window_overlap,
                            "latencyNs": time.perf_counter_ns() - started,
                            "orderedIds": actual,
                            "oracleOrderedIds": oracle,
                            "orderedResultSha256": canonical_hash(actual),
                            "oracleOrderedResultSha256": canonical_hash(oracle),
                            "membershipMismatch": membership_mismatch,
                            "orderMismatch": order_mismatch,
                            "tieMismatch": None,
                            "batchSize": config.batch_size,
                            "certificateDepth": result.depth,
                            "sourcePulls": [result.depth, result.depth],
                            "workRatio": result.depth / size,
                            "certificateChecks": result.checks,
                            "kthLowerAtStop": result.kth_lower,
                            "anonymousUpperAtStop": result.anonymous_upper,
                        }
                    except (MemoryError, RuntimeError, ValueError) as error:
                        error_count += 1
                        record = {
                            "recordType": "query",
                            "runId": run_id,
                            "queryId": query_id,
                            "sequence": sequence,
                            "status": "error",
                            "regime": regime,
                            "size": size,
                            "seed": seed,
                            "latencyNs": time.perf_counter_ns() - started,
                            "errorType": type(error).__name__,
                            "error": str(error),
                        }
                    writer.write(record)
                    query_hashes.append(canonical_hash(record))
                    sequence += 1

        summary = {
            "recordType": "summary",
            "runId": run_id,
            "attemptedQueries": attempted,
            "okQueries": ok,
            "mismatchQueries": mismatch_count,
            "timeoutQueries": 0,
            "errorQueries": error_count,
            "finishedAtUtc": datetime.now(UTC).isoformat(),
            "queryRecordSha256": canonical_hash(query_hashes),
        }
        writer.write(summary)
        writer.commit()
    _sha256_output(config.output)
    return summary
