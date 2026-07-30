"""E1 ordered-equivalence execution against Qdrant exhaustive channel orders."""

from __future__ import annotations

import hashlib
import math
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from .artifacts import DatasetSnapshot, load_dataset_snapshot
from .client import QueryClient
from .fusion import PointId, exact_wrrf_with_scores
from .logs import AtomicJsonlWriter
from .provenance import (
    canonical_hash,
    git_is_dirty,
    git_revision,
    runner_source_sha256,
    runtime_metadata,
)

SCHEMA = "ed-wrrf-results-v1"


@dataclass(frozen=True)
class E1Config:
    artifact_root: Path
    dataset: str
    collection: str
    url: str
    output: Path
    bench_repo: Path
    system_repo: Path
    system_commit: str
    system_artifact: str
    hardware_profile: str
    dense_name: str = "dense"
    sparse_name: str = "sparse"
    limit: int = 20
    rrf_k: int = 60
    weights: tuple[float, float] = (1.0, 1.0)
    query_limit: int | None = None
    allow_dirty: bool = False
    max_live_oracle_points: int = 10_000


def _mismatch(expected: list[PointId], actual: list[PointId]) -> tuple[bool, bool]:
    return set(expected) != set(actual), expected != actual and set(expected) == set(actual)


def _sha256_output(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    checksum_path = path.with_suffix(path.suffix + ".sha256")
    with checksum_path.open("x", encoding="utf-8") as handle:
        handle.write(f"{digest}  {path.name}\n")


def _run_record(
    config: E1Config,
    snapshot: DatasetSnapshot,
    run_id: str,
    dirty: bool,
    server_info: dict[str, Any],
    collection_info: dict[str, Any],
) -> dict[str, Any]:
    runtime = runtime_metadata(config.hardware_profile)
    return {
        "recordType": "run",
        "schema": SCHEMA,
        "runId": run_id,
        "experiment": "E1",
        "method": "ed-wrrf",
        "dataset": snapshot.dataset,
        "split": snapshot.split,
        "datasetManifestSha256": snapshot.source_manifest_sha256,
        "denseManifestSha256": snapshot.dense_manifest_sha256,
        "sparseManifestSha256": snapshot.sparse_manifest_sha256,
        "systemCommit": config.system_commit,
        "systemArtifact": config.system_artifact,
        "benchCommit": git_revision(config.bench_repo),
        "dirty": dirty,
        "runnerSourceSha256": runner_source_sha256(config.bench_repo),
        "buildProfile": "canonical-python-http-v1",
        **runtime,
        "cacheState": "correctness-validation",
        "repetition": 1,
        "warmup": False,
        "collection": config.collection,
        "server": server_info,
        "collectionConfigSha256": canonical_hash(collection_info["config"]),
        "collectionPoints": collection_info["pointsCount"],
        "collectionIndexedVectors": collection_info["indexedVectorsCount"],
        "collectionSegments": collection_info["segmentsCount"],
        "command": {
            "entrypoint": "scripts/run_canonical.py",
            "experiment": "e1",
            "dataset": snapshot.dataset,
        },
        "environmentAllowlist": {"python": runtime["python"]},
        "parameters": {
            "limit": config.limit,
            "rrfK": config.rrf_k,
            "weights": list(config.weights),
            "denseName": config.dense_name,
            "sparseName": config.sparse_name,
            "oracle": "qdrant-exhaustive-exact-channels",
            "queryLimit": config.query_limit,
        },
        "startedAtUtc": datetime.now(UTC).isoformat(),
    }


def run_e1(config: E1Config) -> dict[str, Any]:
    if config.limit <= 0 or config.rrf_k <= 0:
        raise ValueError("limit and WRRF k must be positive")
    if any(not math.isfinite(weight) or weight < 0 for weight in config.weights) or not any(
        weight > 0 for weight in config.weights
    ):
        raise ValueError("weights must be finite and non-negative with one positive value")
    if not config.system_artifact.strip():
        raise ValueError("system artifact digest must be non-empty")
    if config.query_limit is not None and config.query_limit <= 0:
        raise ValueError("query limit must be positive")
    if config.query_limit is not None and not config.allow_dirty:
        raise RuntimeError("query-limited E1 is a development dry run and requires --allow-dirty")

    bench_dirty = git_is_dirty(config.bench_repo)
    system_dirty = git_is_dirty(config.system_repo)
    actual_system_commit = git_revision(config.system_repo)
    if actual_system_commit != config.system_commit:
        raise RuntimeError(
            f"system commit mismatch: {actual_system_commit} != {config.system_commit}"
        )
    dirty = bench_dirty or system_dirty
    if dirty and not config.allow_dirty:
        raise RuntimeError("canonical runner refuses a dirty bench or system repository")

    snapshot = load_dataset_snapshot(config.artifact_root, config.dataset)
    if snapshot.document_count > config.max_live_oracle_points:
        raise RuntimeError(
            "live exhaustive HTTP oracle is bounded to "
            f"{config.max_live_oracle_points} points; use a frozen oracle artifact"
        )
    queries = snapshot.queries[: config.query_limit]
    run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:12]}"
    attempted = ok = mismatch_count = timeout_count = error_count = 0
    query_hashes: list[str] = []

    with QueryClient(config.url, config.collection) as client:
        server_info = client.server_info()
        initial_collection = client.collection_info()
        if initial_collection["pointsCount"] != snapshot.document_count:
            raise RuntimeError(
                "collection/source document count mismatch: "
                f"{initial_collection['pointsCount']} != {snapshot.document_count}"
            )
        with AtomicJsonlWriter(config.output) as writer:
            writer.write(
                _run_record(
                    config,
                    snapshot,
                    run_id,
                    dirty,
                    server_info,
                    initial_collection,
                )
            )
            for sequence, query in enumerate(queries):
                attempted += 1
                validation_started = time.perf_counter_ns()
                try:
                    oracle_started = time.perf_counter_ns()
                    dense_order = client.exact_channel_order(
                        query,
                        channel="dense",
                        limit=snapshot.document_count,
                        dense_name=config.dense_name,
                        sparse_name=config.sparse_name,
                    )
                    sparse_order = client.exact_channel_order(
                        query,
                        channel="sparse",
                        limit=snapshot.document_count,
                        dense_name=config.dense_name,
                        sparse_name=config.sparse_name,
                    )
                    oracle_scored = exact_wrrf_with_scores(
                        [dense_order, sparse_order],
                        k=config.rrf_k,
                        weights=config.weights,
                        limit=config.limit,
                    )
                    oracle = [point_id for point_id, _ in oracle_scored]
                    oracle_scores = dict(oracle_scored)
                    oracle_latency_ns = time.perf_counter_ns() - oracle_started
                    method_started = time.perf_counter_ns()
                    result = client.exact_rrf(
                        query,
                        dense_name=config.dense_name,
                        sparse_name=config.sparse_name,
                        k=config.rrf_k,
                        weights=config.weights,
                        limit=config.limit,
                    )
                    method_latency_ns = time.perf_counter_ns() - method_started
                    actual = list(result.point_ids)
                    membership_mismatch, order_mismatch = _mismatch(oracle, actual)
                    tie_mismatch = order_mismatch and any(
                        actual_id in oracle_scores
                        and expected_id in oracle_scores
                        and oracle_scores[actual_id] == oracle_scores[expected_id]
                        for actual_id, expected_id in zip(actual, oracle, strict=True)
                        if actual_id != expected_id
                    )
                    mismatch = membership_mismatch or order_mismatch
                    status = "mismatch" if mismatch else "ok"
                    if mismatch:
                        mismatch_count += 1
                    else:
                        ok += 1
                    execution = result.execution
                    record: dict[str, Any] = {
                        "recordType": "query",
                        "runId": run_id,
                        "queryId": query.query_id,
                        "sequence": sequence,
                        "status": status,
                        "latencyNs": method_latency_ns,
                        "oracleLatencyNs": oracle_latency_ns,
                        "validationLatencyNs": time.perf_counter_ns() - validation_started,
                        "orderedIds": actual,
                        "oracleOrderedIds": oracle,
                        "orderedResultSha256": canonical_hash(actual),
                        "oracleOrderedResultSha256": canonical_hash(oracle),
                        "membershipMismatch": membership_mismatch,
                        "orderMismatch": order_mismatch,
                        "tieMismatch": tie_mismatch,
                        "sourcePulls": execution.get("sourcePulls"),
                        "sourceExhausted": execution.get("sourceExhausted"),
                        "certificationChecks": execution.get("certificationChecks"),
                        "queryRounds": execution.get("queryRounds"),
                        "sourcePointsMaterialized": execution.get("sourcePointsMaterialized"),
                        "corpusPointsObserved": execution.get("corpusPointsObserved"),
                        "exhaustiveFallback": execution.get("exhaustiveFallback"),
                        "dense": {
                            "quantizedEvaluations": None,
                            "exactRescores": None,
                            "rankBatches": None,
                        },
                        "sparse": {
                            "postingElementsDecoded": None,
                            "blocksExpanded": None,
                            "rankBatches": None,
                        },
                        "process": {
                            "cpuTimeNs": None,
                            "peakRssBytes": None,
                            "majorPageFaults": None,
                            "minorPageFaults": None,
                        },
                    }
                except httpx.TimeoutException as error:
                    timeout_count += 1
                    record = {
                        "recordType": "query",
                        "runId": run_id,
                        "queryId": query.query_id,
                        "sequence": sequence,
                        "status": "timeout",
                        "latencyNs": None,
                        "validationLatencyNs": time.perf_counter_ns() - validation_started,
                        "errorType": type(error).__name__,
                        "error": str(error),
                    }
                except (httpx.HTTPError, RuntimeError, ValueError) as error:
                    error_count += 1
                    record = {
                        "recordType": "query",
                        "runId": run_id,
                        "queryId": query.query_id,
                        "sequence": sequence,
                        "status": "error",
                        "latencyNs": None,
                        "validationLatencyNs": time.perf_counter_ns() - validation_started,
                        "errorType": type(error).__name__,
                        "error": str(error),
                    }
                writer.write(record)
                query_hashes.append(canonical_hash(record))

            final_collection = client.collection_info()
            if final_collection["pointsCount"] != initial_collection[
                "pointsCount"
            ] or canonical_hash(final_collection["config"]) != canonical_hash(
                initial_collection["config"]
            ):
                raise RuntimeError("collection point count or configuration changed during E1")
            summary = {
                "recordType": "summary",
                "runId": run_id,
                "attemptedQueries": attempted,
                "okQueries": ok,
                "mismatchQueries": mismatch_count,
                "timeoutQueries": timeout_count,
                "errorQueries": error_count,
                "finishedAtUtc": datetime.now(UTC).isoformat(),
                "queryRecordSha256": canonical_hash(query_hashes),
            }
            writer.write(summary)
            writer.commit()

    _sha256_output(config.output)
    return summary
