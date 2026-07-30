"""E5 producer ablation under one proof-driven ED-WRRF execution."""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from .artifacts import (
    CollectionSnapshot,
    DatasetSnapshot,
    load_collection_snapshot,
    load_dataset_snapshot,
)
from .client import E5_PRODUCER_PLANS, ExactRrfResult, QueryClient
from .logs import AtomicJsonlWriter
from .provenance import (
    canonical_hash,
    git_is_dirty,
    git_revision,
    runner_source_sha256,
    runtime_metadata,
    verify_system_build_manifest,
)
from .runner import SCHEMA, _mismatch, _sha256_output
from .server import ManagedQdrant, ManagedServerEvidence, sha256_file

PRODUCERS = tuple(E5_PRODUCER_PLANS)
REFERENCE_PRODUCER = "pvs-pbm"


@dataclass(frozen=True)
class E5Config:
    artifact_root: Path
    dataset: str
    collection: str
    output: Path
    bench_repo: Path
    system_repo: Path
    system_commit: str
    system_artifact: str
    hardware_profile: str
    system_binary: Path
    system_build_manifest: Path
    dense_name: str = "dense"
    sparse_name: str = "sparse"
    limit: int = 20
    rrf_k: int = 60
    weights: tuple[float, float] = (1.0, 1.0)
    warmups: int = 2
    repetitions: int = 5
    process_starts: int = 3
    query_limit: int | None = None
    allow_dirty: bool = False


def _validate(config: E5Config) -> None:
    if config.limit <= 0 or config.rrf_k <= 0:
        raise ValueError("limit and WRRF k must be positive")
    if any(not math.isfinite(weight) or weight < 0 for weight in config.weights) or not any(
        weight > 0 for weight in config.weights
    ):
        raise ValueError("weights must be finite and non-negative with one positive value")
    if config.warmups < 0 or config.repetitions <= 0 or config.process_starts <= 0:
        raise ValueError("warmups must be non-negative; repetitions and process starts positive")
    if config.query_limit is not None and config.query_limit <= 0:
        raise ValueError("query limit must be positive")
    if config.query_limit is not None and not config.allow_dirty:
        raise RuntimeError("query-limited E5 is a development dry run and requires --allow-dirty")
    if not config.system_artifact.strip():
        raise ValueError("system artifact digest must be non-empty")


def _run_record(
    config: E5Config,
    snapshot: DatasetSnapshot,
    collection_snapshot: CollectionSnapshot,
    run_id: str,
    dirty: bool,
    binary_sha256: str,
    build_manifest_sha256: str,
) -> dict[str, Any]:
    runtime = runtime_metadata(config.hardware_profile)
    return {
        "recordType": "run",
        "schema": SCHEMA,
        "runId": run_id,
        "experiment": "E5",
        "method": "proof-driven-producer-ablation",
        "dataset": snapshot.dataset,
        "split": snapshot.split,
        "datasetManifestSha256": snapshot.source_manifest_sha256,
        "denseManifestSha256": snapshot.dense_manifest_sha256,
        "sparseManifestSha256": snapshot.sparse_manifest_sha256,
        "systemCommit": config.system_commit,
        "systemArtifact": config.system_artifact,
        "serverProvenance": {
            "mode": "managed-isolated-snapshot",
            "binarySha256": binary_sha256,
            "snapshotSha256": collection_snapshot.snapshot_sha256,
            "collectionSnapshotManifestSha256": collection_snapshot.manifest_sha256,
            "systemBuildManifestSha256": build_manifest_sha256,
        },
        "benchCommit": git_revision(config.bench_repo),
        "dirty": dirty,
        "runnerSourceSha256": runner_source_sha256(config.bench_repo),
        "buildProfile": "canonical-bench-release-v1",
        **runtime,
        "cacheState": "warm-counterbalanced",
        "collection": config.collection,
        "command": {
            "entrypoint": "scripts/run_canonical.py",
            "experiment": "e5",
            "dataset": snapshot.dataset,
        },
        "environmentAllowlist": {"python": runtime["python"]},
        "parameters": {
            "limit": config.limit,
            "rrfK": config.rrf_k,
            "weights": list(config.weights),
            "denseName": config.dense_name,
            "sparseName": config.sparse_name,
            "warmups": config.warmups,
            "repetitions": config.repetitions,
            "processStarts": config.process_starts,
            "queryLimit": config.query_limit,
            "producers": list(PRODUCERS),
            "referenceProducer": REFERENCE_PRODUCER,
            "indexBytes": None,
            "indexBytesReason": "shared-snapshot-component-bytes-not-attributable",
            "buildTimeNs": None,
            "buildTimeReason": "query-only-ablation-without-clean-rebuild",
        },
        "startedAtUtc": datetime.now(UTC).isoformat(),
    }


def _invoke(
    client: QueryClient,
    producer: str,
    query: Any,
    config: E5Config,
) -> tuple[ExactRrfResult, int]:
    started = time.perf_counter_ns()
    result = client.producer_rrf(
        query,
        producer=producer,
        dense_name=config.dense_name,
        sparse_name=config.sparse_name,
        k=config.rrf_k,
        weights=config.weights,
        limit=config.limit,
    )
    return result, time.perf_counter_ns() - started


def _measurement(result: ExactRrfResult, latency_ns: int) -> dict[str, Any]:
    execution = result.execution
    return {
        "latencyNs": latency_ns,
        "plan": execution.get("plan"),
        "stopReason": execution.get("stopReason"),
        "sourcePulls": execution.get("sourcePulls"),
        "sourceExhausted": execution.get("sourceExhausted"),
        "certificationChecks": execution.get("certificationChecks"),
        "sourcePointsMaterialized": execution.get("sourcePointsMaterialized"),
        "corpusPointsObserved": execution.get("corpusPointsObserved"),
        "exhaustiveFallback": execution.get("exhaustiveFallback"),
        "producer": execution.get("producer"),
    }


def _rotated_order(query_index: int, observation: int, process_start: int) -> list[str]:
    offset = (query_index + observation + process_start) % len(PRODUCERS)
    return list(PRODUCERS[offset:] + PRODUCERS[:offset])


def run_e5(config: E5Config) -> dict[str, Any]:
    _validate(config)
    actual_system_commit = git_revision(config.system_repo)
    if actual_system_commit != config.system_commit:
        raise RuntimeError(
            f"system commit mismatch: {actual_system_commit} != {config.system_commit}"
        )
    dirty = git_is_dirty(config.bench_repo) or git_is_dirty(config.system_repo)
    if dirty and not config.allow_dirty:
        raise RuntimeError("canonical runner refuses a dirty bench or system repository")

    snapshot = load_dataset_snapshot(config.artifact_root, config.dataset)
    collection_snapshot = load_collection_snapshot(
        config.artifact_root, snapshot, config.collection
    )
    binary_sha256 = sha256_file(config.system_binary)
    if config.system_artifact != f"sha256:{binary_sha256}":
        raise RuntimeError("system artifact does not match the managed Qdrant binary")
    build_manifest_sha256 = verify_system_build_manifest(
        config.system_build_manifest,
        binary_sha256=binary_sha256,
        system_commit=config.system_commit,
    )

    queries = snapshot.queries[: config.query_limit]
    run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:12]}"
    attempted = ok = mismatch_count = timeout_count = error_count = 0
    sequence = 0
    query_hashes: list[str] = []
    observation_count = config.warmups + config.repetitions

    with AtomicJsonlWriter(config.output) as writer:
        writer.write(
            _run_record(
                config,
                snapshot,
                collection_snapshot,
                run_id,
                dirty,
                binary_sha256,
                build_manifest_sha256,
            )
        )
        for process_start in range(config.process_starts):
            with ManagedQdrant(
                binary=config.system_binary,
                system_repo=config.system_repo,
                collection=config.collection,
                snapshot=collection_snapshot.path,
            ) as server:
                initial_collection = _validate_server(
                    server,
                    config,
                    snapshot,
                    collection_snapshot,
                    binary_sha256,
                )
                with QueryClient(server.url, config.collection) as client:
                    for query_index, query in enumerate(queries):
                        for observation in range(observation_count):
                            attempted += 1
                            warmup = observation < config.warmups
                            repetition = observation if warmup else observation - config.warmups + 1
                            order = _rotated_order(query_index, observation, process_start)
                            validation_started = time.perf_counter_ns()
                            try:
                                results: dict[str, ExactRrfResult] = {}
                                measurements: dict[str, dict[str, Any]] = {}
                                for producer in order:
                                    result, latency_ns = _invoke(client, producer, query, config)
                                    results[producer] = result
                                    measurements[producer] = _measurement(result, latency_ns)
                                reference = list(results[REFERENCE_PRODUCER].point_ids)
                                ordered_by_producer = {
                                    producer: list(results[producer].point_ids)
                                    for producer in PRODUCERS
                                }
                                mismatch_variants = [
                                    producer
                                    for producer in PRODUCERS
                                    if ordered_by_producer[producer] != reference
                                ]
                                first_actual = (
                                    ordered_by_producer[mismatch_variants[0]]
                                    if mismatch_variants
                                    else reference
                                )
                                membership_mismatch, order_mismatch = _mismatch(
                                    reference, first_actual
                                )
                                status = "mismatch" if mismatch_variants else "ok"
                                if mismatch_variants:
                                    mismatch_count += 1
                                else:
                                    ok += 1
                                record: dict[str, Any] = {
                                    "recordType": "query",
                                    "runId": run_id,
                                    "queryId": query.query_id,
                                    "sequence": sequence,
                                    "processStart": process_start + 1,
                                    "repetition": repetition,
                                    "warmup": warmup,
                                    "counterbalanceOrder": order,
                                    "status": status,
                                    "orderedIds": first_actual,
                                    "oracleOrderedIds": reference,
                                    "orderedResultSha256": canonical_hash(first_actual),
                                    "oracleOrderedResultSha256": canonical_hash(reference),
                                    "membershipMismatch": membership_mismatch,
                                    "orderMismatch": order_mismatch,
                                    "tieMismatch": None,
                                    "mismatchVariants": mismatch_variants,
                                    "orderedIdsByProducer": ordered_by_producer,
                                    "measurements": measurements,
                                    "latencyNs": measurements[REFERENCE_PRODUCER]["latencyNs"],
                                    "validationLatencyNs": (
                                        time.perf_counter_ns() - validation_started
                                    ),
                                }
                            except httpx.TimeoutException as error:
                                timeout_count += 1
                                record = _error_record(
                                    run_id,
                                    query.query_id,
                                    sequence,
                                    process_start,
                                    repetition,
                                    warmup,
                                    validation_started,
                                    "timeout",
                                    error,
                                )
                            except (httpx.HTTPError, RuntimeError, ValueError) as error:
                                error_count += 1
                                record = _error_record(
                                    run_id,
                                    query.query_id,
                                    sequence,
                                    process_start,
                                    repetition,
                                    warmup,
                                    validation_started,
                                    "error",
                                    error,
                                )
                            writer.write(record)
                            query_hashes.append(canonical_hash(record))
                            sequence += 1
                    final_collection = client.collection_info()
                    if final_collection["pointsCount"] != initial_collection[
                        "pointsCount"
                    ] or canonical_hash(final_collection["config"]) != canonical_hash(
                        initial_collection["config"]
                    ):
                        raise RuntimeError(
                            "Collection point count or configuration changed during E5"
                        )

        summary = {
            "recordType": "summary",
            "runId": run_id,
            "attemptedQueries": attempted,
            "uniqueQueries": len(queries),
            "warmupObservations": len(queries) * config.warmups * config.process_starts,
            "measuredObservations": len(queries) * config.repetitions * config.process_starts,
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


def _validate_server(
    server: ManagedServerEvidence,
    config: E5Config,
    snapshot: DatasetSnapshot,
    collection_snapshot: CollectionSnapshot,
    binary_sha256: str,
) -> dict[str, Any]:
    if server.binary_sha256 != binary_sha256:
        raise RuntimeError("managed E5 process binary differs from the attested artifact")
    if server.snapshot_sha256 != collection_snapshot.snapshot_sha256:
        raise RuntimeError("managed E5 process snapshot differs from the frozen artifact")
    with QueryClient(server.url, config.collection) as client:
        server_info = client.server_info()
        if server_info.get("commit") != config.system_commit:
            raise RuntimeError("managed Qdrant runtime commit does not match E5 system commit")
        collection = client.collection_info()
        if collection["pointsCount"] != snapshot.document_count:
            raise RuntimeError("E5 collection/source document count mismatch")
        if canonical_hash(collection["config"]) != collection_snapshot.collection_config_sha256:
            raise RuntimeError("restored E5 Collection config differs from frozen attestation")
        return collection


def _error_record(
    run_id: str,
    query_id: str,
    sequence: int,
    process_start: int,
    repetition: int,
    warmup: bool,
    validation_started: int,
    status: str,
    error: Exception,
) -> dict[str, Any]:
    return {
        "recordType": "query",
        "runId": run_id,
        "queryId": query_id,
        "sequence": sequence,
        "processStart": process_start + 1,
        "repetition": repetition,
        "warmup": warmup,
        "status": status,
        "latencyNs": None,
        "validationLatencyNs": time.perf_counter_ns() - validation_started,
        "errorType": type(error).__name__,
        "error": str(error),
    }
