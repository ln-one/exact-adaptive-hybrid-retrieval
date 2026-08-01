"""Paired E2 execution: proof-driven stopping versus native bulk exhaustion."""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import httpx

from .artifacts import (
    CollectionSnapshot,
    DatasetSnapshot,
    QueryInput,
    load_collection_snapshot,
    load_dataset_snapshot,
)
from .client import ExactRrfResult, QueryClient
from .logs import AtomicJsonlWriter
from .provenance import (
    canonical_hash,
    git_is_dirty,
    git_revision,
    runner_source_sha256,
    runtime_metadata,
    verify_hardware_manifest,
    verify_system_build_manifest,
)
from .runner import SCHEMA, _mismatch, _sha256_output
from .server import ManagedQdrant, ManagedServerEvidence, sha256_file

E2_BASELINES = ("same-producer-exhaustive", "native-bulk-exhaustive")


@dataclass(frozen=True)
class E2Config:
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
    hardware_manifest: Path | None = None
    system_binary: Path | None = None
    system_build_manifest: Path | None = None
    dense_name: str = "dense"
    sparse_name: str = "sparse"
    limit: int = 20
    rrf_k: int = 60
    weights: tuple[float, float] = (1.0, 1.0)
    warmups: int = 2
    repetitions: int = 5
    request_timeout_seconds: float = 120.0
    query_limit: int | None = None
    allow_dirty: bool = False
    baseline: str = "native-bulk-exhaustive"


def _validate(config: E2Config) -> None:
    if config.limit <= 0 or config.rrf_k <= 0:
        raise ValueError("limit and WRRF k must be positive")
    if any(not math.isfinite(weight) or weight < 0 for weight in config.weights) or not any(
        weight > 0 for weight in config.weights
    ):
        raise ValueError("weights must be finite and non-negative with one positive value")
    if config.warmups < 0 or config.repetitions <= 0:
        raise ValueError("warmups must be non-negative and repetitions must be positive")
    if not math.isfinite(config.request_timeout_seconds) or config.request_timeout_seconds <= 0:
        raise ValueError("request timeout must be finite and positive")
    if config.baseline not in E2_BASELINES:
        raise ValueError(f"unsupported E2 baseline: {config.baseline}")
    if config.query_limit is not None and config.query_limit <= 0:
        raise ValueError("query limit must be positive")
    if config.query_limit is not None and not config.allow_dirty:
        raise RuntimeError("query-limited E2 is a development dry run and requires --allow-dirty")
    if not config.system_artifact.strip():
        raise ValueError("system artifact digest must be non-empty")
    if config.system_binary is None and not config.allow_dirty:
        raise RuntimeError(
            "publication E2 requires a managed --system-binary and canonical snapshot"
        )
    if config.system_build_manifest is None and not config.allow_dirty:
        raise RuntimeError("publication E2 requires --system-build-manifest")
    if config.hardware_manifest is None and not config.allow_dirty:
        raise RuntimeError("publication E2 requires --hardware-manifest")


def _run_record(
    config: E2Config,
    snapshot: DatasetSnapshot,
    run_id: str,
    dirty: bool,
    server_info: dict[str, Any],
    collection_info: dict[str, Any],
    server_evidence: ManagedServerEvidence | None,
    collection_snapshot: CollectionSnapshot | None,
    system_build_manifest_sha256: str | None,
    hardware_manifest_sha256: str | None,
) -> dict[str, Any]:
    runtime = runtime_metadata(config.hardware_profile)
    return {
        "recordType": "run",
        "schema": SCHEMA,
        "runId": run_id,
        "experiment": "E2",
        "method": f"paired-ed-wrrf-vs-{config.baseline}",
        "dataset": snapshot.dataset,
        "split": snapshot.split,
        "datasetManifestSha256": snapshot.source_manifest_sha256,
        "denseManifestSha256": snapshot.dense_manifest_sha256,
        "sparseManifestSha256": snapshot.sparse_manifest_sha256,
        "systemCommit": config.system_commit,
        "systemArtifact": config.system_artifact,
        "serverProvenance": (
            {
                "mode": "managed-isolated-snapshot",
                "binarySha256": server_evidence.binary_sha256,
                "snapshotSha256": server_evidence.snapshot_sha256,
                "collectionSnapshotManifestSha256": collection_snapshot.manifest_sha256,
                "systemBuildManifestSha256": system_build_manifest_sha256,
            }
            if server_evidence is not None and collection_snapshot is not None
            else {"mode": "external-unbound-development"}
        ),
        "benchCommit": git_revision(config.bench_repo),
        "dirty": dirty,
        "runnerSourceSha256": runner_source_sha256(config.bench_repo),
        "hardwareManifestSha256": hardware_manifest_sha256,
        "buildProfile": "canonical-bench-release-v1",
        **runtime,
        "cacheState": "warm-counterbalanced",
        "collection": config.collection,
        "server": server_info,
        "collectionConfigSha256": canonical_hash(collection_info["config"]),
        "collectionPoints": collection_info["pointsCount"],
        "collectionIndexedVectors": collection_info["indexedVectorsCount"],
        "collectionSegments": collection_info["segmentsCount"],
        "command": {
            "entrypoint": "scripts/run_canonical.py",
            "experiment": "e2",
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
            "requestTimeoutSeconds": config.request_timeout_seconds,
            "queryLimit": config.query_limit,
            "producer": "pvs-pbm",
            "baseline": config.baseline,
        },
        "startedAtUtc": datetime.now(UTC).isoformat(),
    }


def _invoke(
    operation: Callable[..., ExactRrfResult],
    query: QueryInput,
    config: E2Config,
) -> tuple[ExactRrfResult, int]:
    started = time.perf_counter_ns()
    result = operation(
        query,
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
        "sourcePulls": execution.get("sourcePulls"),
        "sourceExhausted": execution.get("sourceExhausted"),
        "certificationChecks": execution.get("certificationChecks"),
        "queryRounds": execution.get("queryRounds"),
        "sourcePointsMaterialized": execution.get("sourcePointsMaterialized"),
        "corpusPointsObserved": execution.get("corpusPointsObserved"),
        "exhaustiveFallback": execution.get("exhaustiveFallback"),
        "stopReason": execution.get("stopReason"),
        "plan": execution.get("plan"),
        "producer": execution.get("producer"),
    }


def run_e2(config: E2Config) -> dict[str, Any]:
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
    collection_snapshot = (
        load_collection_snapshot(config.artifact_root, snapshot, config.collection)
        if config.system_binary is not None
        else None
    )
    system_build_manifest_sha256 = None
    hardware_manifest_sha256 = None
    if config.system_binary is not None:
        binary_sha256 = sha256_file(config.system_binary)
        if config.system_artifact != f"sha256:{binary_sha256}":
            raise RuntimeError(
                "system artifact does not match the managed Qdrant binary: "
                f"{config.system_artifact} != sha256:{binary_sha256}"
            )
        if config.system_build_manifest is not None:
            system_build_manifest_sha256 = verify_system_build_manifest(
                config.system_build_manifest,
                binary_sha256=binary_sha256,
                system_commit=config.system_commit,
            )
    if config.hardware_manifest is not None:
        hardware_manifest_sha256 = verify_hardware_manifest(
            config.hardware_manifest,
            hardware_profile=config.hardware_profile,
        )
    queries = snapshot.queries[: config.query_limit]
    run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:12]}"
    attempted = ok = mismatch_count = timeout_count = error_count = 0
    query_hashes: list[str] = []

    server_context = (
        ManagedQdrant(
            binary=config.system_binary,
            system_repo=config.system_repo,
            collection=config.collection,
            snapshot=collection_snapshot.path,
        )
        if config.system_binary is not None and collection_snapshot is not None
        else nullcontext(None)
    )
    with server_context as server_evidence:
        url = server_evidence.url if server_evidence is not None else config.url
        with QueryClient(
            url,
            config.collection,
            timeout_seconds=config.request_timeout_seconds,
        ) as client:
            server_info = client.server_info()
            if server_evidence is not None and server_info.get("commit") != config.system_commit:
                raise RuntimeError(
                    "managed Qdrant runtime commit does not match the frozen system commit: "
                    f"{server_info.get('commit')} != {config.system_commit}"
                )
            initial_collection = client.collection_info()
            if initial_collection["pointsCount"] != snapshot.document_count:
                raise RuntimeError(
                    "collection/source document count mismatch: "
                    f"{initial_collection['pointsCount']} != {snapshot.document_count}"
                )
            if (
                collection_snapshot is not None
                and canonical_hash(initial_collection["config"])
                != collection_snapshot.collection_config_sha256
            ):
                raise RuntimeError("restored Collection config differs from its frozen attestation")
            with AtomicJsonlWriter(config.output) as writer:
                writer.write(
                    _run_record(
                        config,
                        snapshot,
                        run_id,
                        dirty,
                        server_info,
                        initial_collection,
                        server_evidence,
                        collection_snapshot,
                        system_build_manifest_sha256,
                        hardware_manifest_sha256,
                    )
                )
                dynamic_operation = partial(client.producer_rrf, producer="pvs-pbm")
                exhaustive_operation = partial(
                    client.producer_rrf,
                    producer="pvs-pbm",
                    mode=config.baseline,
                )
                sequence = 0
                observation_count = config.warmups + config.repetitions
                for query_index, query in enumerate(queries):
                    for observation in range(observation_count):
                        attempted += 1
                        warmup = observation < config.warmups
                        repetition = observation if warmup else observation - config.warmups + 1
                        dynamic_first = (query_index + observation) % 2 == 0
                        validation_started = time.perf_counter_ns()
                        try:
                            if dynamic_first:
                                dynamic, dynamic_ns = _invoke(dynamic_operation, query, config)
                                exhaustive, exhaustive_ns = _invoke(
                                    exhaustive_operation,
                                    query,
                                    config,
                                )
                            else:
                                exhaustive, exhaustive_ns = _invoke(
                                    exhaustive_operation,
                                    query,
                                    config,
                                )
                                dynamic, dynamic_ns = _invoke(dynamic_operation, query, config)
                            actual = list(dynamic.point_ids)
                            oracle = list(exhaustive.point_ids)
                            membership_mismatch, order_mismatch = _mismatch(oracle, actual)
                            mismatch = membership_mismatch or order_mismatch
                            status = "mismatch" if mismatch else "ok"
                            if mismatch:
                                mismatch_count += 1
                            else:
                                ok += 1
                            dynamic_measurement = _measurement(dynamic, dynamic_ns)
                            exhaustive_measurement = _measurement(exhaustive, exhaustive_ns)
                            source_pull_ratios = [
                                dynamic_pull / exhaustive_pull if exhaustive_pull else None
                                for dynamic_pull, exhaustive_pull in zip(
                                    dynamic_measurement["sourcePulls"],
                                    exhaustive_measurement["sourcePulls"],
                                    strict=True,
                                )
                            ]
                            record: dict[str, Any] = {
                                "recordType": "query",
                                "runId": run_id,
                                "queryId": query.query_id,
                                "sequence": sequence,
                                "repetition": repetition,
                                "warmup": warmup,
                                "counterbalanceOrder": (
                                    ["ed-wrrf", "exhaustive"]
                                    if dynamic_first
                                    else ["exhaustive", "ed-wrrf"]
                                ),
                                "status": status,
                                "latencyNs": dynamic_ns,
                                "baselineLatencyNs": exhaustive_ns,
                                "validationLatencyNs": time.perf_counter_ns() - validation_started,
                                "orderedIds": actual,
                                "oracleOrderedIds": oracle,
                                "orderedResultSha256": canonical_hash(actual),
                                "oracleOrderedResultSha256": canonical_hash(oracle),
                                "membershipMismatch": membership_mismatch,
                                "orderMismatch": order_mismatch,
                                "tieMismatch": None,
                                "sourcePulls": dynamic_measurement["sourcePulls"],
                                "sourceExhausted": dynamic_measurement["sourceExhausted"],
                                "certificationChecks": dynamic_measurement["certificationChecks"],
                                "sourcePointsMaterialized": dynamic_measurement[
                                    "sourcePointsMaterialized"
                                ],
                                "corpusPointsObserved": dynamic_measurement["corpusPointsObserved"],
                                "exhaustiveFallback": dynamic_measurement["exhaustiveFallback"],
                                "sourcePullRatios": source_pull_ratios,
                                "dynamic": dynamic_measurement,
                                "exhaustive": exhaustive_measurement,
                            }
                        except httpx.TimeoutException as error:
                            timeout_count += 1
                            record = {
                                "recordType": "query",
                                "runId": run_id,
                                "queryId": query.query_id,
                                "sequence": sequence,
                                "repetition": repetition,
                                "warmup": warmup,
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
                                "repetition": repetition,
                                "warmup": warmup,
                                "status": "error",
                                "latencyNs": None,
                                "validationLatencyNs": time.perf_counter_ns() - validation_started,
                                "errorType": type(error).__name__,
                                "error": str(error),
                            }
                        writer.write(record)
                        query_hashes.append(canonical_hash(record))
                        sequence += 1

                final_collection = client.collection_info()
                if final_collection["pointsCount"] != initial_collection[
                    "pointsCount"
                ] or canonical_hash(final_collection["config"]) != canonical_hash(
                    initial_collection["config"]
                ):
                    raise RuntimeError("collection point count or configuration changed during E2")
                summary = {
                    "recordType": "summary",
                    "runId": run_id,
                    "attemptedQueries": attempted,
                    "uniqueQueries": len(queries),
                    "warmupObservations": len(queries) * config.warmups,
                    "measuredObservations": len(queries) * config.repetitions,
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
