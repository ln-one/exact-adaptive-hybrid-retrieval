"""E3 exact fixed-prefix information frontier against exhaustive WRRF."""

from __future__ import annotations

import math
import time
import uuid
from contextlib import nullcontext
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
from .client import QueryClient
from .fusion import PointId, exact_wrrf_with_scores
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

DEFAULT_DEPTHS = (20, 50, 100, 200, 500, 1_000, 2_000, 5_000)


@dataclass(frozen=True)
class E3Config:
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
    system_binary: Path | None = None
    system_build_manifest: Path | None = None
    dense_name: str = "dense"
    sparse_name: str = "sparse"
    limit: int = 20
    rrf_k: int = 60
    weights: tuple[float, float] = (1.0, 1.0)
    depths: tuple[int, ...] = DEFAULT_DEPTHS
    query_limit: int | None = None
    allow_dirty: bool = False


def _validate(config: E3Config) -> None:
    if config.limit <= 0 or config.rrf_k <= 0:
        raise ValueError("limit and WRRF k must be positive")
    if any(not math.isfinite(weight) or weight < 0 for weight in config.weights) or not any(
        weight > 0 for weight in config.weights
    ):
        raise ValueError("weights must be finite and non-negative with one positive value")
    if not config.depths or any(depth <= 0 for depth in config.depths):
        raise ValueError("fixed-prefix depths must be positive")
    if tuple(sorted(set(config.depths))) != config.depths:
        raise ValueError("fixed-prefix depths must be unique and strictly increasing")
    if config.query_limit is not None and config.query_limit <= 0:
        raise ValueError("query limit must be positive")
    if config.query_limit is not None and not config.allow_dirty:
        raise RuntimeError("query-limited E3 is a development dry run and requires --allow-dirty")
    if not config.system_artifact.strip():
        raise ValueError("system artifact digest must be non-empty")
    if config.system_binary is None and not config.allow_dirty:
        raise RuntimeError(
            "publication E3 requires a managed --system-binary and canonical snapshot"
        )
    if config.system_build_manifest is None and not config.allow_dirty:
        raise RuntimeError("publication E3 requires --system-build-manifest")


def _run_record(
    config: E3Config,
    snapshot: DatasetSnapshot,
    run_id: str,
    dirty: bool,
    server_info: dict[str, Any],
    collection_info: dict[str, Any],
    server_evidence: ManagedServerEvidence | None,
    collection_snapshot: CollectionSnapshot | None,
    system_build_manifest_sha256: str | None,
) -> dict[str, Any]:
    runtime = runtime_metadata(config.hardware_profile)
    return {
        "recordType": "run",
        "schema": SCHEMA,
        "runId": run_id,
        "experiment": "E3",
        "method": "exact-fixed-prefix-wrrf-frontier",
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
        "buildProfile": "canonical-bench-release-v1",
        **runtime,
        "cacheState": "correctness-frontier-no-latency-claim",
        "collection": config.collection,
        "server": server_info,
        "collectionConfigSha256": canonical_hash(collection_info["config"]),
        "collectionPoints": collection_info["pointsCount"],
        "collectionIndexedVectors": collection_info["indexedVectorsCount"],
        "collectionSegments": collection_info["segmentsCount"],
        "command": {
            "entrypoint": "scripts/run_canonical.py",
            "experiment": "e3",
            "dataset": snapshot.dataset,
        },
        "environmentAllowlist": {"python": runtime["python"]},
        "parameters": {
            "limit": config.limit,
            "rrfK": config.rrf_k,
            "weights": list(config.weights),
            "denseName": config.dense_name,
            "sparseName": config.sparse_name,
            "depths": list(config.depths),
            "queryLimit": config.query_limit,
            "oracle": "same-pvs-pbm-producers-forced-to-exhaustion",
            "prefixes": "qdrant-exact-channel-orders-with-complete-boundary-ties",
            "latencyEligibility": "none-information-frontier-only",
        },
        "startedAtUtc": datetime.now(UTC).isoformat(),
    }


def _common_record(
    *,
    run_id: str,
    query_id: str,
    sequence: int,
    depth: int,
) -> dict[str, Any]:
    return {
        "recordType": "query",
        "runId": run_id,
        "queryId": query_id,
        "sequence": sequence,
        "depth": depth,
    }


def _prefix_length(expected: list[PointId], actual: list[PointId]) -> int:
    length = 0
    for expected_id, actual_id in zip(expected, actual, strict=False):
        if expected_id != actual_id:
            break
        length += 1
    return length


def run_e3(config: E3Config) -> dict[str, Any]:
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
    if config.system_binary is not None:
        binary_sha256 = sha256_file(config.system_binary)
        if config.system_artifact != f"sha256:{binary_sha256}":
            raise RuntimeError("system artifact does not match the managed Qdrant binary")
        if config.system_build_manifest is not None:
            system_build_manifest_sha256 = verify_system_build_manifest(
                config.system_build_manifest,
                binary_sha256=binary_sha256,
                system_commit=config.system_commit,
            )

    queries = snapshot.queries[: config.query_limit]
    run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:12]}"
    attempted = ok = mismatch_count = timeout_count = error_count = 0
    query_hashes: list[str] = []
    sequence = 0

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
        with QueryClient(url, config.collection) as client:
            server_info = client.server_info()
            if server_evidence is not None and server_info.get("commit") != config.system_commit:
                raise RuntimeError("managed Qdrant runtime commit does not match E3 system commit")
            initial_collection = client.collection_info()
            if initial_collection["pointsCount"] != snapshot.document_count:
                raise RuntimeError("collection/source document count mismatch")
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
                    )
                )
                max_depth = config.depths[-1]
                for query in queries:
                    try:
                        oracle_started = time.perf_counter_ns()
                        oracle_result = client.exhaustive_rrf(
                            query,
                            dense_name=config.dense_name,
                            sparse_name=config.sparse_name,
                            k=config.rrf_k,
                            weights=config.weights,
                            limit=config.limit,
                        )
                        oracle_latency_ns = time.perf_counter_ns() - oracle_started
                        prefix_started = time.perf_counter_ns()
                        dense = client.exact_channel_prefix(
                            query,
                            channel="dense",
                            limit=max_depth,
                            corpus_points=snapshot.document_count,
                            dense_name=config.dense_name,
                            sparse_name=config.sparse_name,
                        )
                        sparse = client.exact_channel_prefix(
                            query,
                            channel="sparse",
                            limit=max_depth,
                            corpus_points=snapshot.document_count,
                            dense_name=config.dense_name,
                            sparse_name=config.sparse_name,
                        )
                        prefix_latency_ns = time.perf_counter_ns() - prefix_started
                        oracle = list(oracle_result.point_ids)
                        oracle_set = set(oracle)
                        for depth in config.depths:
                            dense_order = list(dense.point_ids[:depth])
                            sparse_order = list(sparse.point_ids[:depth])
                            actual = [
                                point_id
                                for point_id, _ in exact_wrrf_with_scores(
                                    [dense_order, sparse_order],
                                    k=config.rrf_k,
                                    weights=config.weights,
                                    limit=config.limit,
                                )
                            ]
                            membership_mismatch, order_mismatch = _mismatch(oracle, actual)
                            mismatch = membership_mismatch or order_mismatch
                            status = "mismatch" if mismatch else "ok"
                            attempted += 1
                            mismatch_count += mismatch
                            ok += not mismatch
                            candidate_union = set(dense_order) | set(sparse_order)
                            record = {
                                **_common_record(
                                    run_id=run_id,
                                    query_id=query.query_id,
                                    sequence=sequence,
                                    depth=depth,
                                ),
                                "status": status,
                                "orderedIds": actual,
                                "oracleOrderedIds": oracle,
                                "orderedResultSha256": canonical_hash(actual),
                                "oracleOrderedResultSha256": canonical_hash(oracle),
                                "membershipMismatch": membership_mismatch,
                                "orderMismatch": order_mismatch,
                                "tieMismatch": None,
                                "oracleRecall": (
                                    len(set(actual) & oracle_set) / len(oracle) if oracle else 1.0
                                ),
                                "exactPrefixLength": _prefix_length(oracle, actual),
                                "candidateUnionContainsOracle": oracle_set <= candidate_union,
                                "candidateUnionSize": len(candidate_union),
                                "densePrefixLength": len(dense_order),
                                "sparsePrefixLength": len(sparse_order),
                                "exposedRanks": len(dense_order) + len(sparse_order),
                                "oracleLatencyNs": oracle_latency_ns,
                                "prefixExtractionLatencyNs": prefix_latency_ns,
                                "prefixFetchedPoints": [
                                    dense.fetched_points,
                                    sparse.fetched_points,
                                ],
                                "prefixRequestCounts": [
                                    dense.request_count,
                                    sparse.request_count,
                                ],
                                "prefixExhausted": [dense.exhausted, sparse.exhausted],
                            }
                            writer.write(record)
                            query_hashes.append(canonical_hash(record))
                            sequence += 1
                    except httpx.TimeoutException as error:
                        status = "timeout"
                        timeout_count += len(config.depths)
                        for depth in config.depths:
                            record = {
                                **_common_record(
                                    run_id=run_id,
                                    query_id=query.query_id,
                                    sequence=sequence,
                                    depth=depth,
                                ),
                                "status": status,
                                "errorType": type(error).__name__,
                                "error": str(error),
                            }
                            writer.write(record)
                            query_hashes.append(canonical_hash(record))
                            sequence += 1
                            attempted += 1
                    except (httpx.HTTPError, RuntimeError, ValueError) as error:
                        status = "error"
                        error_count += len(config.depths)
                        for depth in config.depths:
                            record = {
                                **_common_record(
                                    run_id=run_id,
                                    query_id=query.query_id,
                                    sequence=sequence,
                                    depth=depth,
                                ),
                                "status": status,
                                "errorType": type(error).__name__,
                                "error": str(error),
                            }
                            writer.write(record)
                            query_hashes.append(canonical_hash(record))
                            sequence += 1
                            attempted += 1

                final_collection = client.collection_info()
                if final_collection["pointsCount"] != initial_collection[
                    "pointsCount"
                ] or canonical_hash(final_collection["config"]) != canonical_hash(
                    initial_collection["config"]
                ):
                    raise RuntimeError("collection point count or configuration changed during E3")
                summary = {
                    "recordType": "summary",
                    "runId": run_id,
                    "attemptedQueries": attempted,
                    "uniqueQueries": len(queries),
                    "depthObservations": attempted,
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
