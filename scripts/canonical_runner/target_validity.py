"""Recoverable target-validity retrieval runner.

This campaign evaluates one retrieval result per Query. Diagnostic wall time is
retained for operations only; no repeated latency observations are produced.
"""

from __future__ import annotations

import json
import math
import os
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
from .client import ExactRrfResult, QueryClient
from .fusion import PointId, exact_wrrf_with_scores
from .provenance import (
    canonical_hash,
    git_is_dirty,
    git_revision,
    runner_source_sha256,
    runtime_metadata,
    verify_system_build_manifest,
)
from .server import ManagedQdrant, ManagedServerEvidence, sha256_file

SCHEMA = "target-validity-retrieval-v1"
DEFAULT_DEPTHS = (10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000)
LOCAL_COMPLETE_REFERENCE_DATASETS = frozenset({"nfcorpus", "scifact"})


@dataclass(frozen=True)
class TargetValidityConfig:
    artifact_root: Path
    dataset: str
    collection: str
    output: Path
    bench_repo: Path
    system_repo: Path
    system_commit: str
    system_artifact: str
    hardware_profile: str
    system_binary: Path | None = None
    system_build_manifest: Path | None = None
    url: str = "http://127.0.0.1:6333"
    dense_name: str = "dense"
    sparse_name: str = "sparse"
    limit: int = 100
    certificate_limit: int = 101
    rrf_k: int = 60
    weights: tuple[float, float] = (1.0, 1.0)
    depths: tuple[int, ...] = DEFAULT_DEPTHS
    request_timeout_seconds: float = 3_600.0
    query_limit: int | None = None
    allow_dirty: bool = False


def _validate(config: TargetValidityConfig) -> None:
    if config.limit <= 0 or config.certificate_limit != config.limit + 1:
        raise ValueError("certificate limit must equal quality limit plus one")
    if config.rrf_k <= 0 or config.request_timeout_seconds <= 0:
        raise ValueError("WRRF k and request timeout must be positive")
    if not config.depths or tuple(sorted(set(config.depths))) != config.depths:
        raise ValueError("fixed depths must be unique and strictly increasing")
    if config.depths[0] <= 0 or config.depths[-1] < config.limit:
        raise ValueError("fixed depths must be positive and cover the quality limit")
    if any(not math.isfinite(weight) or weight < 0 for weight in config.weights) or not any(
        weight > 0 for weight in config.weights
    ):
        raise ValueError("weights must be finite and non-negative with one positive value")
    if config.query_limit is not None and config.query_limit <= 0:
        raise ValueError("query limit must be positive")
    if config.query_limit is not None and not config.allow_dirty:
        raise RuntimeError("query-limited target-validity runs require --allow-dirty")
    if config.system_binary is None and not config.allow_dirty:
        raise RuntimeError("canonical target-validity execution requires --system-binary")
    if config.system_build_manifest is None and not config.allow_dirty:
        raise RuntimeError("canonical target-validity execution requires a build manifest")


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"immutable artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, path)
    temporary.unlink()


def _query_path(output: Path, sequence: int, query_id: str) -> Path:
    suffix = canonical_hash(query_id)[:12]
    return output / "queries" / f"{sequence:04d}-{suffix}.json"


def _failure_path(output: Path, sequence: int, query_id: str) -> Path:
    suffix = canonical_hash(query_id)[:12]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return output / "failures" / f"{sequence:04d}-{suffix}-{stamp}.json"


def _method_order(
    point_ids: list[PointId] | tuple[PointId, ...],
    external_ids: dict[PointId, str],
) -> list[str]:
    return [external_ids[point_id] for point_id in point_ids]


def _assert_same_result(reference: ExactRrfResult, actual: ExactRrfResult) -> None:
    if reference.point_ids != actual.point_ids:
        raise RuntimeError("native-bulk Full WRRF and EAHR identities/order differ")
    if bool(reference.point_scores) != bool(actual.point_scores):
        raise RuntimeError("native-bulk Full WRRF and EAHR score availability differs")
    if reference.point_scores and reference.point_scores != actual.point_scores:
        raise RuntimeError("native-bulk Full WRRF and EAHR fused scores differ")


def _local_complete_reference(
    client: QueryClient,
    query: Any,
    snapshot: DatasetSnapshot,
    config: TargetValidityConfig,
    full: ExactRrfResult,
) -> dict[str, Any] | None:
    if snapshot.dataset not in LOCAL_COMPLETE_REFERENCE_DATASETS:
        return None
    dense = client.exact_channel_prefix(
        query,
        channel="dense",
        limit=snapshot.document_count,
        corpus_points=snapshot.document_count,
        dense_name=config.dense_name,
        sparse_name=config.sparse_name,
    )
    sparse = client.exact_channel_prefix(
        query,
        channel="sparse",
        limit=snapshot.document_count,
        corpus_points=snapshot.document_count,
        dense_name=config.dense_name,
        sparse_name=config.sparse_name,
    )
    if not dense.exhausted or not sparse.exhausted:
        raise RuntimeError("local complete-reference channel did not report exhaustion")
    fused = exact_wrrf_with_scores(
        [dense.point_ids, sparse.point_ids],
        k=config.rrf_k,
        weights=config.weights,
        limit=config.certificate_limit,
    )
    local_ids = tuple(point_id for point_id, _ in fused)
    local_scores = tuple(float(score) for _, score in fused)
    if local_ids != full.point_ids:
        raise RuntimeError("local complete-support reconstruction differs from Full WRRF")
    if full.point_scores and local_scores != full.point_scores:
        raise RuntimeError("local complete-support scores differ from Full WRRF")
    return {
        "denseExhausted": dense.exhausted,
        "sparseExhausted": sparse.exhausted,
        "denseOrder": list(dense.point_ids),
        "sparsePositiveOrder": list(sparse.point_ids),
        "denseOrderSha256": canonical_hash(dense.point_ids),
        "sparseOrderSha256": canonical_hash(sparse.point_ids),
        "orderedTop101": list(local_ids),
        "scoresTop101": list(local_scores),
    }


def _boundary(result: ExactRrfResult, limit: int) -> dict[str, Any]:
    def item(rank: int) -> dict[str, Any] | None:
        index = rank - 1
        if index >= len(result.point_ids):
            return None
        return {
            "rank": rank,
            "pointId": result.point_ids[index],
            "score": result.point_scores[index] if result.point_scores else None,
        }

    return {"rankK": item(limit), "rankKPlusOne": item(limit + 1)}


def _run_spec(
    config: TargetValidityConfig,
    snapshot: DatasetSnapshot,
    *,
    bench_commit: str,
    dirty: bool,
    collection_snapshot: CollectionSnapshot | None,
    system_build_manifest_sha256: str | None,
) -> dict[str, Any]:
    runtime = runtime_metadata(config.hardware_profile)
    parameters = {
        "limit": config.limit,
        "certificateLimit": config.certificate_limit,
        "depths": list(config.depths),
        "rrfK": config.rrf_k,
        "weights": list(config.weights),
        "denseName": config.dense_name,
        "sparseName": config.sparse_name,
        "queryLimit": config.query_limit,
        "sparseSupport": "strictly-positive-query-score-only",
        "tieRule": "stable-system-point-identity-ascending",
        "prefixProducer": "one-hot-certified-exact-rank-stream",
        "timingEligibility": "diagnostic-only",
    }
    reproducibility = {
        "schema": SCHEMA,
        "dataset": snapshot.dataset,
        "datasetManifestSha256": snapshot.source_manifest_sha256,
        "denseManifestSha256": snapshot.dense_manifest_sha256,
        "sparseManifestSha256": snapshot.sparse_manifest_sha256,
        "collection": config.collection,
        "collectionSnapshotManifestSha256": (
            collection_snapshot.manifest_sha256 if collection_snapshot else None
        ),
        "systemCommit": config.system_commit,
        "systemArtifact": config.system_artifact,
        "systemBuildManifestSha256": system_build_manifest_sha256,
        "benchCommit": bench_commit,
        "runnerSourceSha256": runner_source_sha256(config.bench_repo),
        "parameters": parameters,
        "dirty": dirty,
    }
    return {
        "recordType": "run",
        **reproducibility,
        "runConfigSha256": canonical_hash(reproducibility),
        "split": snapshot.split,
        "documents": snapshot.document_count,
        "queries": len(snapshot.queries[: config.query_limit]),
        "hardware": runtime,
        "createdAtUtc": datetime.now(UTC).isoformat(),
    }


def _load_or_create_run(output: Path, spec: dict[str, Any]) -> dict[str, Any]:
    path = output / "run.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("runConfigSha256") != spec["runConfigSha256"]:
            raise RuntimeError("existing target-validity run has a different frozen config")
        return existing
    output.mkdir(parents=True, exist_ok=True)
    _write_json_exclusive(path, spec)
    return spec


def _verify_query_checkpoint(
    path: Path,
    *,
    query_id: str,
    run_config_sha256: str,
) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if (
        record.get("queryId") != query_id
        or record.get("runConfigSha256") != run_config_sha256
        or record.get("status") != "ok"
    ):
        raise RuntimeError(f"invalid existing Query checkpoint: {path}")
    expected_hash = record.pop("recordSha256", None)
    if expected_hash != canonical_hash(record):
        raise RuntimeError(f"existing Query checkpoint hash mismatch: {path}")
    record["recordSha256"] = expected_hash
    return record


def _session_record(
    output: Path,
    run: dict[str, Any],
    server_info: dict[str, Any],
    collection_info: dict[str, Any],
    evidence: ManagedServerEvidence | None,
) -> None:
    record = {
        "recordType": "session",
        "schema": SCHEMA,
        "runConfigSha256": run["runConfigSha256"],
        "startedAtUtc": datetime.now(UTC).isoformat(),
        "server": server_info,
        "collection": collection_info,
        "serverProvenance": (
            {
                "mode": "managed-isolated-snapshot",
                "binarySha256": evidence.binary_sha256,
                "snapshotSha256": evidence.snapshot_sha256,
            }
            if evidence is not None
            else {"mode": "external-unbound-development"}
        ),
    }
    name = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex[:8]}.json"
    _write_json_exclusive(output / "sessions" / name, record)


def _one_query(
    *,
    client: QueryClient,
    query: Any,
    sequence: int,
    snapshot: DatasetSnapshot,
    config: TargetValidityConfig,
    run_config_sha256: str,
) -> dict[str, Any]:
    started = time.perf_counter_ns()
    max_depth = config.depths[-1]
    dense = client.certified_channel_prefix(
        query,
        channel="dense",
        limit=max_depth,
        dense_name=config.dense_name,
        sparse_name=config.sparse_name,
        k=config.rrf_k,
    )
    sparse = client.certified_channel_prefix(
        query,
        channel="sparse",
        limit=max_depth,
        dense_name=config.dense_name,
        sparse_name=config.sparse_name,
        k=config.rrf_k,
    )
    full = client.producer_rrf(
        query,
        producer="pvs-pbm",
        mode="native-bulk-exhaustive",
        dense_name=config.dense_name,
        sparse_name=config.sparse_name,
        k=config.rrf_k,
        weights=config.weights,
        limit=config.certificate_limit,
    )
    eahr = client.exact_rrf(
        query,
        dense_name=config.dense_name,
        sparse_name=config.sparse_name,
        k=config.rrf_k,
        weights=config.weights,
        limit=config.certificate_limit,
    )
    _assert_same_result(full, eahr)
    local_reference = _local_complete_reference(client, query, snapshot, config, full)

    fixed_point_orders: dict[str, list[PointId]] = {}
    for depth in config.depths:
        fixed_point_orders[str(depth)] = [
            point_id
            for point_id, _ in exact_wrrf_with_scores(
                [dense.point_ids[:depth], sparse.point_ids[:depth]],
                k=config.rrf_k,
                weights=config.weights,
                limit=config.limit,
            )
        ]

    all_points: list[PointId] = [
        *dense.point_ids,
        *sparse.point_ids,
        *full.point_ids,
        *eahr.point_ids,
    ]
    external = client.external_ids(all_points)
    methods = {
        "dense": _method_order(dense.point_ids[: config.limit], external),
        "sparse": _method_order(sparse.point_ids[: config.limit], external),
        "full-wrrf": _method_order(full.point_ids[: config.limit], external),
        **{
            f"fixed-L{depth}": _method_order(order, external)
            for depth, order in ((int(key), value) for key, value in fixed_point_orders.items())
        },
    }
    record = {
        "recordType": "query",
        "schema": SCHEMA,
        "runConfigSha256": run_config_sha256,
        "dataset": snapshot.dataset,
        "queryId": query.query_id,
        "sequence": sequence,
        "status": "ok",
        "methods": methods,
        "densePrefixPointIds": list(dense.point_ids),
        "sparsePositivePrefixPointIds": list(sparse.point_ids),
        "densePrefixExternalIds": _method_order(dense.point_ids, external),
        "sparsePositivePrefixExternalIds": _method_order(sparse.point_ids, external),
        "prefix": {
            "denseFetchedPoints": dense.fetched_points,
            "sparseFetchedPoints": sparse.fetched_points,
            "denseRequestCount": dense.request_count,
            "sparseRequestCount": sparse.request_count,
            "denseExhausted": dense.exhausted,
            "sparseExhausted": sparse.exhausted,
        },
        "full": {
            "orderedPointIdsTop101": list(full.point_ids),
            "scoresTop101": list(full.point_scores),
            "scoresAvailable": bool(full.point_scores),
            "orderedExternalIdsTop100": methods["full-wrrf"],
            "boundary": _boundary(full, config.limit),
            "execution": full.execution,
        },
        "eahr": {
            "orderedPointIdsTop101": list(eahr.point_ids),
            "scoresTop101": list(eahr.point_scores),
            "scoresAvailable": bool(eahr.point_scores),
            "execution": eahr.execution,
            "matchesFull": True,
        },
        "localCompleteReference": local_reference,
        "diagnosticElapsedNs": time.perf_counter_ns() - started,
    }
    record["recordSha256"] = canonical_hash(record)
    return record


def run_target_validity(config: TargetValidityConfig) -> dict[str, Any]:
    _validate(config)
    actual_system_commit = git_revision(config.system_repo)
    if actual_system_commit != config.system_commit:
        raise RuntimeError(
            f"system commit mismatch: {actual_system_commit} != {config.system_commit}"
        )
    bench_commit = git_revision(config.bench_repo)
    dirty = git_is_dirty(config.bench_repo) or git_is_dirty(config.system_repo)
    if dirty and not config.allow_dirty:
        raise RuntimeError("canonical target-validity runner refuses dirty repositories")

    snapshot = load_dataset_snapshot(config.artifact_root, config.dataset)
    queries = snapshot.queries[: config.query_limit]
    collection_snapshot = (
        load_collection_snapshot(config.artifact_root, snapshot, config.collection)
        if config.system_binary is not None
        else None
    )
    build_manifest_sha256 = None
    if config.system_binary is not None:
        binary_sha256 = sha256_file(config.system_binary)
        if config.system_artifact != f"sha256:{binary_sha256}":
            raise RuntimeError("system artifact does not match the managed binary")
        if config.system_build_manifest is not None:
            build_manifest_sha256 = verify_system_build_manifest(
                config.system_build_manifest,
                binary_sha256=binary_sha256,
                system_commit=config.system_commit,
            )

    spec = _run_spec(
        config,
        snapshot,
        bench_commit=bench_commit,
        dirty=dirty,
        collection_snapshot=collection_snapshot,
        system_build_manifest_sha256=build_manifest_sha256,
    )
    run = _load_or_create_run(config.output, spec)
    summary_path = config.output / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

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
    attempted = completed = failed = 0
    with server_context as server_evidence:
        url = server_evidence.url if server_evidence is not None else config.url
        with QueryClient(
            url,
            config.collection,
            timeout_seconds=config.request_timeout_seconds,
        ) as client:
            server_info = client.server_info()
            if server_info.get("commit") != config.system_commit:
                raise RuntimeError("Qdrant runtime commit differs from the frozen system commit")
            collection_info = client.collection_info()
            if collection_info["pointsCount"] != snapshot.document_count:
                raise RuntimeError("collection/source document count mismatch")
            _session_record(config.output, run, server_info, collection_info, server_evidence)

            for sequence, query in enumerate(queries):
                destination = _query_path(config.output, sequence, query.query_id)
                if destination.exists():
                    _verify_query_checkpoint(
                        destination,
                        query_id=query.query_id,
                        run_config_sha256=run["runConfigSha256"],
                    )
                    completed += 1
                    continue
                attempted += 1
                try:
                    record = _one_query(
                        client=client,
                        query=query,
                        sequence=sequence,
                        snapshot=snapshot,
                        config=config,
                        run_config_sha256=run["runConfigSha256"],
                    )
                    _write_json_exclusive(destination, record)
                    completed += 1
                except (httpx.HTTPError, RuntimeError, ValueError) as error:
                    failed += 1
                    _write_json_exclusive(
                        _failure_path(config.output, sequence, query.query_id),
                        {
                            "recordType": "failure",
                            "schema": SCHEMA,
                            "runConfigSha256": run["runConfigSha256"],
                            "dataset": snapshot.dataset,
                            "queryId": query.query_id,
                            "sequence": sequence,
                            "status": "error",
                            "errorType": type(error).__name__,
                            "error": str(error),
                            "createdAtUtc": datetime.now(UTC).isoformat(),
                        },
                    )

    query_files = sorted((config.output / "queries").glob("*.json"))
    records = [
        _verify_query_checkpoint(
            path,
            query_id=queries[index].query_id,
            run_config_sha256=run["runConfigSha256"],
        )
        for index, path in enumerate(query_files)
    ]
    observed_ids = [record.get("queryId") for record in records]
    expected_ids = [query.query_id for query in queries]
    all_complete = len(records) == len(queries) and set(observed_ids) == set(expected_ids)
    failure_files = sorted((config.output / "failures").glob("*.json"))
    summary = {
        "recordType": "summary",
        "schema": SCHEMA,
        "runConfigSha256": run["runConfigSha256"],
        "dataset": snapshot.dataset,
        "expectedQueries": len(queries),
        "completedQueries": len(records),
        "attemptedThisSession": attempted,
        "failedThisSession": failed,
        "recordedFailureAttempts": len(failure_files),
        "unresolvedFailures": len(set(expected_ids) - set(observed_ids)),
        "allQueriesComplete": all_complete,
        "queryRecordSha256": canonical_hash(sorted(record["recordSha256"] for record in records)),
        "finishedAtUtc": datetime.now(UTC).isoformat(),
    }
    if all_complete:
        _write_json_exclusive(summary_path, summary)
        _write_json_exclusive(
            config.output / "manifest.json",
            {
                "schema": SCHEMA,
                "run": {"path": "run.json", "sha256": sha256_file(config.output / "run.json")},
                "summary": {
                    "path": "summary.json",
                    "sha256": sha256_file(config.output / "summary.json"),
                },
                "queries": [
                    {
                        "path": path.relative_to(config.output).as_posix(),
                        "sha256": sha256_file(path),
                    }
                    for path in query_files
                ],
                "failureAttempts": [
                    {
                        "path": path.relative_to(config.output).as_posix(),
                        "sha256": sha256_file(path),
                    }
                    for path in failure_files
                ],
            },
        )
    return summary
