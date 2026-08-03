"""Failure-contained E5-v2 producer ablation protocols."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .artifacts import (
    CollectionSnapshot,
    DatasetSnapshot,
    QueryInput,
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
    verify_hardware_manifest,
    verify_system_build_manifest,
)
from .runner import SCHEMA, _sha256_output
from .server import ManagedQdrant, ManagedServerEvidence, sha256_file

CAMPAIGN_SCHEMA = "ed-wrrf-e5-v2-campaign-v1"
EXPERIMENT = "E5-v2"
PRODUCERS = tuple(E5_PRODUCER_PLANS)
REFERENCE_PRODUCER = "pvs-pbm"
WILLIAMS_ORDERS = (
    ("pvs-pbm", "scalar-pbm", "pvs-sparse-materialized", "scan-pbm"),
    ("scalar-pbm", "scan-pbm", "pvs-pbm", "pvs-sparse-materialized"),
    ("scan-pbm", "pvs-sparse-materialized", "scalar-pbm", "pvs-pbm"),
    ("pvs-sparse-materialized", "pvs-pbm", "scan-pbm", "scalar-pbm"),
)
LOGICAL_FIELDS = (
    "sourcePulls",
    "sourceExhausted",
    "certificationChecks",
    "sourcePointsMaterialized",
    "stopReason",
)


def _protocol_labels(warmups: int) -> tuple[str, str]:
    if warmups == 0:
        return (
            "counterbalanced-producer-ablation",
            "counterbalanced-no-per-plan-warmup",
        )
    return (
        "method-self-warmed-producer-ablation",
        "method-self-warmed-blocked",
    )


@dataclass(frozen=True)
class E5V2PlanConfig:
    artifact_root: Path
    dataset: str
    collection: str
    output: Path
    bench_repo: Path
    system_repo: Path
    system_commit: str
    system_artifact: str
    hardware_profile: str
    hardware_manifest: Path
    system_binary: Path
    system_build_manifest: Path
    dense_name: str = "dense"
    sparse_name: str = "sparse"
    limit: int = 20
    rrf_k: int = 60
    weights: tuple[float, float] = (1.0, 1.0)
    rounds: int = 3
    shard_size: int = 0
    warmups: int = 2
    repetitions: int = 4
    request_timeout_seconds: float = 1_200.0
    startup_timeout_seconds: float = 1_800.0
    shard_wall_timeout_seconds: float = 21_600.0
    query_ids: tuple[str, ...] = ()
    allow_dirty: bool = False


@dataclass(frozen=True)
class E5V2ShardConfig:
    artifact_root: Path
    campaign_manifest: Path
    round_number: int
    shard_number: int
    output: Path
    failed_dir: Path
    bench_repo: Path
    system_repo: Path
    system_binary: Path
    system_build_manifest: Path
    hardware_manifest: Path
    allow_dirty: bool = False


class E5V2ShardRejected(RuntimeError):
    """A shard failed closed and its partial evidence was preserved."""


def _validate_common(config: E5V2PlanConfig) -> None:
    if config.limit <= 0 or config.rrf_k <= 0:
        raise ValueError("limit and WRRF k must be positive")
    if any(not math.isfinite(weight) or weight < 0 for weight in config.weights) or not any(
        weight > 0 for weight in config.weights
    ):
        raise ValueError("weights must be finite and non-negative with one positive value")
    if config.rounds <= 0 or config.warmups < 0 or config.repetitions <= 0:
        raise ValueError("rounds/repetitions must be positive and warmups non-negative")
    if config.shard_size < 0:
        raise ValueError("shard size cannot be negative")
    for name, value in (
        ("request timeout", config.request_timeout_seconds),
        ("startup timeout", config.startup_timeout_seconds),
        ("shard wall timeout", config.shard_wall_timeout_seconds),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if config.query_ids and not config.allow_dirty:
        raise RuntimeError("query-selected E5-v2 is a development dry run")


def _write_json_with_checksum(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"canonical artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    encoded = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    _sha256_output(path)


def _manifest_digest(path: Path) -> str:
    checksum = path.with_suffix(path.suffix + ".sha256")
    if not checksum.is_file():
        raise ValueError("E5-v2 campaign checksum is missing")
    expected = checksum.read_text(encoding="utf-8").split()[0]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected != actual:
        raise ValueError("E5-v2 campaign checksum mismatch")
    return actual


def _checked_file_digest(path: Path) -> str:
    checksum = path.with_suffix(path.suffix + ".sha256")
    if not checksum.is_file():
        raise ValueError(f"artifact checksum is missing: {path}")
    expected = checksum.read_text(encoding="utf-8").split()[0]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected != actual:
        raise ValueError(f"artifact checksum mismatch: {path}")
    return actual


def _superseded_failures(
    failed_dir: Path,
    campaign_id: str,
    round_number: int,
    shard_number: int,
) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    pattern = f"r{round_number:02d}-s{shard_number:02d}-*.failed.jsonl"
    for path in sorted(failed_dir.glob(pattern)):
        first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        if first.get("campaignId") != campaign_id:
            continue
        failures.append({"file": path.name, "sha256": _checked_file_digest(path)})
    return failures


def load_campaign(path: Path) -> dict[str, Any]:
    _manifest_digest(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != CAMPAIGN_SCHEMA:
        raise ValueError("unsupported E5-v2 campaign manifest")
    rounds = value.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        raise ValueError("E5-v2 campaign has no rounds")
    scheduled: list[tuple[int, int, str]] = []
    round_numbers = [round_spec.get("round") for round_spec in rounds]
    if round_numbers != list(range(1, len(rounds) + 1)):
        raise ValueError("E5-v2 round numbers must be contiguous and one-based")
    for round_spec in rounds:
        round_number = round_spec.get("round")
        shards = round_spec.get("shards")
        if not isinstance(round_number, int) or not isinstance(shards, list) or not shards:
            raise ValueError("invalid E5-v2 round specification")
        shard_numbers = [shard.get("shard") for shard in shards]
        if shard_numbers != list(range(1, len(shards) + 1)):
            raise ValueError("E5-v2 shard numbers must be contiguous and one-based")
        for shard in shards:
            shard_number = shard.get("shard")
            queries = shard.get("queries")
            if not isinstance(shard_number, int) or not isinstance(queries, list) or not queries:
                raise ValueError("invalid E5-v2 shard specification")
            for query in queries:
                query_id = query.get("queryId")
                order = query.get("blockOrder")
                if (
                    not isinstance(query_id, str)
                    or not isinstance(order, list)
                    or tuple(order) not in WILLIAMS_ORDERS
                ):
                    raise ValueError("invalid E5-v2 Query schedule")
                scheduled.append((round_number, shard_number, query_id))
    query_ids = value.get("queryIds")
    round_count = value.get("parameters", {}).get("rounds")
    if not isinstance(query_ids, list) or not isinstance(round_count, int):
        raise ValueError("E5-v2 campaign is missing its Query matrix")
    counts = {query_id: 0 for query_id in query_ids}
    for _, _, query_id in scheduled:
        if query_id not in counts:
            raise ValueError("E5-v2 schedule contains an undeclared Query")
        counts[query_id] += 1
    if any(count != round_count for count in counts.values()):
        raise ValueError("E5-v2 campaign does not schedule every Query in every round")
    return value


def _selected_queries(snapshot: DatasetSnapshot, query_ids: tuple[str, ...]) -> list[QueryInput]:
    if not query_ids:
        return list(snapshot.queries)
    by_id = {query.query_id: query for query in snapshot.queries}
    missing = [query_id for query_id in query_ids if query_id not in by_id]
    if missing:
        raise ValueError(f"Query IDs not found in dataset: {missing}")
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("E5-v2 Query IDs must be unique")
    return [by_id[query_id] for query_id in query_ids]


def _round_order(queries: list[QueryInput], round_index: int, rounds: int) -> list[QueryInput]:
    if not queries:
        return []
    offset = (len(queries) * round_index) // rounds
    return queries[offset:] + queries[:offset]


def _default_shard_size(snapshot: DatasetSnapshot, query_count: int) -> int:
    return 9 if snapshot.document_count >= 1_000_000 else query_count


def plan_e5_v2(config: E5V2PlanConfig) -> dict[str, Any]:
    _validate_common(config)
    if git_revision(config.system_repo) != config.system_commit:
        raise RuntimeError("system commit does not match the E5-v2 plan")
    dirty = git_is_dirty(config.bench_repo) or git_is_dirty(config.system_repo)
    if dirty and not config.allow_dirty:
        raise RuntimeError("publication E5-v2 planning refuses dirty repositories")
    snapshot = load_dataset_snapshot(config.artifact_root, config.dataset)
    collection = load_collection_snapshot(config.artifact_root, snapshot, config.collection)
    queries = _selected_queries(snapshot, config.query_ids)
    if not queries:
        raise ValueError("E5-v2 requires at least one Query")
    binary_sha256 = sha256_file(config.system_binary)
    if config.system_artifact != f"sha256:{binary_sha256}":
        raise RuntimeError("system artifact does not match the E5-v2 binary")
    build_sha256 = verify_system_build_manifest(
        config.system_build_manifest,
        binary_sha256=binary_sha256,
        system_commit=config.system_commit,
    )
    hardware_sha256 = verify_hardware_manifest(
        config.hardware_manifest,
        hardware_profile=config.hardware_profile,
    )
    shard_size = config.shard_size or _default_shard_size(snapshot, len(queries))
    canonical_index = {query.query_id: index for index, query in enumerate(snapshot.queries)}
    round_specs: list[dict[str, Any]] = []
    for round_index in range(config.rounds):
        ordered = _round_order(queries, round_index, config.rounds)
        shards: list[dict[str, Any]] = []
        for start in range(0, len(ordered), shard_size):
            shard_queries = ordered[start : start + shard_size]
            shards.append(
                {
                    "shard": len(shards) + 1,
                    "queries": [
                        {
                            "queryId": query.query_id,
                            "canonicalIndex": canonical_index[query.query_id],
                            "blockOrder": list(
                                WILLIAMS_ORDERS[
                                    (canonical_index[query.query_id] + round_index) % 4
                                ]
                            ),
                        }
                        for query in shard_queries
                    ],
                }
            )
        round_specs.append(
            {
                "round": round_index + 1,
                "rotation": (len(queries) * round_index) // config.rounds,
                "shards": shards,
            }
        )
    method, cache_state = _protocol_labels(config.warmups)
    campaign = {
        "schema": CAMPAIGN_SCHEMA,
        "campaignId": f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:12]}",
        "experiment": EXPERIMENT,
        "method": method,
        "dataset": snapshot.dataset,
        "split": snapshot.split,
        "collection": config.collection,
        "queryIds": [query.query_id for query in queries],
        "datasetManifestSha256": snapshot.source_manifest_sha256,
        "denseManifestSha256": snapshot.dense_manifest_sha256,
        "sparseManifestSha256": snapshot.sparse_manifest_sha256,
        "collectionSnapshotManifestSha256": collection.manifest_sha256,
        "collectionSnapshotSha256": collection.snapshot_sha256,
        "collectionConfigSha256": collection.collection_config_sha256,
        "collectionPoints": collection.points,
        "systemCommit": config.system_commit,
        "systemArtifact": config.system_artifact,
        "binarySha256": binary_sha256,
        "systemBuildManifestSha256": build_sha256,
        "hardwareProfile": config.hardware_profile,
        "hardwareManifestSha256": hardware_sha256,
        "benchCommit": git_revision(config.bench_repo),
        "runnerSourceSha256": runner_source_sha256(config.bench_repo),
        "campaignDriverSha256": sha256_file(
            config.bench_repo / "scripts" / "run_e5_v2_campaign.py"
        ),
        "aggregateScriptSha256": sha256_file(
            config.bench_repo / "scripts" / "aggregate_e5_v2.py"
        ),
        "dirty": dirty,
        "publicationEligible": not dirty and not config.query_ids,
        "parameters": {
            "denseName": config.dense_name,
            "sparseName": config.sparse_name,
            "limit": config.limit,
            "rrfK": config.rrf_k,
            "weights": list(config.weights),
            "producers": list(PRODUCERS),
            "referenceProducer": REFERENCE_PRODUCER,
            "rounds": config.rounds,
            "shardSize": shard_size,
            "warmups": config.warmups,
            "repetitions": config.repetitions,
            "requestTimeoutSeconds": config.request_timeout_seconds,
            "startupTimeoutSeconds": config.startup_timeout_seconds,
            "shardWallTimeoutSeconds": config.shard_wall_timeout_seconds,
            "cacheState": cache_state,
            "slotRetryMax": 1,
        },
        "rounds": round_specs,
        "createdAtUtc": datetime.now(UTC).isoformat(),
    }
    _write_json_with_checksum(config.output, campaign)
    return campaign


def _find_shard(campaign: dict[str, Any], round_number: int, shard_number: int) -> dict[str, Any]:
    for round_spec in campaign["rounds"]:
        if round_spec["round"] != round_number:
            continue
        for shard in round_spec["shards"]:
            if shard["shard"] == shard_number:
                return shard
    raise ValueError(f"E5-v2 shard not scheduled: round={round_number}, shard={shard_number}")


def _measurement(result: ExactRrfResult, latency_ns: int) -> dict[str, Any]:
    execution = result.execution
    return {
        "latencyNs": latency_ns,
        "plan": execution.get("plan"),
        "stopReason": execution.get("stopReason"),
        "sourcePulls": execution.get("sourcePulls"),
        "sourceExhausted": execution.get("sourceExhausted"),
        "certificationChecks": execution.get("certificationChecks"),
        "queryRounds": execution.get("queryRounds"),
        "sourcePointsMaterialized": execution.get("sourcePointsMaterialized"),
        "corpusPointsObserved": execution.get("corpusPointsObserved"),
        "exhaustiveFallback": execution.get("exhaustiveFallback"),
        "producer": execution.get("producer"),
    }


def _logical_signature(measurement: dict[str, Any]) -> dict[str, Any]:
    return {field: measurement.get(field) for field in LOGICAL_FIELDS}


def _invoke(
    client: QueryClient,
    producer: str,
    query: QueryInput,
    parameters: dict[str, Any],
) -> tuple[ExactRrfResult, int]:
    started = time.perf_counter_ns()
    result = client.producer_rrf(
        query,
        producer=producer,
        dense_name=parameters["denseName"],
        sparse_name=parameters["sparseName"],
        k=parameters["rrfK"],
        weights=tuple(parameters["weights"]),
        limit=parameters["limit"],
    )
    return result, time.perf_counter_ns() - started


def _validate_server(
    server: ManagedServerEvidence,
    client: QueryClient,
    campaign: dict[str, Any],
) -> dict[str, Any]:
    if server.binary_sha256 != campaign["binarySha256"]:
        raise RuntimeError("managed E5-v2 binary differs from the campaign")
    if server.snapshot_sha256 != campaign["collectionSnapshotSha256"]:
        raise RuntimeError("managed E5-v2 snapshot differs from the campaign")
    server_info = client.server_info()
    if server_info.get("commit") != campaign["systemCommit"]:
        raise RuntimeError("managed E5-v2 runtime commit mismatch")
    collection = client.collection_info()
    if collection["pointsCount"] != campaign["collectionPoints"]:
        raise RuntimeError("E5-v2 Collection point count mismatch")
    if canonical_hash(collection["config"]) != campaign["collectionConfigSha256"]:
        raise RuntimeError("E5-v2 Collection configuration mismatch")
    return collection


def _verify_frozen_inputs(config: E5V2ShardConfig, campaign: dict[str, Any]) -> tuple[
    DatasetSnapshot, CollectionSnapshot, str
]:
    dirty = git_is_dirty(config.bench_repo) or git_is_dirty(config.system_repo)
    if dirty and not config.allow_dirty:
        raise RuntimeError("publication E5-v2 refuses dirty repositories")
    if dirty != campaign["dirty"]:
        raise RuntimeError("E5-v2 repository cleanliness differs from the campaign")
    if git_revision(config.bench_repo) != campaign["benchCommit"]:
        raise RuntimeError("E5-v2 bench commit differs from the campaign")
    if git_revision(config.system_repo) != campaign["systemCommit"]:
        raise RuntimeError("E5-v2 system commit differs from the campaign")
    if runner_source_sha256(config.bench_repo) != campaign["runnerSourceSha256"]:
        raise RuntimeError("E5-v2 runner source differs from the campaign")
    if (
        sha256_file(config.bench_repo / "scripts" / "run_e5_v2_campaign.py")
        != campaign["campaignDriverSha256"]
        or sha256_file(config.bench_repo / "scripts" / "aggregate_e5_v2.py")
        != campaign["aggregateScriptSha256"]
    ):
        raise RuntimeError("E5-v2 campaign driver or aggregator differs from the campaign")
    binary_sha256 = sha256_file(config.system_binary)
    if binary_sha256 != campaign["binarySha256"]:
        raise RuntimeError("E5-v2 binary differs from the campaign")
    build_sha256 = verify_system_build_manifest(
        config.system_build_manifest,
        binary_sha256=binary_sha256,
        system_commit=campaign["systemCommit"],
    )
    if build_sha256 != campaign["systemBuildManifestSha256"]:
        raise RuntimeError("E5-v2 build manifest differs from the campaign")
    hardware_sha256 = verify_hardware_manifest(
        config.hardware_manifest,
        hardware_profile=campaign["hardwareProfile"],
    )
    if hardware_sha256 != campaign["hardwareManifestSha256"]:
        raise RuntimeError("E5-v2 hardware manifest differs from the campaign")
    snapshot = load_dataset_snapshot(config.artifact_root, campaign["dataset"])
    collection = load_collection_snapshot(config.artifact_root, snapshot, campaign["collection"])
    if (
        snapshot.source_manifest_sha256 != campaign["datasetManifestSha256"]
        or snapshot.dense_manifest_sha256 != campaign["denseManifestSha256"]
        or snapshot.sparse_manifest_sha256 != campaign["sparseManifestSha256"]
        or collection.manifest_sha256 != campaign["collectionSnapshotManifestSha256"]
        or collection.snapshot_sha256 != campaign["collectionSnapshotSha256"]
    ):
        raise RuntimeError("E5-v2 data or snapshot provenance differs from the campaign")
    return snapshot, collection, hardware_sha256


def _capture_machine_state(output_parent: Path, snapshot_path: Path) -> dict[str, Any]:
    def command_output(command: list[str]) -> str | None:
        if shutil.which(command[0]) is None:
            return None
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        output = (completed.stdout + completed.stderr).strip()
        return output[-4_000:] if output else None

    power = command_output(["pmset", "-g", "batt"])
    if platform.system() == "Darwin" and (power is None or "AC Power" not in power):
        raise RuntimeError("publication E5-v2 requires AC power")
    thermal = command_output(["pmset", "-g", "therm"])
    if platform.system() == "Darwin" and (
        thermal is None
        or "No thermal warning level has been recorded" not in thermal
        or "No performance warning level has been recorded" not in thermal
    ):
        raise RuntimeError("publication E5-v2 requires nominal thermal/performance state")
    memory_pressure = command_output(["memory_pressure", "-Q"])
    if platform.system() == "Darwin":
        match = re.search(r"memory free percentage:\s*(\d+)%", memory_pressure or "")
        if match is None or int(match.group(1)) < 20:
            raise RuntimeError("publication E5-v2 requires at least 20% free memory")
    qdrant_processes = command_output(["pgrep", "-x", "qdrant"])
    if qdrant_processes:
        raise RuntimeError("publication E5-v2 refuses another running Qdrant process")
    output_parent.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(output_parent)
    minimum_free = snapshot_path.stat().st_size * 2 + 10 * 1024**3
    if disk.free < minimum_free:
        raise RuntimeError("publication E5-v2 has insufficient temporary disk space")
    return {
        "capturedAtUtc": datetime.now(UTC).isoformat(),
        "power": power,
        "thermal": thermal,
        "memoryPressure": memory_pressure,
        "freeDiskBytes": disk.free,
        "minimumFreeDiskBytes": minimum_free,
        "snapshotBytes": snapshot_path.stat().st_size,
    }


def _query_record(
    client: QueryClient,
    query: QueryInput,
    query_spec: dict[str, Any],
    campaign: dict[str, Any],
    sequence: int,
    round_number: int,
    shard_number: int,
    shard_started: float,
) -> dict[str, Any]:
    parameters = campaign["parameters"]
    blocks: dict[str, list[dict[str, Any]]] = {}
    reference_ids: list[Any] | None = None
    reference_logical: dict[str, Any] | None = None
    for producer in query_spec["blockOrder"]:
        observations: list[dict[str, Any]] = []
        for observation in range(parameters["warmups"] + parameters["repetitions"]):
            if time.monotonic() - shard_started > parameters["shardWallTimeoutSeconds"]:
                raise TimeoutError("E5-v2 shard exceeded its wall-clock deadline")
            result, latency_ns = _invoke(client, producer, query, parameters)
            measurement = _measurement(result, latency_ns)
            ordered_ids = list(result.point_ids)
            if reference_ids is None:
                reference_ids = ordered_ids
                reference_logical = _logical_signature(measurement)
            if ordered_ids != reference_ids:
                raise RuntimeError(
                    f"E5-v2 ordered result mismatch for Query {query.query_id}: {producer}"
                )
            logical = _logical_signature(measurement)
            if logical != reference_logical:
                raise RuntimeError(
                    f"E5-v2 logical fusion work mismatch for Query {query.query_id}: {producer}"
                )
            observations.append(
                {
                    "warmup": observation < parameters["warmups"],
                    "repetition": (
                        observation + 1
                        if observation < parameters["warmups"]
                        else observation - parameters["warmups"] + 1
                    ),
                    "orderedIds": ordered_ids,
                    "orderedResultSha256": canonical_hash(ordered_ids),
                    **measurement,
                }
            )
        blocks[producer] = observations
    assert reference_ids is not None
    return {
        "recordType": "query",
        "runId": None,
        "queryId": query.query_id,
        "sequence": sequence,
        "round": round_number,
        "shard": shard_number,
        "blockOrder": query_spec["blockOrder"],
        "status": "ok",
        "orderedIds": reference_ids,
        "oracleOrderedIds": reference_ids,
        "orderedResultSha256": canonical_hash(reference_ids),
        "oracleOrderedResultSha256": canonical_hash(reference_ids),
        "membershipMismatch": False,
        "orderMismatch": False,
        "tieMismatch": False,
        "logicalSignature": reference_logical,
        "blocks": blocks,
    }


def run_e5_v2_shard(config: E5V2ShardConfig) -> dict[str, Any]:
    campaign = load_campaign(config.campaign_manifest)
    manifest_sha256 = _manifest_digest(config.campaign_manifest)
    if campaign.get("experiment") != EXPERIMENT:
        raise ValueError("campaign is not E5-v2")
    shard = _find_shard(campaign, config.round_number, config.shard_number)
    snapshot, collection, hardware_sha256 = _verify_frozen_inputs(config, campaign)
    by_id = {query.query_id: query for query in snapshot.queries}
    run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:12]}"
    failure_stem = f"r{config.round_number:02d}-s{config.shard_number:02d}-{run_id}"
    failed_path = config.failed_dir / f"{failure_stem}.failed.jsonl"
    failure_log_path = config.failed_dir / f"{failure_stem}.qdrant.log"
    parameters = campaign["parameters"]
    query_records: list[dict[str, Any]] = []
    shard_started = time.monotonic()
    runtime = runtime_metadata(campaign["hardwareProfile"])
    machine_state = _capture_machine_state(config.output.parent, collection.path)
    superseded_failures = _superseded_failures(
        config.failed_dir,
        campaign["campaignId"],
        config.round_number,
        config.shard_number,
    )
    run_record = {
        "recordType": "run",
        "schema": SCHEMA,
        "runId": run_id,
        "campaignId": campaign["campaignId"],
        "campaignManifestSha256": manifest_sha256,
        "experiment": EXPERIMENT,
        "method": campaign.get(
            "method", _protocol_labels(parameters["warmups"])[0]
        ),
        "dataset": campaign["dataset"],
        "split": campaign["split"],
        "round": config.round_number,
        "shard": config.shard_number,
        "queryIds": [query["queryId"] for query in shard["queries"]],
        "datasetManifestSha256": campaign["datasetManifestSha256"],
        "denseManifestSha256": campaign["denseManifestSha256"],
        "sparseManifestSha256": campaign["sparseManifestSha256"],
        "systemCommit": campaign["systemCommit"],
        "systemArtifact": campaign["systemArtifact"],
        "serverProvenance": {
            "mode": "managed-isolated-snapshot",
            "binarySha256": campaign["binarySha256"],
            "snapshotSha256": campaign["collectionSnapshotSha256"],
            "collectionSnapshotManifestSha256": campaign[
                "collectionSnapshotManifestSha256"
            ],
            "systemBuildManifestSha256": campaign["systemBuildManifestSha256"],
        },
        "hardwareManifestSha256": hardware_sha256,
        "benchCommit": campaign["benchCommit"],
        "dirty": campaign["dirty"],
        "runnerSourceSha256": campaign["runnerSourceSha256"],
        "supersedesFailedAttempts": superseded_failures,
        "buildProfile": "canonical-bench-release-v1",
        **runtime,
        "cacheState": parameters["cacheState"],
        "machineState": machine_state,
        "collection": campaign["collection"],
        "collectionConfigSha256": campaign["collectionConfigSha256"],
        "collectionPoints": campaign["collectionPoints"],
        "parameters": {
            **parameters,
            "round": config.round_number,
            "shard": config.shard_number,
            "queryLimit": None if campaign["publicationEligible"] else len(shard["queries"]),
        },
        "startedAtUtc": datetime.now(UTC).isoformat(),
    }
    with AtomicJsonlWriter(config.output) as writer:
        writer.write(run_record)
        try:
            with ManagedQdrant(
                binary=config.system_binary,
                system_repo=config.system_repo,
                collection=campaign["collection"],
                snapshot=collection.path,
                startup_timeout_seconds=parameters["startupTimeoutSeconds"],
                failure_log_path=failure_log_path,
            ) as server:
                with QueryClient(
                    server.url,
                    campaign["collection"],
                    timeout_seconds=parameters["requestTimeoutSeconds"],
                    slot_retry_max=parameters["slotRetryMax"],
                ) as client:
                    initial_collection = _validate_server(server, client, campaign)
                    for sequence, query_spec in enumerate(shard["queries"]):
                        record = _query_record(
                            client,
                            by_id[query_spec["queryId"]],
                            query_spec,
                            campaign,
                            sequence,
                            config.round_number,
                            config.shard_number,
                            shard_started,
                        )
                        record["runId"] = run_id
                        writer.write(record)
                        query_records.append(record)
                    final_collection = client.collection_info()
                    if (
                        final_collection["pointsCount"] != initial_collection["pointsCount"]
                        or canonical_hash(final_collection["config"])
                        != canonical_hash(initial_collection["config"])
                    ):
                        raise RuntimeError("E5-v2 Collection changed during the shard")
        except Exception as error:
            writer.write(
                {
                    "recordType": "failure",
                    "runId": run_id,
                    "round": config.round_number,
                    "shard": config.shard_number,
                    "completedQueries": len(query_records),
                    "errorType": type(error).__name__,
                    "error": str(error),
                    "failedAtUtc": datetime.now(UTC).isoformat(),
                }
            )
            writer.commit_as(failed_path)
            _sha256_output(failed_path)
            raise E5V2ShardRejected(
                f"E5-v2 shard rejected; evidence preserved at {failed_path}"
            ) from error
        summary = {
            "recordType": "summary",
            "runId": run_id,
            "attemptedQueries": len(query_records),
            "uniqueQueries": len(query_records),
            "warmupObservations": len(query_records) * len(PRODUCERS) * parameters["warmups"],
            "measuredObservations": (
                len(query_records) * len(PRODUCERS) * parameters["repetitions"]
            ),
            "okQueries": len(query_records),
            "mismatchQueries": 0,
            "timeoutQueries": 0,
            "errorQueries": 0,
            "finishedAtUtc": datetime.now(UTC).isoformat(),
            "queryRecordSha256": canonical_hash(
                [canonical_hash(record) for record in query_records]
            ),
        }
        writer.write(summary)
        writer.commit()
    _sha256_output(config.output)
    return summary
