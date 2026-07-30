"""Structural and integrity checks for canonical JSONL logs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .provenance import canonical_hash


def _read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at line {line_number}: {error}") from error
        if not isinstance(record, dict):
            raise ValueError(f"line {line_number} is not a JSON object")
        records.append(record)
    return records


def validate_log(path: Path, *, require_clean: bool = True) -> dict[str, int]:
    records = _read_records(path)
    if len(records) < 2 or records[0].get("recordType") != "run":
        raise ValueError("canonical log must begin with one run record")
    if records[-1].get("recordType") != "summary":
        raise ValueError("canonical log must end with one summary record")
    if any(record.get("recordType") != "query" for record in records[1:-1]):
        raise ValueError("only query records may appear between run and summary")

    run = records[0]
    summary = records[-1]
    run_id = run.get("runId")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("runId must be a non-empty string")
    if run.get("schema") != "ed-wrrf-results-v1":
        raise ValueError("unsupported canonical log schema")
    if require_clean and run.get("dirty") is not False:
        raise ValueError("publication log was produced from a dirty repository")
    if require_clean and run.get("parameters", {}).get("queryLimit") is not None:
        raise ValueError("query-limited development log is not publication evidence")
    if require_clean and run.get("experiment") in {"E2", "E3", "E5"}:
        provenance = run.get("serverProvenance")
        if not isinstance(provenance, dict) or provenance.get("mode") != (
            "managed-isolated-snapshot"
        ):
            raise ValueError("publication E2/E3/E5 requires a managed binary and isolated snapshot")
        binary_sha256 = provenance.get("binarySha256")
        if run.get("systemArtifact") != f"sha256:{binary_sha256}":
            raise ValueError("system artifact is not bound to the managed binary")
        for field in (
            "binarySha256",
            "snapshotSha256",
            "collectionSnapshotManifestSha256",
            "systemBuildManifestSha256",
        ):
            value = provenance.get(field)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"managed provenance is missing a SHA-256 field: {field}")
    if any(record.get("runId") != run_id for record in records):
        raise ValueError("record runId mismatch")

    queries = records[1:-1]
    sequences = [record.get("sequence") for record in queries]
    if sequences != list(range(len(queries))):
        raise ValueError("query sequence must be contiguous and zero-based")
    query_ids = [record.get("queryId") for record in queries]
    if any(not isinstance(query_id, str) or not query_id for query_id in query_ids):
        raise ValueError("queryId must be a non-empty string")
    if run.get("experiment") == "E2":
        observation_ids = [
            (record.get("queryId"), record.get("repetition"), record.get("warmup"))
            for record in queries
        ]
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("duplicate E2 query observation in canonical log")
        for record in queries:
            if not isinstance(record.get("warmup"), bool):
                raise ValueError("E2 query observation is missing a warmup flag")
            if not isinstance(record.get("repetition"), int):
                raise ValueError("E2 query observation is missing a repetition")
            if record.get("status") not in {"ok", "mismatch"}:
                continue
            dynamic = record.get("dynamic")
            exhaustive = record.get("exhaustive")
            if not isinstance(dynamic, dict) or not isinstance(exhaustive, dict):
                raise ValueError("E2 observation is missing paired measurements")
            if exhaustive.get("stopReason") != "all-sources-exhausted":
                raise ValueError("E2 baseline did not report exhaustive termination")
            if exhaustive.get("sourceExhausted") != [True, True]:
                raise ValueError("E2 baseline did not exhaust both channel streams")
    elif run.get("experiment") == "E3":
        depths = run.get("parameters", {}).get("depths")
        if (
            not isinstance(depths, list)
            or not depths
            or any(not isinstance(depth, int) or depth <= 0 for depth in depths)
            or depths != sorted(set(depths))
        ):
            raise ValueError("E3 run is missing unique increasing fixed-prefix depths")
        observation_ids = [(record.get("queryId"), record.get("depth")) for record in queries]
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("duplicate E3 query/depth observation in canonical log")
        if any(record.get("depth") not in depths for record in queries):
            raise ValueError("E3 query observation uses an undeclared depth")
        expected_observations = summary.get("uniqueQueries", 0) * len(depths)
        if len(queries) != expected_observations:
            raise ValueError("E3 log does not contain a complete query/depth matrix")
    elif run.get("experiment") == "E4":
        parameters = run.get("parameters", {})
        sizes = parameters.get("sizes")
        seeds = parameters.get("seeds")
        regimes = parameters.get("regimes")
        if (
            not isinstance(sizes, list)
            or not isinstance(seeds, list)
            or not isinstance(regimes, list)
            or not sizes
            or not seeds
            or not regimes
        ):
            raise ValueError("E4 run is missing its size/seed/regime matrix")
        expected_cases = {
            (size, regime, seed) for size in sizes for regime in regimes for seed in seeds
        }
        actual_cases = {
            (record.get("size"), record.get("regime"), record.get("seed")) for record in queries
        }
        if (
            actual_cases != expected_cases
            or len(queries) != len(expected_cases)
            or len(query_ids) != len(set(query_ids))
        ):
            raise ValueError("E4 log does not contain the declared case matrix")
    elif run.get("experiment") == "E5":
        parameters = run.get("parameters", {})
        producers = parameters.get("producers")
        process_starts = parameters.get("processStarts")
        repetitions = parameters.get("repetitions")
        warmups = parameters.get("warmups")
        if (
            not isinstance(producers, list)
            or not producers
            or len(producers) != len(set(producers))
            or not isinstance(process_starts, int)
            or process_starts <= 0
            or not isinstance(repetitions, int)
            or repetitions <= 0
            or not isinstance(warmups, int)
            or warmups < 0
        ):
            raise ValueError("E5 run is missing its producer/observation matrix")
        observation_ids = [
            (
                record.get("queryId"),
                record.get("processStart"),
                record.get("repetition"),
                record.get("warmup"),
            )
            for record in queries
        ]
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("duplicate E5 query observation in canonical log")
        expected = summary.get("uniqueQueries", 0) * process_starts * (warmups + repetitions)
        if len(queries) != expected:
            raise ValueError("E5 log does not contain the declared observation matrix")
        for record in queries:
            if record.get("status") not in {"ok", "mismatch"}:
                continue
            measurements = record.get("measurements")
            outputs = record.get("orderedIdsByProducer")
            order = record.get("counterbalanceOrder")
            if (
                not isinstance(measurements, dict)
                or set(measurements) != set(producers)
                or not isinstance(outputs, dict)
                or set(outputs) != set(producers)
                or not isinstance(order, list)
                or set(order) != set(producers)
                or len(order) != len(producers)
            ):
                raise ValueError("E5 observation is missing a complete producer matrix")
            reference = outputs.get(parameters.get("referenceProducer"))
            mismatch_variants = [
                producer for producer in producers if outputs[producer] != reference
            ]
            if record.get("mismatchVariants") != mismatch_variants:
                raise ValueError("E5 mismatch producer list is inconsistent")
            if (record.get("status") == "mismatch") is not bool(mismatch_variants):
                raise ValueError("E5 status is inconsistent with producer outputs")
            for producer in producers:
                measurement = measurements[producer]
                if (
                    not isinstance(measurement, dict)
                    or not isinstance(measurement.get("latencyNs"), int)
                    or measurement["latencyNs"] <= 0
                    or not isinstance(measurement.get("producer"), dict)
                ):
                    raise ValueError("E5 producer measurement is incomplete")
    elif len(query_ids) != len(set(query_ids)):
        raise ValueError("duplicate queryId in canonical log")

    statuses = [record.get("status") for record in queries]
    allowed_statuses = {"ok", "mismatch", "timeout", "cancelled", "error"}
    if any(status not in allowed_statuses for status in statuses):
        raise ValueError("unknown query status")
    expected = {
        "attemptedQueries": len(queries),
        "okQueries": statuses.count("ok"),
        "mismatchQueries": statuses.count("mismatch"),
        "timeoutQueries": statuses.count("timeout"),
        "errorQueries": statuses.count("error"),
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            raise ValueError(f"summary {field} mismatch: {summary.get(field)} != {value}")

    for record in queries:
        if record["status"] not in {"ok", "mismatch"}:
            continue
        ordered = record.get("orderedIds")
        oracle = record.get("oracleOrderedIds")
        if not isinstance(ordered, list) or not isinstance(oracle, list):
            raise ValueError("successful query records must retain both ordered identity lists")
        if len(ordered) != len(set(map(str, ordered))):
            raise ValueError("duplicate identity in ordered result")
        if record.get("orderedResultSha256") != canonical_hash(ordered):
            raise ValueError("ordered result hash mismatch")
        if record.get("oracleOrderedResultSha256") != canonical_hash(oracle):
            raise ValueError("oracle ordered result hash mismatch")
        membership_mismatch = set(map(str, ordered)) != set(map(str, oracle))
        order_mismatch = not membership_mismatch and ordered != oracle
        if record.get("membershipMismatch") is not membership_mismatch:
            raise ValueError("membership mismatch flag is inconsistent")
        if record.get("orderMismatch") is not order_mismatch:
            raise ValueError("order mismatch flag is inconsistent")

    query_record_hash = canonical_hash([canonical_hash(record) for record in queries])
    if summary.get("queryRecordSha256") != query_record_hash:
        raise ValueError("summary query record hash mismatch")

    checksum_path = path.with_suffix(path.suffix + ".sha256")
    if not checksum_path.is_file():
        raise ValueError("canonical log checksum file is missing")
    expected_digest = checksum_path.read_text(encoding="utf-8").split()[0]
    actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_digest != actual_digest:
        raise ValueError("canonical log checksum mismatch")
    return expected
