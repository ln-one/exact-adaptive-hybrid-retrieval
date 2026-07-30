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
