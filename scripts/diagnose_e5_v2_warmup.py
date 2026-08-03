#!/usr/bin/env python3
"""Diagnose E5-v2 warmup stability without producing publication evidence."""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import shutil
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from canonical_runner.artifacts import load_collection_snapshot, load_dataset_snapshot
from canonical_runner.client import QueryClient
from canonical_runner.e5_v2 import (
    _invoke,
    _logical_signature,
    _measurement,
    _validate_server,
    load_campaign,
)
from canonical_runner.provenance import canonical_hash
from canonical_runner.runner import _sha256_output
from canonical_runner.server import ManagedQdrant, sha256_file

SCHEMA = "ed-wrrf-e5-v2-warmup-diagnostic-v1"
PLANS = ("pvs-sparse-materialized", "pvs-pbm")
WORK_FIELDS = (
    "producer",
    "corpusPointsObserved",
    "sourcePointsMaterialized",
    "queryRounds",
    "exhaustiveFallback",
)


def _command_output(command: list[str]) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    output = (completed.stdout + completed.stderr).strip()
    return output[-8_000:] if output else None


def _parse_memory_free(output: str | None) -> int | None:
    match = re.search(r"memory free percentage:\s*(\d+)%", output or "")
    return int(match.group(1)) if match else None


def _parse_vm_stat(output: str | None) -> dict[str, int]:
    counters: dict[str, int] = {}
    for line in (output or "").splitlines():
        match = re.match(r"([^:]+):\s+([0-9.]+)\.?$", line.strip())
        if match:
            counters[match.group(1)] = int(float(match.group(2)))
    return counters


def _process_state(process_id: int) -> dict[str, Any] | None:
    output = _command_output(
        ["ps", "-o", "rss=,vsz=,%cpu=,state=,time=", "-p", str(process_id)]
    )
    if not output:
        return None
    fields = output.split(None, 4)
    if len(fields) != 5:
        return {"raw": output}
    return {
        "rssKiB": int(fields[0]),
        "vszKiB": int(fields[1]),
        "cpuPercent": float(fields[2]),
        "state": fields[3],
        "cpuTime": fields[4],
    }


def _telemetry(process_id: int) -> dict[str, Any]:
    memory_pressure = _command_output(["memory_pressure", "-Q"])
    thermal = _command_output(["pmset", "-g", "therm"])
    return {
        "capturedAtUtc": datetime.now(UTC).isoformat(),
        "memoryFreePercent": _parse_memory_free(memory_pressure),
        "memoryPressureRaw": memory_pressure,
        "vmStat": _parse_vm_stat(_command_output(["vm_stat"])),
        "swapUsage": _command_output(["sysctl", "vm.swapusage"]),
        "thermal": thermal,
        "qdrant": _process_state(process_id),
    }


def _geometric_mean(values: list[int]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _stability(block: list[dict[str, Any]], warmups: int) -> dict[str, Any]:
    formal = [row["latencyNs"] for row in block[warmups:]]
    formal_geomean = _geometric_mean(formal)
    last_warmup = block[warmups - 1]["latencyNs"]
    spread = max(formal) / min(formal)
    warmup_ratio = last_warmup / formal_geomean
    return {
        "formalGeomeanNs": formal_geomean,
        "formalSpread": spread,
        "lastWarmupToFormalGeomean": warmup_ratio,
        "withinStartPass": spread <= 2.0 and 0.5 <= warmup_ratio <= 2.0,
    }


def _work_signature(measurement: dict[str, Any]) -> dict[str, Any]:
    return {
        **_logical_signature(measurement),
        **{field: measurement.get(field) for field in WORK_FIELDS},
    }


def _write_output(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"diagnostic output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _sha256_output(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("--system-repo", type=Path, required=True)
    parser.add_argument("--system-binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--query-id", default="1105792")
    parser.add_argument("--process-starts", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=4)
    parser.add_argument("--measurements", type=int, default=4)
    args = parser.parse_args()
    if args.process_starts <= 0 or args.warmups <= 0 or args.measurements <= 0:
        raise ValueError("process starts, warmups and measurements must be positive")

    campaign = load_campaign(args.campaign_manifest)
    if campaign.get("publicationEligible"):
        raise RuntimeError("warmup diagnosis requires a non-publication campaign")
    if sha256_file(args.system_binary) != campaign["binarySha256"]:
        raise RuntimeError("diagnostic binary differs from the dry-run campaign")
    snapshot = load_dataset_snapshot(args.artifact_root, campaign["dataset"])
    collection = load_collection_snapshot(args.artifact_root, snapshot, campaign["collection"])
    by_id = {query.query_id: query for query in snapshot.queries}
    query = by_id.get(args.query_id)
    if query is None:
        raise ValueError(f"Query is absent from the campaign dataset: {args.query_id}")

    args.log_dir.mkdir(parents=True, exist_ok=True)
    all_starts: list[dict[str, Any]] = []
    reference_ids: list[Any] | None = None
    signatures: dict[str, dict[str, Any]] = {}
    for process_start in range(1, args.process_starts + 1):
        manager = ManagedQdrant(
            binary=args.system_binary,
            system_repo=args.system_repo,
            collection=campaign["collection"],
            snapshot=collection.path,
            startup_timeout_seconds=campaign["parameters"]["startupTimeoutSeconds"],
            failure_log_path=args.log_dir / f"process-{process_start:02d}.failed.log",
        )
        start_record: dict[str, Any] = {
            "processStart": process_start,
            "startedAtUtc": datetime.now(UTC).isoformat(),
            "blocks": {},
        }
        try:
            with manager as server:
                with QueryClient(
                    server.url,
                    campaign["collection"],
                    timeout_seconds=campaign["parameters"]["requestTimeoutSeconds"],
                    slot_retry_max=1,
                ) as client:
                    _validate_server(server, client, campaign)
                    for plan in PLANS:
                        block: list[dict[str, Any]] = []
                        count = args.warmups + args.measurements
                        for index in range(count):
                            before = _telemetry(server.process_id)
                            result, latency_ns = _invoke(
                                client, plan, query, campaign["parameters"]
                            )
                            after = _telemetry(server.process_id)
                            measurement = _measurement(result, latency_ns)
                            ordered_ids = list(result.point_ids)
                            if reference_ids is None:
                                reference_ids = ordered_ids
                            if ordered_ids != reference_ids:
                                raise RuntimeError(
                                    f"ordered result mismatch in {plan}, process {process_start}"
                                )
                            signature = _work_signature(measurement)
                            if plan in signatures and signature != signatures[plan]:
                                raise RuntimeError(
                                    f"work signature changed in {plan}, process {process_start}"
                                )
                            signatures.setdefault(plan, signature)
                            block.append(
                                {
                                    "phase": "warmup" if index < args.warmups else "diagnostic",
                                    "repetition": (
                                        index + 1
                                        if index < args.warmups
                                        else index - args.warmups + 1
                                    ),
                                    "latencyNs": latency_ns,
                                    "orderedResultSha256": canonical_hash(ordered_ids),
                                    "workSignature": signature,
                                    "measurement": measurement,
                                    "before": before,
                                    "after": after,
                                }
                            )
                        start_record["blocks"][plan] = {
                            "observations": block,
                            "stability": _stability(block, args.warmups),
                        }
                    if manager._log is not None:  # Diagnostic-only log retention.
                        manager._log.flush()
                        shutil.copyfile(
                            manager._log.name,
                            args.log_dir / f"process-{process_start:02d}.success.log",
                        )
        finally:
            start_record["finishedAtUtc"] = datetime.now(UTC).isoformat()
        all_starts.append(start_record)

    cross_start: dict[str, Any] = {}
    for plan in PLANS:
        means = [
            start["blocks"][plan]["stability"]["formalGeomeanNs"]
            for start in all_starts
        ]
        ratio = max(means) / min(means)
        cross_start[plan] = {
            "formalGeomeansNs": means,
            "maxToMinRatio": ratio,
            "pass": ratio <= 2.0,
        }
    within_pass = all(
        start["blocks"][plan]["stability"]["withinStartPass"]
        for start in all_starts
        for plan in PLANS
    )
    overall_pass = within_pass and all(value["pass"] for value in cross_start.values())
    payload = {
        "schema": SCHEMA,
        "publicationEligible": False,
        "createdAtUtc": datetime.now(UTC).isoformat(),
        "campaignId": campaign["campaignId"],
        "campaignManifestSha256": sha256_file(args.campaign_manifest),
        "binarySha256": campaign["binarySha256"],
        "snapshotSha256": campaign["collectionSnapshotSha256"],
        "machine": {"architecture": platform.machine(), "system": platform.system()},
        "queryId": args.query_id,
        "plans": list(PLANS),
        "processStarts": args.process_starts,
        "warmups": args.warmups,
        "measurements": args.measurements,
        "acceptance": {
            "withinStartFormalSpreadMax": 2.0,
            "lastWarmupToFormalGeomeanRange": [0.5, 2.0],
            "crossStartGeomeanSpreadMax": 2.0,
        },
        "workSignatures": signatures,
        "starts": all_starts,
        "crossStart": cross_start,
        "diagnosticPass": overall_pass,
    }
    _write_output(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "diagnosticPass": overall_pass,
                "crossStart": cross_start,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
