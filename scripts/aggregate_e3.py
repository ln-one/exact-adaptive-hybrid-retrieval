#!/usr/bin/env python3
"""Aggregate one immutable E3 exact fixed-prefix frontier."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from canonical_runner.validation import validate_log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def wilson_interval(successes: int, observations: int) -> dict[str, float]:
    if observations <= 0:
        raise ValueError("Wilson interval requires at least one observation")
    z = 1.959963984540054
    rate = successes / observations
    denominator = 1 + z * z / observations
    center = (rate + z * z / (2 * observations)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1 - rate) / observations + z * z / (4 * observations**2))
        / denominator
    )
    return {
        "estimate": rate,
        "ci95Low": max(0.0, center - radius),
        "ci95High": min(1.0, center + radius),
    }


def aggregate(path: Path) -> dict[str, Any]:
    validate_log(path)
    with path.open(encoding="utf-8") as handle:
        run = json.loads(next(handle))
        if run.get("experiment") != "E3":
            raise ValueError(f"not an E3 log: {path}")
        records_by_depth: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for line in handle:
            record = json.loads(line)
            if record.get("recordType") != "query":
                continue
            if record.get("status") not in {"ok", "mismatch"}:
                raise ValueError(
                    "publication E3 aggregation refuses timeout, cancellation, or error: "
                    f"{record.get('queryId')}={record.get('status')}"
                )
            records_by_depth[int(record["depth"])].append(record)

    frontier: list[dict[str, Any]] = []
    for depth in run["parameters"]["depths"]:
        records = records_by_depth[depth]
        if not records:
            raise ValueError(f"E3 depth has no observations: {depth}")
        observations = len(records)
        ordered_exact = sum(record["status"] == "ok" for record in records)
        membership_exact = sum(not record["membershipMismatch"] for record in records)
        contains_oracle = sum(record["candidateUnionContainsOracle"] for record in records)
        frontier.append(
            {
                "depth": depth,
                "queries": observations,
                "orderedExact": wilson_interval(ordered_exact, observations),
                "membershipExactRate": membership_exact / observations,
                "candidateUnionContainsOracleRate": contains_oracle / observations,
                "meanOracleRecall": sum(record["oracleRecall"] for record in records)
                / observations,
                "medianExactPrefixLength": float(
                    median(record["exactPrefixLength"] for record in records)
                ),
                "medianCandidateUnionSize": float(
                    median(record["candidateUnionSize"] for record in records)
                ),
                "medianExposedRanks": float(median(record["exposedRanks"] for record in records)),
                "failureModes": {
                    "candidateMissing": sum(
                        record["status"] == "mismatch"
                        and not record["candidateUnionContainsOracle"]
                        for record in records
                    ),
                    "candidatePresentButWrong": sum(
                        record["status"] == "mismatch" and record["candidateUnionContainsOracle"]
                        for record in records
                    ),
                    "orderOnly": sum(record["orderMismatch"] for record in records),
                },
            }
        )

    return {
        "schema": "ed-wrrf-e3-aggregate-v1",
        "dataset": run["dataset"],
        "runId": run["runId"],
        "queries": len(
            {record["queryId"] for records in records_by_depth.values() for record in records}
        ),
        "limit": run["parameters"]["limit"],
        "rrfK": run["parameters"]["rrfK"],
        "weights": run["parameters"]["weights"],
        "frontier": frontier,
        "provenance": {
            field: run.get(field)
            for field in (
                "datasetManifestSha256",
                "denseManifestSha256",
                "sparseManifestSha256",
                "systemCommit",
                "systemArtifact",
                "benchCommit",
                "runnerSourceSha256",
                "hardwareProfile",
                "collectionConfigSha256",
                "serverProvenance",
            )
        },
    }


def main() -> None:
    args = parse_args()
    result = aggregate(args.log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    checksum_path = args.output.with_suffix(args.output.suffix + ".sha256")
    with checksum_path.open("x", encoding="utf-8") as handle:
        handle.write(f"{digest}  {args.output.name}\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
