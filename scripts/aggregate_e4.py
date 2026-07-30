#!/usr/bin/env python3
"""Aggregate one immutable E4 controlled-rank run."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from canonical_runner.validation import validate_log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def aggregate(path: Path) -> dict[str, Any]:
    validate_log(path)
    with path.open(encoding="utf-8") as handle:
        run = json.loads(next(handle))
        if run.get("experiment") != "E4":
            raise ValueError(f"not an E4 log: {path}")
        grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        for line in handle:
            record = json.loads(line)
            if record.get("recordType") != "query":
                continue
            if record.get("status") != "ok":
                raise ValueError(
                    "publication E4 aggregation requires zero mismatch/error: "
                    f"{record.get('queryId')}={record.get('status')}"
                )
            grouped[(int(record["size"]), str(record["regime"]))].append(record)

    rows: list[dict[str, Any]] = []
    for size in run["parameters"]["sizes"]:
        for regime in run["parameters"]["regimes"]:
            records = grouped[(size, regime)]
            if len(records) != len(run["parameters"]["seeds"]):
                raise ValueError(f"incomplete E4 seed matrix: size={size}, regime={regime}")
            ratios = [float(record["workRatio"]) for record in records]
            depths = [int(record["certificateDepth"]) for record in records]
            rows.append(
                {
                    "size": size,
                    "regime": regime,
                    "seeds": sorted(int(record["seed"]) for record in records),
                    "workRatio": {
                        "mean": mean(ratios),
                        "median": median(ratios),
                        "min": min(ratios),
                        "max": max(ratios),
                    },
                    "certificateDepth": {
                        "median": median(depths),
                        "min": min(depths),
                        "max": max(depths),
                    },
                    "topWindowOverlap": mean(
                        float(record["topWindowOverlap"]) for record in records
                    ),
                    "exhaustiveCases": sum(ratio == 1.0 for ratio in ratios),
                }
            )

    return {
        "schema": "ed-wrrf-e4-aggregate-v1",
        "runId": run["runId"],
        "cases": sum(len(records) for records in grouped.values()),
        "parameters": run["parameters"],
        "rows": rows,
        "provenance": {
            field: run.get(field)
            for field in (
                "benchCommit",
                "runnerSourceSha256",
                "buildProfile",
                "hardwareProfile",
                "architecture",
                "os",
                "environmentAllowlist",
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
