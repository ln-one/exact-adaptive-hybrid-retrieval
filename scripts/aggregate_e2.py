#!/usr/bin/env python3
"""Aggregate immutable paired E2 logs without modifying raw evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from canonical_runner.validation import validate_log

BOOTSTRAP_SEED = 20_260_730
BOOTSTRAP_SAMPLES = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    return parser.parse_args()


def percentile_summary(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
    }


def geometric_mean(values: np.ndarray) -> float:
    return float(np.exp(np.mean(np.log(values))))


def clustered_ratio_interval(
    ratios_by_query: dict[str, list[float]], *, seed: int, samples: int
) -> dict[str, float]:
    query_ratios = np.asarray(
        [
            geometric_mean(np.asarray(ratios, dtype=np.float64))
            for _, ratios in sorted(ratios_by_query.items())
        ],
        dtype=np.float64,
    )
    estimate = geometric_mean(query_ratios)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.integers(0, len(query_ratios), size=len(query_ratios))
        bootstrap[index] = geometric_mean(query_ratios[selected])
    return {
        "estimate": estimate,
        "ci95Low": float(np.percentile(bootstrap, 2.5)),
        "ci95High": float(np.percentile(bootstrap, 97.5)),
    }


def aggregate(paths: list[Path], *, seed: int, samples: int) -> dict[str, Any]:
    dynamic_latencies: list[float] = []
    exhaustive_latencies: list[float] = []
    ratios_by_query: dict[str, list[float]] = defaultdict(list)
    pull_ratios: list[list[float]] = [[], []]
    run_ids: list[str] = []
    per_run: dict[str, dict[str, Any]] = {}
    comparison_signature: str | None = None
    dataset: str | None = None
    wins = observations = 0

    for path in paths:
        validate_log(path)
        with path.open(encoding="utf-8") as handle:
            run = json.loads(next(handle))
            if run.get("experiment") != "E2":
                raise ValueError(f"not an E2 log: {path}")
            if dataset is None:
                dataset = run["dataset"]
            elif dataset != run["dataset"]:
                raise ValueError("one E2 aggregate may contain only one dataset")
            signature = json.dumps(
                {
                    field: run.get(field)
                    for field in (
                        "datasetManifestSha256",
                        "denseManifestSha256",
                        "sparseManifestSha256",
                        "systemCommit",
                        "systemArtifact",
                        "benchCommit",
                        "runnerSourceSha256",
                        "buildProfile",
                        "hardwareProfile",
                        "architecture",
                        "os",
                        "cacheState",
                        "collectionConfigSha256",
                        "parameters",
                        "serverProvenance",
                    )
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            if comparison_signature is None:
                comparison_signature = signature
            elif comparison_signature != signature:
                raise ValueError("E2 logs have incompatible provenance or experiment parameters")
            run_id = run["runId"]
            run_ids.append(run_id)
            run_data = {
                "dynamic": [],
                "exhaustive": [],
                "ratios": [],
                "wins": 0,
            }
            per_run[run_id] = run_data
            for line in handle:
                record = json.loads(line)
                if record.get("recordType") != "query":
                    continue
                if record.get("status") != "ok":
                    raise ValueError(
                        "publication E2 aggregation refuses mismatch, timeout, cancellation, "
                        f"or error records: {record.get('queryId')}={record.get('status')}"
                    )
                if record.get("warmup"):
                    continue
                dynamic_ns = float(record["dynamic"]["latencyNs"])
                exhaustive_ns = float(record["exhaustive"]["latencyNs"])
                if dynamic_ns <= 0 or exhaustive_ns <= 0:
                    raise ValueError("E2 latency must be positive")
                ratio = dynamic_ns / exhaustive_ns
                dynamic_latencies.append(dynamic_ns)
                exhaustive_latencies.append(exhaustive_ns)
                ratios_by_query[record["queryId"]].append(ratio)
                wins += dynamic_ns < exhaustive_ns
                observations += 1
                run_data["dynamic"].append(dynamic_ns)
                run_data["exhaustive"].append(exhaustive_ns)
                run_data["ratios"].append(ratio)
                run_data["wins"] += dynamic_ns < exhaustive_ns
                for channel, value in enumerate(record["sourcePullRatios"]):
                    if value is not None:
                        pull_ratios[channel].append(float(value))

    if not observations:
        raise ValueError("no measured successful E2 observations")
    if any(not math.isfinite(value) or value <= 0 for value in dynamic_latencies):
        raise ValueError("invalid E2 dynamic latency")

    return {
        "schema": "ed-wrrf-e2-aggregate-v2",
        "dataset": dataset,
        "runIds": sorted(run_ids),
        "measuredQueries": len(ratios_by_query),
        "measuredObservations": observations,
        "latencyNs": {
            "edWrrf": percentile_summary(dynamic_latencies),
            "exhaustive": percentile_summary(exhaustive_latencies),
        },
        "pairedLatencyRatio": clustered_ratio_interval(ratios_by_query, seed=seed, samples=samples),
        "pairedWinRate": wins / observations,
        "perRun": [
            {
                "runId": run_id,
                "measuredObservations": len(data["ratios"]),
                "latencyNs": {
                    "edWrrf": percentile_summary(data["dynamic"]),
                    "exhaustive": percentile_summary(data["exhaustive"]),
                },
                "pairedLatencyRatio": geometric_mean(np.asarray(data["ratios"], dtype=np.float64)),
                "pairedWinRate": data["wins"] / len(data["ratios"]),
            }
            for run_id, data in sorted(per_run.items())
        ],
        "sourcePullRatio": {
            "denseMedian": float(np.median(pull_ratios[0])),
            "sparseMedian": float(np.median(pull_ratios[1])),
        },
        "bootstrap": {
            "unit": "query-cluster",
            "seed": seed,
            "samples": samples,
        },
    }


def main() -> None:
    args = parse_args()
    result = aggregate(
        args.logs,
        seed=args.bootstrap_seed,
        samples=args.bootstrap_samples,
    )
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
