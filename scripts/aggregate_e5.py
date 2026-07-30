#!/usr/bin/env python3
"""Aggregate immutable paired E5 producer-ablation logs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from canonical_runner.e5 import PRODUCERS, REFERENCE_PRODUCER
from canonical_runner.validation import validate_log

BOOTSTRAP_SEED = 20_260_730
BOOTSTRAP_SAMPLES = 10_000


def _percentiles(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
    }


def _geometric_mean(values: np.ndarray) -> float:
    return float(np.exp(np.mean(np.log(values))))


def _clustered_ratio(
    ratios_by_query: dict[str, list[float]], *, seed: int, samples: int
) -> dict[str, float]:
    clustered = np.asarray(
        [
            _geometric_mean(np.asarray(ratios, dtype=np.float64))
            for _, ratios in sorted(ratios_by_query.items())
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.integers(0, len(clustered), size=len(clustered))
        bootstrap[index] = _geometric_mean(clustered[selected])
    return {
        "estimate": _geometric_mean(clustered),
        "ci95Low": float(np.percentile(bootstrap, 2.5)),
        "ci95High": float(np.percentile(bootstrap, 97.5)),
    }


def aggregate(paths: list[Path], *, seed: int, samples: int) -> dict[str, Any]:
    latencies: dict[str, list[float]] = {producer: [] for producer in PRODUCERS}
    ratios: dict[str, dict[str, list[float]]] = {
        producer: defaultdict(list) for producer in PRODUCERS if producer != REFERENCE_PRODUCER
    }
    wins = {producer: 0 for producer in ratios}
    counters: dict[str, dict[str, list[float]]] = {
        producer: defaultdict(list) for producer in PRODUCERS
    }
    run_ids: list[str] = []
    dataset: str | None = None
    observations = 0

    for path in paths:
        validate_log(path)
        with path.open(encoding="utf-8") as handle:
            run = json.loads(next(handle))
            if run.get("experiment") != "E5":
                raise ValueError(f"not an E5 log: {path}")
            if dataset is None:
                dataset = run["dataset"]
            elif dataset != run["dataset"]:
                raise ValueError("one E5 aggregate may contain only one dataset")
            run_ids.append(run["runId"])
            for line in handle:
                record = json.loads(line)
                if record.get("recordType") != "query":
                    continue
                if record.get("status") != "ok":
                    raise ValueError(
                        "publication E5 aggregation refuses non-ok observations: "
                        f"{record.get('queryId')}={record.get('status')}"
                    )
                if record.get("warmup"):
                    continue
                measurements = record["measurements"]
                reference_latency = float(measurements[REFERENCE_PRODUCER]["latencyNs"])
                if reference_latency <= 0:
                    raise ValueError("E5 latency must be positive")
                observations += 1
                for producer in PRODUCERS:
                    measurement = measurements[producer]
                    latency = float(measurement["latencyNs"])
                    if latency <= 0:
                        raise ValueError("E5 latency must be positive")
                    latencies[producer].append(latency)
                    for name, value in measurement["producer"].items():
                        if isinstance(value, int | float) and not isinstance(value, bool):
                            counters[producer][name].append(float(value))
                    if producer != REFERENCE_PRODUCER:
                        ratio = latency / reference_latency
                        ratios[producer][record["queryId"]].append(ratio)
                        wins[producer] += latency < reference_latency

    if not observations:
        raise ValueError("no measured successful E5 observations")
    return {
        "schema": "ed-wrrf-e5-aggregate-v1",
        "dataset": dataset,
        "runIds": sorted(run_ids),
        "referenceProducer": REFERENCE_PRODUCER,
        "measuredObservations": observations,
        "latencyNs": {producer: _percentiles(values) for producer, values in latencies.items()},
        "pairedAgainstPvsPbm": {
            producer: {
                "latencyRatio": _clustered_ratio(ratios[producer], seed=seed, samples=samples),
                "winRate": wins[producer] / observations,
            }
            for producer in ratios
        },
        "physicalWork": {
            producer: {
                name: {
                    "median": float(np.median(values)),
                    "mean": float(np.mean(values)),
                }
                for name, values in sorted(producer_counters.items())
            }
            for producer, producer_counters in counters.items()
        },
        "bootstrap": {
            "unit": "query-cluster",
            "seed": seed,
            "samples": samples,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    args = parser.parse_args()
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
    checksum = args.output.with_suffix(args.output.suffix + ".sha256")
    with checksum.open("x", encoding="utf-8") as handle:
        handle.write(f"{digest}  {args.output.name}\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
