#!/usr/bin/env python3
"""Aggregate a complete immutable E5-v2 campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from canonical_runner.e5_v2 import PRODUCERS, REFERENCE_PRODUCER, load_campaign
from canonical_runner.validation import validate_log

BOOTSTRAP_SEED = 20_260_730
BOOTSTRAP_SAMPLES = 10_000


def _geometric_mean(values: list[float]) -> float:
    data = np.asarray(values, dtype=np.float64)
    if len(data) == 0 or np.any(data <= 0):
        raise ValueError("geometric mean requires positive observations")
    return float(np.exp(np.mean(np.log(data))))


def _percentiles(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
    }


def _bootstrap(values: list[float], *, seed: int, samples: int) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.integers(0, len(data), size=len(data))
        estimates[index] = float(np.exp(np.mean(np.log(data[selected]))))
    return {
        "estimate": _geometric_mean(values),
        "ci95Low": float(np.percentile(estimates, 2.5)),
        "ci95High": float(np.percentile(estimates, 97.5)),
    }


def _scheduled_shards(campaign: dict[str, Any]) -> dict[tuple[int, int], list[str]]:
    return {
        (round_spec["round"], shard["shard"]): [
            query["queryId"] for query in shard["queries"]
        ]
        for round_spec in campaign["rounds"]
        for shard in round_spec["shards"]
    }


def aggregate(
    campaign_path: Path,
    paths: list[Path],
    *,
    seed: int,
    samples: int,
    require_clean: bool = True,
    failed_dir: Path | None = None,
) -> dict[str, Any]:
    campaign = load_campaign(campaign_path)
    if require_clean:
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != campaign.get(
            "aggregateScriptSha256"
        ):
            raise ValueError("E5-v2 aggregator differs from the frozen campaign")
        if failed_dir is None:
            raise ValueError("publication E5-v2 aggregation requires the failed-attempt directory")
    campaign_sha256 = hashlib.sha256(campaign_path.read_bytes()).hexdigest()
    expected_shards = _scheduled_shards(campaign)
    actual_shards: dict[tuple[int, int], Path] = {}
    raw_latencies: dict[str, list[float]] = {producer: [] for producer in PRODUCERS}
    round_values: dict[tuple[str, int, str], float] = {}
    counters: dict[str, dict[str, list[float]]] = {
        producer: defaultdict(list) for producer in PRODUCERS
    }
    run_ids: list[str] = []
    superseded_failures: dict[str, str] = {}

    for path in paths:
        validate_log(path, require_clean=require_clean)
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        run = records[0]
        if run.get("experiment") != "E5-v2":
            raise ValueError(f"not an E5-v2 log: {path}")
        if (
            run.get("campaignId") != campaign["campaignId"]
            or run.get("campaignManifestSha256") != campaign_sha256
        ):
            raise ValueError("E5-v2 log belongs to a different campaign")
        shard_key = (run["round"], run["shard"])
        if shard_key not in expected_shards:
            raise ValueError(f"unscheduled E5-v2 shard: {shard_key}")
        if shard_key in actual_shards:
            raise ValueError(f"duplicate eligible E5-v2 shard: {shard_key}")
        query_records = [record for record in records if record.get("recordType") == "query"]
        if [record["queryId"] for record in query_records] != expected_shards[shard_key]:
            raise ValueError(f"E5-v2 shard Query coverage mismatch: {shard_key}")
        actual_shards[shard_key] = path
        run_ids.append(run["runId"])
        for failure in run.get("supersedesFailedAttempts", []):
            if (
                not isinstance(failure, dict)
                or not isinstance(failure.get("file"), str)
                or not isinstance(failure.get("sha256"), str)
                or failure["file"] in superseded_failures
            ):
                raise ValueError("invalid or duplicate E5-v2 failed-attempt supersession")
            superseded_failures[failure["file"]] = failure["sha256"]
        for record in query_records:
            query_id = record["queryId"]
            round_number = record["round"]
            for producer in PRODUCERS:
                formal = [
                    observation
                    for observation in record["blocks"][producer]
                    if not observation["warmup"]
                ]
                latencies = [float(observation["latencyNs"]) for observation in formal]
                raw_latencies[producer].extend(latencies)
                key = (query_id, round_number, producer)
                if key in round_values:
                    raise ValueError(f"duplicate E5-v2 Query-round-plan: {key}")
                round_values[key] = _geometric_mean(latencies)
                for observation in formal:
                    for name, value in observation["producer"].items():
                        if isinstance(value, int | float) and not isinstance(value, bool):
                            counters[producer][name].append(float(value))

    missing = set(expected_shards) - set(actual_shards)
    if missing:
        raise ValueError(f"missing E5-v2 shards: {sorted(missing)}")
    if set(actual_shards) != set(expected_shards):
        raise ValueError("E5-v2 campaign shard matrix is not exact")
    if failed_dir is not None:
        observed_failures: dict[str, str] = {}
        for path in sorted(failed_dir.glob("*.failed.jsonl")):
            lines = path.read_text(encoding="utf-8").splitlines()
            if not lines:
                raise ValueError(f"empty E5-v2 failed artifact: {path}")
            first = json.loads(lines[0])
            if first.get("campaignId") != campaign["campaignId"]:
                continue
            checksum = path.with_suffix(path.suffix + ".sha256")
            if not checksum.is_file():
                raise ValueError(f"E5-v2 failed artifact checksum is missing: {path}")
            expected = checksum.read_text(encoding="utf-8").split()[0]
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if expected != actual:
                raise ValueError(f"E5-v2 failed artifact checksum mismatch: {path}")
            observed_failures[path.name] = actual
        if observed_failures != superseded_failures:
            raise ValueError("E5-v2 has unresolved or incorrectly superseded failed attempts")

    rounds = campaign["parameters"]["rounds"]
    query_plan: dict[tuple[str, str], float] = {}
    for query_id in campaign["queryIds"]:
        for producer in PRODUCERS:
            values = [
                round_values[(query_id, round_number, producer)]
                for round_number in range(1, rounds + 1)
            ]
            query_plan[(query_id, producer)] = _geometric_mean(values)

    paired: dict[str, Any] = {}
    for producer in PRODUCERS:
        if producer == REFERENCE_PRODUCER:
            continue
        query_ratios = [
            query_plan[(query_id, producer)] / query_plan[(query_id, REFERENCE_PRODUCER)]
            for query_id in campaign["queryIds"]
        ]
        round_ratios = {
            str(round_number): _geometric_mean(
                [
                    round_values[(query_id, round_number, producer)]
                    / round_values[(query_id, round_number, REFERENCE_PRODUCER)]
                    for query_id in campaign["queryIds"]
                ]
            )
            for round_number in range(1, rounds + 1)
        }
        query_round_wins = sum(
            round_values[(query_id, round_number, producer)]
            > round_values[(query_id, round_number, REFERENCE_PRODUCER)]
            for query_id in campaign["queryIds"]
            for round_number in range(1, rounds + 1)
        )
        paired[producer] = {
            "latencyRatio": _bootstrap(query_ratios, seed=seed, samples=samples),
            "queryWinRate": sum(ratio > 1 for ratio in query_ratios) / len(query_ratios),
            "queryRoundWinRate": query_round_wins / (len(query_ratios) * rounds),
            "roundRatios": round_ratios,
        }

    return {
        "schema": "ed-wrrf-e5-v2-aggregate-v1",
        "campaignId": campaign["campaignId"],
        "campaignManifestSha256": campaign_sha256,
        "dataset": campaign["dataset"],
        "runIds": sorted(run_ids),
        "referenceProducer": REFERENCE_PRODUCER,
        "queryCount": len(campaign["queryIds"]),
        "rounds": rounds,
        "latencyNs": {
            producer: _percentiles(values) for producer, values in raw_latencies.items()
        },
        "pairedAgainstPvsPbm": paired,
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
        "bootstrap": {"unit": "query", "seed": seed, "samples": samples},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--failed-dir", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(
        args.campaign_manifest,
        args.logs,
        seed=args.bootstrap_seed,
        samples=args.bootstrap_samples,
        require_clean=not args.allow_dirty,
        failed_dir=args.failed_dir,
    )
    if args.output.exists():
        raise FileExistsError(f"E5-v2 aggregate already exists: {args.output}")
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
