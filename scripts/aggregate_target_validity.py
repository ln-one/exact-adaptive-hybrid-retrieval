#!/usr/bin/env python3
"""Aggregate target-validity retrieval records with standard IR measures."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from canonical_runner.provenance import canonical_hash
from canonical_runner.server import sha256_file
from ir_measures import RR, Qrel, Recall, ScoredDoc, iter_calc, nDCG

MEASURES = (nDCG @ 10, RR(rel=1) @ 10, Recall(rel=1) @ 100)
MEASURE_LABEL = {"nDCG@10": "nDCG@10", "RR@10": "MRR@10", "R@100": "Recall@100"}
FIXED_PREFIX = "fixed-L"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260730


@dataclass(frozen=True)
class RunData:
    run_dir: Path
    dataset: str
    query_ids: tuple[str, ...]
    records: dict[str, dict[str, Any]]
    qrels: tuple[Qrel, ...]


def write_json_exclusive(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"immutable aggregate already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.link(temporary, path)
    temporary.unlink()


def read_qrels(path: Path) -> tuple[Qrel, ...]:
    rows: list[Qrel] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["query_id", "doc_id", "relevance"]:
            raise RuntimeError(f"unexpected qrels header: {reader.fieldnames!r}")
        seen: set[tuple[str, str]] = set()
        for row in reader:
            key = (row["query_id"], row["doc_id"])
            if key in seen:
                raise RuntimeError(f"duplicate qrel: {key}")
            seen.add(key)
            rows.append(Qrel(row["query_id"], row["doc_id"], int(row["relevance"])))
    return tuple(rows)


def load_run(run_dir: Path, artifact_root: Path) -> RunData:
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if not summary.get("allQueriesComplete"):
        raise RuntimeError(f"target-validity run is incomplete: {run_dir}")
    records: dict[str, dict[str, Any]] = {}
    for path in sorted((run_dir / "queries").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        query_id = record.get("queryId")
        if not isinstance(query_id, str) or record.get("status") != "ok":
            raise RuntimeError(f"invalid successful Query record: {path}")
        if query_id in records:
            raise RuntimeError(f"duplicate Query record: {query_id}")
        expected_hash = record.pop("recordSha256", None)
        if expected_hash != canonical_hash(record):
            raise RuntimeError(f"Query record hash mismatch: {path}")
        record["recordSha256"] = expected_hash
        records[query_id] = record
    dataset = run["dataset"]
    source = artifact_root / "datasets" / dataset / "source"
    queries_table = pq.read_table(source / "queries.parquet", columns=["id"])
    declared_query_ids = tuple(str(value) for value in queries_table.column("id").to_pylist())
    query_limit = run.get("parameters", {}).get("queryLimit")
    query_ids = declared_query_ids[:query_limit]
    if set(query_ids) != records.keys() or len(query_ids) != summary["expectedQueries"]:
        raise RuntimeError("run records do not cover every declared Query")
    qrels = read_qrels(source / "qrels.tsv")
    qrel_queries = {qrel.query_id for qrel in qrels}
    missing_qrels = set(query_ids) - qrel_queries
    if missing_qrels:
        raise RuntimeError(f"declared Queries are absent from qrels: {sorted(missing_qrels)[:5]}")
    return RunData(run_dir, dataset, query_ids, records, qrels)


def method_names(data: RunData) -> tuple[str, ...]:
    names: tuple[str, ...] | None = None
    for query_id in data.query_ids:
        current = tuple(sorted(data.records[query_id]["methods"]))
        if names is None:
            names = current
        elif current != names:
            raise RuntimeError("method rows differ between Queries")
    if names is None or "full-wrrf" not in names:
        raise RuntimeError("run contains no Full WRRF method")
    return names


def metric_values(data: RunData) -> dict[str, dict[str, dict[str, float]]]:
    output: dict[str, dict[str, dict[str, float]]] = {}
    for method in method_names(data):
        run: list[ScoredDoc] = []
        for query_id in data.query_ids:
            order = data.records[query_id]["methods"][method]
            if len(order) != len(set(order)):
                raise RuntimeError(f"duplicate document in {method}/{query_id}")
            run.extend(
                ScoredDoc(query_id, document_id, float(len(order) - rank + 1))
                for rank, document_id in enumerate(order, start=1)
            )
        values = {
            query_id: {label: 0.0 for label in MEASURE_LABEL.values()}
            for query_id in data.query_ids
        }
        for metric in iter_calc(MEASURES, data.qrels, run):
            label = MEASURE_LABEL[str(metric.measure)]
            if metric.query_id in values:
                values[metric.query_id][label] = float(metric.value)
        output[method] = values
    return output


def paired_bootstrap_ci(
    deltas: np.ndarray,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    if deltas.ndim != 1 or len(deltas) == 0 or not np.isfinite(deltas).all():
        raise ValueError("paired bootstrap requires a finite non-empty vector")
    generator = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    chunk = 1_000
    for offset in range(0, replicates, chunk):
        count = min(chunk, replicates - offset)
        indices = generator.integers(0, len(deltas), size=(count, len(deltas)))
        means[offset : offset + count] = deltas[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def longest_prefix(reference: list[str], actual: list[str]) -> int:
    length = 0
    for expected, observed in zip(reference, actual, strict=False):
        if expected != observed:
            break
        length += 1
    return length


def semantic_values(data: RunData) -> dict[str, dict[str, dict[str, float | bool | int]]]:
    output: dict[str, dict[str, dict[str, float | bool | int]]] = {}
    for method in method_names(data):
        if not method.startswith(FIXED_PREFIX):
            continue
        depth = int(method.removeprefix(FIXED_PREFIX))
        per_query: dict[str, dict[str, float | bool | int]] = {}
        for query_id in data.query_ids:
            record = data.records[query_id]
            full = record["methods"]["full-wrrf"][:20]
            actual = record["methods"][method][:20]
            candidates = set(record["densePrefixExternalIds"][:depth]) | set(
                record["sparsePositivePrefixExternalIds"][:depth]
            )
            intersection = len(set(actual) & set(full))
            per_query[query_id] = {
                "candidateUnionContainsFull": set(full) <= candidates,
                "membershipExact": set(actual) == set(full) and len(actual) == len(full),
                "orderedExact": actual == full,
                "longestExactPrefix": longest_prefix(full, actual),
                "fullMembershipRecall": intersection / len(full) if full else 1.0,
            }
        output[method] = per_query
    return output


def write_trec_runs(data: RunData, output: Path) -> list[dict[str, str]]:
    manifests: list[dict[str, str]] = []
    runs = output / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    for method in method_names(data):
        path = runs / f"{method}.run"
        if path.exists():
            raise FileExistsError(path)
        with path.open("x", encoding="utf-8") as handle:
            for query_id in data.query_ids:
                order = data.records[query_id]["methods"][method]
                for rank, document_id in enumerate(order, start=1):
                    score = len(order) - rank + 1
                    handle.write(f"{query_id} Q0 {document_id} {rank} {score:.1f} {method}\n")
        manifests.append({"path": path.relative_to(output).as_posix(), "sha256": sha256_file(path)})
    return manifests


def aggregate_static(run_dir: Path, artifact_root: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    data = load_run(run_dir, artifact_root)
    metrics = metric_values(data)
    semantics = semantic_values(data)
    run_manifest = write_trec_runs(data, output)
    aggregate_rows: list[dict[str, Any]] = []
    per_query_rows: list[dict[str, Any]] = []
    for method in method_names(data):
        for metric_name in MEASURE_LABEL.values():
            observed = np.array(
                [metrics[method][query_id][metric_name] for query_id in data.query_ids],
                dtype=np.float64,
            )
            reference = np.array(
                [metrics["full-wrrf"][query_id][metric_name] for query_id in data.query_ids],
                dtype=np.float64,
            )
            low, high = paired_bootstrap_ci(observed - reference)
            recall_eligible = not (
                metric_name == "Recall@100"
                and method.startswith(FIXED_PREFIX)
                and int(method.removeprefix(FIXED_PREFIX)) < 100
            )
            aggregate_rows.append(
                {
                    "dataset": data.dataset,
                    "method": method,
                    "metric": metric_name,
                    "mean": float(observed.mean()),
                    "pairedDifferenceFromFull": float((observed - reference).mean()),
                    "pairedDifferenceCi95": [low, high],
                    "primaryEligible": recall_eligible,
                }
            )
        for query_id in data.query_ids:
            row = {"dataset": data.dataset, "queryId": query_id, "method": method}
            row.update(metrics[method][query_id])
            if method in semantics:
                row.update(semantics[method][query_id])
            per_query_rows.append(row)

    aggregate = {
        "schema": "target-validity-static-aggregate-v1",
        "dataset": data.dataset,
        "queries": len(data.query_ids),
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED},
        "primaryMetric": "nDCG@10",
        "rows": aggregate_rows,
        "semanticMeans": {
            method: {
                key: float(np.mean([float(row[key]) for row in per_query.values()]))
                for key in (
                    "candidateUnionContainsFull",
                    "membershipExact",
                    "orderedExact",
                    "longestExactPrefix",
                    "fullMembershipRecall",
                )
            }
            for method, per_query in semantics.items()
        },
    }
    write_json_exclusive(output / "aggregate.json", aggregate)
    with (output / "per-query.csv").open("x", encoding="utf-8", newline="") as handle:
        fieldnames = sorted({key for row in per_query_rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_query_rows)
    write_json_exclusive(
        output / "manifest.json",
        {
            "schema": "target-validity-static-aggregate-v1",
            "sourceRunManifestSha256": sha256_file(run_dir / "manifest.json"),
            "files": [
                {"path": "aggregate.json", "sha256": sha256_file(output / "aggregate.json")},
                {"path": "per-query.csv", "sha256": sha256_file(output / "per-query.csv")},
                *run_manifest,
            ],
        },
    )
    return aggregate


def fold_id(query_id: str) -> int:
    value = int(query_id)
    if not 1 <= value <= 30:
        raise ValueError(f"chronological Query is outside topics 1--30: {query_id}")
    return (value - 1) % 5


def fixed_methods(metrics: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        sorted(
            (method for method in metrics if method.startswith(FIXED_PREFIX)),
            key=lambda method: int(method.removeprefix(FIXED_PREFIX)),
        )
    )


def select_l(
    round_one_metrics: dict[str, dict[str, dict[str, float]]],
    calibration_ids: list[str],
) -> str:
    choices = fixed_methods(round_one_metrics)
    scored = []
    for method in choices:
        values = [round_one_metrics[method][query_id]["nDCG@10"] for query_id in calibration_ids]
        mean = float(np.mean(values))
        scored.append((mean, -int(method.removeprefix(FIXED_PREFIX)), method))
    return max(scored)[2]


def _observed_chronological(
    rounds: dict[int, RunData],
    metrics: dict[int, dict[str, dict[str, dict[str, float]]]],
) -> tuple[dict[int, str], list[dict[str, Any]], dict[int, dict[str, dict[str, float]]]]:
    query_ids = rounds[1].query_ids
    selected: dict[int, str] = {}
    calibration_curves: dict[int, dict[str, dict[str, float]]] = {}
    for fold in range(5):
        train = [query_id for query_id in query_ids if fold_id(query_id) != fold]
        selected[fold] = select_l(metrics[1], train)
        calibration_curves[fold] = {
            method: {
                "meanNdcg10": float(
                    np.mean([metrics[1][method][query_id]["nDCG@10"] for query_id in train])
                )
            }
            for method in fixed_methods(metrics[1])
        }
    rows: list[dict[str, Any]] = []
    for round_id in sorted(rounds):
        for metric_name in MEASURE_LABEL.values():
            cross = np.array(
                [
                    metrics[round_id][selected[fold_id(query_id)]][query_id][metric_name]
                    for query_id in query_ids
                ]
            )
            full = np.array(
                [metrics[round_id]["full-wrrf"][query_id][metric_name] for query_id in query_ids]
            )
            rows.append(
                {
                    "round": round_id,
                    "method": "cross-fitted-fixed-L",
                    "metric": metric_name,
                    "mean": float(cross.mean()),
                    "pairedDifferenceFromFull": float((cross - full).mean()),
                }
            )
            for method in ("full-wrrf", "dense", "sparse"):
                values = np.array(
                    [metrics[round_id][method][query_id][metric_name] for query_id in query_ids]
                )
                rows.append(
                    {
                        "round": round_id,
                        "method": method,
                        "metric": metric_name,
                        "mean": float(values.mean()),
                        "pairedDifferenceFromFull": float((values - full).mean()),
                    }
                )
    return selected, rows, calibration_curves


def nested_chronological_bootstrap(
    rounds: dict[int, RunData],
    metrics: dict[int, dict[str, dict[str, dict[str, float]]]],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[dict[str, list[float]], dict[int, dict[str, int]]]:
    generator = np.random.default_rng(seed)
    query_ids = list(rounds[1].query_ids)
    folds = {
        fold: {
            "train": [query_id for query_id in query_ids if fold_id(query_id) != fold],
            "test": [query_id for query_id in query_ids if fold_id(query_id) == fold],
        }
        for fold in range(5)
    }
    distributions: dict[str, np.ndarray] = {
        f"round-{round_id}/{metric_name}": np.empty(replicates, dtype=np.float64)
        for round_id in rounds
        for metric_name in MEASURE_LABEL.values()
    }
    selections = {fold: Counter() for fold in range(5)}
    for replicate in range(replicates):
        sampled: list[tuple[str, str]] = []
        for fold, identities in folds.items():
            train = identities["train"]
            train_sample = [train[index] for index in generator.integers(0, len(train), len(train))]
            method = select_l(metrics[1], train_sample)
            selections[fold][method] += 1
            test = identities["test"]
            test_sample = [test[index] for index in generator.integers(0, len(test), len(test))]
            sampled.extend((query_id, method) for query_id in test_sample)
        for round_id in rounds:
            for metric_name in MEASURE_LABEL.values():
                deltas = [
                    metrics[round_id][method][query_id][metric_name]
                    - metrics[round_id]["full-wrrf"][query_id][metric_name]
                    for query_id, method in sampled
                ]
                distributions[f"round-{round_id}/{metric_name}"][replicate] = np.mean(deltas)
    intervals = {
        key: [float(value) for value in np.quantile(distribution, [0.025, 0.975])]
        for key, distribution in distributions.items()
    }
    return intervals, {fold: dict(counter) for fold, counter in selections.items()}


def aggregate_chronological(
    run_dirs: dict[int, Path], artifact_root: Path, output: Path
) -> dict[str, Any]:
    if set(run_dirs) != set(range(1, 6)):
        raise ValueError("chronological aggregate requires exactly Rounds 1--5")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    rounds = {round_id: load_run(path, artifact_root) for round_id, path in run_dirs.items()}
    if any(set(data.query_ids) != set(rounds[1].query_ids) for data in rounds.values()):
        raise RuntimeError("chronological rounds do not share topics 1--30")
    metrics = {round_id: metric_values(data) for round_id, data in rounds.items()}
    selected, rows, curves = _observed_chronological(rounds, metrics)
    intervals, selection_frequency = nested_chronological_bootstrap(rounds, metrics)
    for row in rows:
        if row["method"] == "cross-fitted-fixed-L":
            row["pairedDifferenceCi95"] = intervals[f"round-{row['round']}/{row['metric']}"]

    semantic_rows: list[dict[str, Any]] = []
    for round_id, data in rounds.items():
        for query_id in data.query_ids:
            method = selected[fold_id(query_id)]
            full = data.records[query_id]["methods"]["full-wrrf"][:20]
            actual = data.records[query_id]["methods"][method][:20]
            semantic_rows.append(
                {
                    "round": round_id,
                    "queryId": query_id,
                    "fold": fold_id(query_id),
                    "selectedMethod": method,
                    "membershipExact": set(actual) == set(full) and len(actual) == len(full),
                    "orderedExact": actual == full,
                    "longestExactPrefix": longest_prefix(full, actual),
                }
            )
    aggregate = {
        "schema": "target-validity-chronological-aggregate-v1",
        "topics": list(rounds[1].query_ids),
        "foldRule": "(integer_topic_id - 1) mod 5",
        "selectedMethodByFold": {str(key): value for key, value in selected.items()},
        "calibrationCurves": {str(key): value for key, value in curves.items()},
        "nestedBootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "pairedDifferenceIntervals": intervals,
            "selectedMethodFrequencyByFold": {
                str(key): value for key, value in selection_frequency.items()
            },
        },
        "rows": rows,
        "semanticRows": semantic_rows,
    }
    write_json_exclusive(output / "aggregate.json", aggregate)
    write_json_exclusive(
        output / "manifest.json",
        {
            "schema": "target-validity-chronological-aggregate-v1",
            "sourceRunManifestSha256": {
                str(round_id): sha256_file(path / "manifest.json")
                for round_id, path in run_dirs.items()
            },
            "aggregateSha256": sha256_file(output / "aggregate.json"),
        },
    )
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    static = subparsers.add_parser("static")
    static.add_argument("--artifact-root", type=Path, required=True)
    static.add_argument("--run", type=Path, required=True)
    static.add_argument("--output", type=Path, required=True)
    chronological = subparsers.add_parser("chronological")
    chronological.add_argument("--artifact-root", type=Path, required=True)
    chronological.add_argument("--round-run", nargs=2, action="append", required=True)
    chronological.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "static":
        result = aggregate_static(args.run, args.artifact_root, args.output)
    else:
        run_dirs = {int(round_id): Path(path) for round_id, path in args.round_run}
        result = aggregate_chronological(run_dirs, args.artifact_root, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
