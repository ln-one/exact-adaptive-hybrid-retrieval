#!/usr/bin/env python3
"""Stream verified canonical BM25-impact vectors into an existing Qdrant collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import httpx
import pyarrow.parquet as pq
from dataset_gate import assert_dataset_eligible
from load_qdrant_dense import point_id


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--vector-name", default="sparse")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--wait", action="store_true")
    return parser.parse_args()


def verified_manifest(base: Path) -> dict[str, object]:
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    representation = manifest["representation"]
    if representation["kind"] != "bm25_impact_sparse_vector" or not representation["nonnegative"]:
        raise RuntimeError("only canonical non-negative BM25-impact vectors are accepted")
    for entry in manifest["files"]:
        path = base / entry["path"]
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"Sparse artifact checksum mismatch: {path}")
    return manifest


def valid_sparse_vector(indices: list[int], values: list[float]) -> bool:
    return (
        len(indices) == len(values)
        and indices == sorted(set(indices))
        and all(index >= 0 for index in indices)
        and all(math.isfinite(value) and value >= 0 for value in values)
    )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    assert_dataset_eligible(args.dataset)
    base = args.artifact_root / "datasets" / args.dataset / "sparse" / "bm25-impact-v1"
    manifest = verified_manifest(base)
    client = httpx.Client(base_url=args.url.rstrip("/"), timeout=120.0)
    sent = 0
    empty = 0
    try:
        for name in manifest["shards"]["documents"]:
            shard = base / name
            for batch in pq.ParquetFile(shard).iter_batches(
                batch_size=args.batch_size, columns=["id", "indices", "values"]
            ):
                records = batch.to_pydict()
                points = []
                for external_id, indices, values in zip(
                    records["id"], records["indices"], records["values"], strict=True
                ):
                    if not valid_sparse_vector(indices, values):
                        raise RuntimeError(f"invalid Sparse vector in {shard.name}: {external_id}")
                    if not indices:
                        empty += 1
                        continue
                    points.append(
                        {
                            "id": point_id(args.dataset, external_id),
                            "vector": {args.vector_name: {"indices": indices, "values": values}},
                        }
                    )
                if points:
                    response = client.put(
                        f"/collections/{args.collection}/points/vectors",
                        params={"wait": str(args.wait).lower()},
                        json={"points": points},
                    )
                    response.raise_for_status()
                sent += len(points)
                if sent % 10_000 == 0:
                    print(json.dumps({"dataset": args.dataset, "sent": sent}), flush=True)
    finally:
        client.close()
    print(
        json.dumps(
            {
                "collection": args.collection,
                "dataset": args.dataset,
                "points": sent,
                "empty": empty,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
