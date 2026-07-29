#!/usr/bin/env python3
"""Verify a canonical Dense artifact without changing it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_ids(path: Path):
    for batch in pq.ParquetFile(path).iter_batches(columns=["id"]):
        yield from batch.column(0).to_pylist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--kind", choices=("documents", "queries"), required=True)
    args = parser.parse_args()

    base = args.artifact_root / "datasets" / args.dataset / "dense" / "bge-small-en-v1.5-f32"
    manifest = json.loads((base / f"{args.kind}-manifest.json").read_text())
    source = args.artifact_root / "datasets" / args.dataset / "source" / f"{args.kind}.parquet"
    if manifest["source_sha256"] != sha256_file(source):
        raise RuntimeError("source checksum differs from embedding manifest")

    expected_ids = source_ids(source)
    total = 0
    max_norm_error = 0.0
    for spec in manifest["shards"]:
        shard = base / spec["name"]
        if sha256_file(shard) != spec["sha256"]:
            raise RuntimeError(f"checksum mismatch: {shard.name}")
        table = pq.read_table(shard, columns=["id", "vector"])
        if table.num_rows != spec["rows"]:
            raise RuntimeError(f"row mismatch: {shard.name}")
        matrix = np.asarray(table.column("vector").to_pylist(), dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != manifest["dimension"]:
            raise RuntimeError(f"invalid vector shape: {shard.name}")
        if not np.isfinite(matrix).all():
            raise RuntimeError(f"non-finite vector: {shard.name}")
        max_norm_error = max(max_norm_error, float(np.max(np.abs(np.linalg.norm(matrix, axis=1) - 1.0))))
        for external_id in table.column("id").to_pylist():
            try:
                expected_id = next(expected_ids)
            except StopIteration as error:
                raise RuntimeError("too many Dense vectors") from error
            if external_id != expected_id:
                raise RuntimeError("embedding identities/order differ from canonical source")
        total += table.num_rows
    try:
        next(expected_ids)
    except StopIteration:
        pass
    else:
        raise RuntimeError("embedding identities/order differ from canonical source")
    if total != manifest["vectors"]:
        raise RuntimeError("embedding count differs from manifest")
    print(json.dumps({"dataset": args.dataset, "kind": args.kind, "vectors": total,
                      "max_norm_error": max_norm_error}, sort_keys=True))


if __name__ == "__main__":
    main()
