#!/usr/bin/env python3
"""Verify a frozen BM25-impact Sparse representation without regenerating it."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pyarrow.parquet as pq


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_ids(path: Path):
    for batch in pq.ParquetFile(path).iter_batches(columns=["id"]):
        yield from batch.column("id").to_pylist()


def verify_vectors(paths: list[Path], source: Path, expected: int) -> int:
    expected_ids = source_ids(source)
    count = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        if parquet.schema_arrow.names != ["id", "indices", "values"]:
            raise RuntimeError(f"unexpected Sparse schema: {path}")
        for batch in parquet.iter_batches(batch_size=5_000, columns=["id", "indices", "values"]):
            for external_id, indices, values in zip(
                batch.column("id").to_pylist(), batch.column("indices").to_pylist(), batch.column("values").to_pylist(), strict=True
            ):
                if not external_id or len(indices) != len(values):
                    raise RuntimeError(f"invalid Sparse row in {path}")
                try:
                    expected_id = next(expected_ids)
                except StopIteration as error:
                    raise RuntimeError(f"too many Sparse rows in {path}") from error
                if external_id != expected_id:
                    raise RuntimeError(f"Sparse identity/order differs from canonical source in {path}")
                if indices != sorted(set(indices)) or any(index < 0 for index in indices):
                    raise RuntimeError(f"non-canonical Sparse indices in {path}")
                if any(not math.isfinite(value) or value < 0 for value in values):
                    raise RuntimeError(f"non-finite or negative Sparse weight in {path}")
                count += 1
    if count != expected:
        raise RuntimeError(f"Sparse row count {count}, expected {expected}")
    try:
        next(expected_ids)
    except StopIteration:
        return count
    raise RuntimeError("Sparse rows ended before canonical source")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    root = args.artifact_root / "datasets" / args.dataset / "sparse" / "bm25-impact-v1"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    source = args.artifact_root / "datasets" / args.dataset / "source"
    for name, expected in (("documents.parquet", manifest["source"]["documents_sha256"]), ("queries.parquet", manifest["source"]["queries_sha256"])):
        if sha256_file(source / name) != expected:
            raise RuntimeError(f"canonical source checksum mismatch: {name}")
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"Sparse artifact checksum mismatch: {path}")
    documents = verify_vectors(
        [root / name for name in manifest["shards"]["documents"]], source / "documents.parquet", manifest["source"]["documents"]
    )
    queries = verify_vectors(
        [root / name for name in manifest["shards"]["queries"]], source / "queries.parquet", manifest["source"]["queries"]
    )
    vocabulary = pq.ParquetFile(root / manifest["vocabulary"]["path"]).metadata.num_rows
    if vocabulary != manifest["vocabulary"]["terms"]:
        raise RuntimeError("vocabulary size mismatch")
    print(json.dumps({"dataset": args.dataset, "documents": documents, "queries": queries, "vocabulary": vocabulary}, sort_keys=True))


if __name__ == "__main__":
    main()
