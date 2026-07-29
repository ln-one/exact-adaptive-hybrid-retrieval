#!/usr/bin/env python3
"""Verify an immutable canonical corpus/qrels export without changing it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def count_qrels(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        header = next(handle).rstrip("\n")
        if header != "query_id\tdoc_id\trelevance":
            raise RuntimeError(f"unexpected qrels header: {header!r}")
        return sum(1 for _ in handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    source = args.artifact_root / "datasets" / args.dataset / "source"
    manifest = json.loads((source / "manifest.json").read_text())
    files = {name: source / name for name in ("documents.parquet", "queries.parquet", "qrels.tsv")}
    for name, path in files.items():
        if sha256_file(path) != manifest["files"][name]["sha256"]:
            raise RuntimeError(f"checksum mismatch: {path}")
    document_count = pq.ParquetFile(files["documents.parquet"]).metadata.num_rows
    query_count = pq.ParquetFile(files["queries.parquet"]).metadata.num_rows
    qrel_count = count_qrels(files["qrels.tsv"])
    actual = {"documents": document_count, "queries": query_count, "qrels": qrel_count}
    if actual != manifest["counts"]:
        raise RuntimeError(f"count mismatch: expected {manifest['counts']}, found {actual}")
    print(json.dumps({"dataset": args.dataset, **actual}, sort_keys=True))


if __name__ == "__main__":
    main()
