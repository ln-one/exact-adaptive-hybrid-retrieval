#!/usr/bin/env python3
"""Build a reproducible Lucene BM25 reference index from canonical Parquet."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
from pyserini.index.lucene import LuceneIndexer


INDEX_SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tree_manifest(directory: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for path in sorted(candidate for candidate in directory.rglob("*") if candidate.is_file()):
        entries.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return entries


def java_runtime() -> dict[str, str]:
    java = shutil.which("java")
    if java is None:
        raise RuntimeError("JDK 21 is required; set JAVA_HOME/PATH before building")
    completed = subprocess.run([java, "-version"], capture_output=True, text=True, check=True)
    version = (completed.stderr or completed.stdout).splitlines()[0]
    if "21." not in version and 'version "21' not in version:
        raise RuntimeError(f"JDK 21 is required, found: {version}")
    return {"binary": os.path.realpath(java), "version": version}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--row-batch-size", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.threads <= 0 or args.row_batch_size <= 0:
        raise ValueError("--threads and --row-batch-size must be positive")

    source = args.artifact_root / "datasets" / args.dataset / "source"
    source_manifest_path = source / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    documents = source / "documents.parquet"
    expected_sha = source_manifest["files"]["documents.parquet"]["sha256"]
    if sha256_file(documents) != expected_sha:
        raise RuntimeError(f"source checksum mismatch: {documents}")

    output = args.artifact_root / "datasets" / args.dataset / "sparse" / "lucene-bm25"
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        print(f"already built: {manifest_path}")
        return
    temporary = output.with_name(output.name + ".building")
    if temporary.exists():
        raise RuntimeError(f"stale build directory exists: {temporary}")
    temporary.mkdir(parents=True)

    try:
        java = java_runtime()
        index_dir = temporary / "index"
        indexer = LuceneIndexer(index_dir=str(index_dir), threads=args.threads)
        indexed = 0
        try:
            parquet = pq.ParquetFile(documents)
            for batch in parquet.iter_batches(batch_size=args.row_batch_size, columns=["id", "text"]):
                ids = batch.column("id").to_pylist()
                texts = batch.column("text").to_pylist()
                for external_id, text in zip(ids, texts, strict=True):
                    indexer.add_doc_dict({"id": external_id, "contents": text})
                indexed += len(ids)
                if indexed % 100_000 == 0:
                    print(f"{args.dataset}: indexed {indexed:,}", flush=True)
        finally:
            indexer.close()
        expected_count = source_manifest["counts"]["documents"]
        if indexed != expected_count:
            raise RuntimeError(f"indexed {indexed}, expected {expected_count}")
        files = tree_manifest(index_dir)
        index_bytes = sum(entry["bytes"] for entry in files)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    manifest = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "dataset": args.dataset,
        "source": {
            "documents_sha256": expected_sha,
            "documents": expected_count,
        },
        "representation": {
            "kind": "lucene_bm25_reference",
            "collection": "JsonCollection",
            "analyzer": "Pyserini DefaultEnglishAnalyzer; Porter stemming; stopwords removed",
            "scoring": "BM25; query-time parameters recorded by each run",
        },
        "builder": {
            "pyserini": importlib.metadata.version("pyserini"),
            "python": sys.version,
            "java": java,
            "threads": args.threads,
            "row_batch_size": args.row_batch_size,
        },
        "index": {"bytes": index_bytes, "files": files},
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(output)
    print(json.dumps({"dataset": args.dataset, "documents": indexed, "bytes": index_bytes}))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("cancelled", file=sys.stderr)
        raise SystemExit(130)
