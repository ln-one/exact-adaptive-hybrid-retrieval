#!/usr/bin/env python3
"""Build frozen non-negative BM25-impact vectors for the Stratumind Sparse channel.

This representation deliberately remains distinct from the Lucene reference
index. It shares Anserini's English analysis pipeline, then materializes the
document-side BM25 impact of each term so Qdrant can consume it as a standard
sparse vector.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pyserini.analysis import Analyzer, get_lucene_analyzer

from dataset_gate import assert_dataset_eligible


SCHEMA_VERSION = 1
GIB = 1024**3
VECTOR_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("indices", pa.list_(pa.int32())),
        ("values", pa.list_(pa.float32())),
    ]
)
VOCABULARY_SCHEMA = pa.schema(
    [("term", pa.string()), ("term_id", pa.int32()), ("document_frequency", pa.int32())]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--k1", type=float, default=0.9)
    parser.add_argument("--b", type=float, default=0.4)
    parser.add_argument("--row-batch-size", type=int, default=10_000)
    parser.add_argument("--shard-rows", type=int, default=100_000)
    parser.add_argument("--minimum-free-after-gib", type=int, default=120)
    parser.add_argument("--reuse-documents-from", help="dataset with an identical verified canonical document corpus")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def assert_capacity(root: Path, rows: int, minimum_free_after_gib: int) -> None:
    """Reserve a conservative 1 KiB per document for sparse shards and vocabulary."""
    if minimum_free_after_gib < 0:
        raise ValueError("--minimum-free-after-gib must be non-negative")
    estimated_bytes = rows * 1024
    free = shutil.disk_usage(root).free
    reserve = minimum_free_after_gib * GIB
    if free - estimated_bytes < reserve:
        raise RuntimeError(
            "capacity gate failed: "
            f"free={free / GIB:.1f} GiB, estimated sparse artifact={estimated_bytes / GIB:.1f} GiB, "
            f"required reserve={minimum_free_after_gib} GiB"
        )


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


def verified_source(root: Path, dataset: str) -> tuple[Path, Path, dict[str, object]]:
    source = root / "datasets" / dataset / "source"
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    documents, queries = source / "documents.parquet", source / "queries.parquet"
    for name, path in (("documents.parquet", documents), ("queries.parquet", queries)):
        if sha256_file(path) != manifest["files"][name]["sha256"]:
            raise RuntimeError(f"source checksum mismatch: {path}")
    return documents, queries, manifest


def term_statistics(documents: Path, analyzer: Analyzer, batch_size: int) -> tuple[Counter[str], int, int]:
    document_frequency: Counter[str] = Counter()
    document_count = 0
    total_length = 0
    for batch in pq.ParquetFile(documents).iter_batches(batch_size=batch_size, columns=["text"]):
        for text in batch.column("text").to_pylist():
            terms = analyzer.analyze(text)
            document_count += 1
            total_length += len(terms)
            document_frequency.update(set(terms))
    return document_frequency, document_count, total_length


def write_vocabulary(path: Path, document_frequency: Counter[str]) -> dict[str, int]:
    terms = sorted(document_frequency)
    if len(terms) >= 2**31:
        raise RuntimeError("vocabulary exceeds int32 term-id domain")
    table = pa.table(
        {
            "term": terms,
            "term_id": list(range(len(terms))),
            "document_frequency": [document_frequency[term] for term in terms],
        },
        schema=VOCABULARY_SCHEMA,
    )
    pq.write_table(table, path, compression="zstd")
    return {term: index for index, term in enumerate(terms)}


def idf(documents: int, frequency: int) -> float:
    return math.log1p((documents - frequency + 0.5) / (frequency + 0.5))


def flush_shard(
    destination: Path, stem: str, shard: int, ids: list[str], indices: list[list[int]], values: list[list[float]]
) -> Path:
    path = destination / f"{stem}-{shard:06d}.parquet"
    pq.write_table(
        pa.table({"id": ids, "indices": indices, "values": values}, schema=VECTOR_SCHEMA),
        path,
        compression="zstd",
    )
    ids.clear()
    indices.clear()
    values.clear()
    return path


def write_document_vectors(
    documents: Path,
    destination: Path,
    analyzer: Analyzer,
    vocabulary: dict[str, int],
    document_frequency: Counter[str],
    total_documents: int,
    average_length: float,
    k1: float,
    b: float,
    batch_size: int,
    shard_rows: int,
) -> tuple[int, list[Path]]:
    ids: list[str] = []
    indices: list[list[int]] = []
    values: list[list[float]] = []
    outputs: list[Path] = []
    count = 0
    shard = 0
    for batch in pq.ParquetFile(documents).iter_batches(batch_size=batch_size, columns=["id", "text"]):
        for external_id, text in zip(batch.column("id").to_pylist(), batch.column("text").to_pylist(), strict=True):
            frequencies = Counter(analyzer.analyze(text))
            length = sum(frequencies.values())
            denominator_base = k1 * (1.0 - b + b * length / average_length)
            terms = sorted(frequencies, key=vocabulary.__getitem__)
            ids.append(external_id)
            indices.append([vocabulary[term] for term in terms])
            values.append(
                [
                    float(idf(total_documents, document_frequency[term]) * (frequency * (k1 + 1.0)) / (frequency + denominator_base))
                    for term, frequency in ((term, frequencies[term]) for term in terms)
                ]
            )
            count += 1
            if len(ids) == shard_rows:
                outputs.append(flush_shard(destination, "documents", shard, ids, indices, values))
                shard += 1
        if count and count % 100_000 == 0:
            print(f"documents: vectorized {count:,}", flush=True)
    if ids:
        outputs.append(flush_shard(destination, "documents", shard, ids, indices, values))
    return count, outputs


def write_query_vectors(
    queries: Path,
    destination: Path,
    analyzer: Analyzer,
    vocabulary: dict[str, int],
    batch_size: int,
    shard_rows: int,
) -> tuple[int, list[Path]]:
    ids: list[str] = []
    indices: list[list[int]] = []
    values: list[list[float]] = []
    outputs: list[Path] = []
    count = 0
    shard = 0
    for batch in pq.ParquetFile(queries).iter_batches(batch_size=batch_size, columns=["id", "text"]):
        for external_id, text in zip(batch.column("id").to_pylist(), batch.column("text").to_pylist(), strict=True):
            terms = sorted(set(analyzer.analyze(text)) & vocabulary.keys(), key=vocabulary.__getitem__)
            ids.append(external_id)
            indices.append([vocabulary[term] for term in terms])
            values.append([1.0] * len(terms))
            count += 1
            if len(ids) == shard_rows:
                outputs.append(flush_shard(destination, "queries", shard, ids, indices, values))
                shard += 1
    if ids:
        outputs.append(flush_shard(destination, "queries", shard, ids, indices, values))
    return count, outputs


def reuse_document_vectors(
    root: Path, source_dataset: str, target_document_sha: str, temporary: Path, k1: float, b: float
) -> tuple[dict[str, int], int, list[Path], dict[str, object]]:
    source = root / "datasets" / source_dataset / "sparse" / "bm25-impact-v1"
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest["source"]["documents_sha256"] != target_document_sha:
        raise RuntimeError("shared Sparse document corpus checksum differs")
    representation = manifest["representation"]
    if representation["kind"] != "bm25_impact_sparse_vector" or representation["k1"] != k1 or representation["b"] != b:
        raise RuntimeError("shared Sparse representation differs from requested profile")
    checksums = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
    names = [manifest["vocabulary"]["path"], *manifest["shards"]["documents"]]
    for name in names:
        path = source / name
        if not path.is_file() or sha256_file(path) != checksums.get(name):
            raise RuntimeError(f"shared Sparse artifact checksum mismatch: {path}")
        os.link(path, temporary / name)
    vocabulary_table = pq.read_table(temporary / manifest["vocabulary"]["path"], columns=["term", "term_id"])
    terms = vocabulary_table.to_pydict()
    vocabulary = dict(zip(terms["term"], terms["term_id"], strict=True))
    document_shards = [temporary / name for name in manifest["shards"]["documents"]]
    return vocabulary, manifest["source"]["documents"], document_shards, representation


def main() -> None:
    args = parse_args()
    if args.k1 < 0 or not 0 <= args.b <= 1 or args.row_batch_size <= 0 or args.shard_rows <= 0:
        raise ValueError("invalid BM25 parameters or batch sizes")
    assert_dataset_eligible(args.dataset)
    documents, queries, source_manifest = verified_source(args.artifact_root, args.dataset)
    estimated_rows = 0 if args.reuse_documents_from else source_manifest["counts"]["documents"]
    assert_capacity(args.artifact_root, estimated_rows, args.minimum_free_after_gib)
    output = args.artifact_root / "datasets" / args.dataset / "sparse" / "bm25-impact-v1"
    if (output / "manifest.json").exists():
        print(f"already built: {output / 'manifest.json'}")
        return
    temporary = output.with_name(output.name + ".building")
    if temporary.exists():
        raise RuntimeError(f"stale build directory exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        java = java_runtime()
        analyzer = Analyzer(get_lucene_analyzer(language="en", stemming=True, stemmer="porter", stopwords=True))
        if args.reuse_documents_from:
            vocabulary, actual_documents, document_shards, representation = reuse_document_vectors(
                args.artifact_root, args.reuse_documents_from,
                source_manifest["files"]["documents.parquet"]["sha256"], temporary, args.k1, args.b,
            )
        else:
            document_frequency, documents_count, total_length = term_statistics(documents, analyzer, args.row_batch_size)
            if documents_count != source_manifest["counts"]["documents"]:
                raise RuntimeError("document count changed during statistics pass")
            if total_length == 0:
                raise RuntimeError("canonical corpus has no analyzed terms")
            average_length = total_length / documents_count
            vocabulary = write_vocabulary(temporary / "vocabulary.parquet", document_frequency)
            actual_documents, document_shards = write_document_vectors(
                documents, temporary, analyzer, vocabulary, document_frequency, documents_count,
                average_length, args.k1, args.b, args.row_batch_size, args.shard_rows,
            )
            representation = {
                "kind": "bm25_impact_sparse_vector",
                "analyzer": "Pyserini DefaultEnglishAnalyzer; Porter stemming; stopwords removed",
                "term_id_assignment": "lexicographic UTF-8 term order",
                "document_weight": "idf * tf*(k1+1)/(tf+k1*(1-b+b*dl/avgdl))",
                "query_weight": "binary unique analyzed terms",
                "k1": args.k1,
                "b": args.b,
                "average_document_length": average_length,
                "nonnegative": True,
                "index_dtype": "int32",
                "value_dtype": "float32",
            }
        actual_queries, query_shards = write_query_vectors(
            queries, temporary, analyzer, vocabulary, args.row_batch_size, args.shard_rows
        )
        expected = source_manifest["counts"]
        if actual_documents != expected["documents"] or actual_queries != expected["queries"]:
            raise RuntimeError("vector count differs from canonical source")
        files = tree_manifest(temporary)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "dataset": args.dataset,
            "source": {
                "documents_sha256": source_manifest["files"]["documents.parquet"]["sha256"],
                "queries_sha256": source_manifest["files"]["queries.parquet"]["sha256"],
                "documents": actual_documents,
                "queries": actual_queries,
            },
            "representation": representation,
            "vocabulary": {"terms": len(vocabulary), "path": "vocabulary.parquet"},
            "builder": {
                "pyserini": importlib.metadata.version("pyserini"),
                "python": sys.version,
                "java": java,
                "row_batch_size": args.row_batch_size,
                "shard_rows": args.shard_rows,
            },
            "shards": {
                "documents": [path.name for path in document_shards],
                "queries": [path.name for path in query_shards],
            },
            "files": files,
        }
        if args.reuse_documents_from:
            manifest["documents_reused_from"] = args.reuse_documents_from
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({"dataset": args.dataset, "documents": actual_documents, "queries": actual_queries}))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("cancelled", file=sys.stderr)
        raise SystemExit(130)
