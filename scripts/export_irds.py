#!/usr/bin/env python3
"""Export an official ir_datasets corpus without changing its retrieval units."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

import ir_datasets
import pyarrow as pa
import pyarrow.parquet as pq


SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--row-group-size", type=int, default=50_000)
    parser.add_argument(
        "--reuse-documents-from",
        help="canonical dataset name with an identical official document corpus",
    )
    return parser.parse_args()


def atomic_replace(source: Path, destination: Path) -> None:
    source.replace(destination)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stringify_document(document: Any) -> tuple[str, str]:
    data = document._asdict()
    document_id = str(data.pop("doc_id"))
    title = str(data.pop("title", "") or "").strip()
    text = str(data.pop("text", "") or "").strip()
    combined = f"{title}\n{text}".strip() if title else text
    # Some official BEIR records deliberately contain no retrievable body
    # (for example a CORD-19 metadata-only record). Keep their identity and
    # empty representation rather than silently dropping them or inventing
    # text from metadata such as URLs.
    return document_id, combined


def stringify_query(query: Any) -> tuple[str, str]:
    data = query._asdict()
    query_id = str(data.pop("query_id"))
    text = str(data.pop("text", "") or "").strip()
    if not text:
        raise ValueError(f"query {query_id!r} has empty text")
    return query_id, text


def write_parquet(
    rows: Iterable[tuple[str, str]],
    output: Path,
    row_group_size: int,
) -> int:
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    schema = pa.schema([("id", pa.string()), ("text", pa.string())])
    writer = pq.ParquetWriter(temporary, schema, compression="zstd")
    count = 0
    ids: list[str] = []
    texts: list[str] = []
    try:
        for item_id, text in rows:
            ids.append(item_id)
            texts.append(text)
            if len(ids) == row_group_size:
                writer.write_table(pa.table({"id": ids, "text": texts}, schema=schema))
                count += len(ids)
                ids.clear()
                texts.clear()
        if ids:
            writer.write_table(pa.table({"id": ids, "text": texts}, schema=schema))
            count += len(ids)
    finally:
        writer.close()
    atomic_replace(temporary, output)
    return count


def write_qrels(dataset: Any, output: Path) -> int:
    temporary = output.with_suffix(output.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write("query_id\tdoc_id\trelevance\n")
        for qrel in dataset.qrels_iter():
            handle.write(f"{qrel.query_id}\t{qrel.doc_id}\t{qrel.relevance}\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    atomic_replace(temporary, output)
    return count


def main() -> None:
    args = parse_args()
    if args.row_group_size <= 0:
        raise ValueError("--row-group-size must be positive")
    output = args.artifact_root / "datasets" / args.name / "source"
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        print(f"already exported: {manifest_path}")
        return

    temporary = output.with_name(output.name + ".building")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    dataset = ir_datasets.load(args.dataset_id)
    shared_documents_from = None
    if args.reuse_documents_from:
        source = args.artifact_root / "datasets" / args.reuse_documents_from / "source"
        source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        source_documents = source / "documents.parquet"
        if not source_documents.exists():
            raise FileNotFoundError(f"missing shared canonical documents: {source_documents}")
        os.link(source_documents, temporary / "documents.parquet")
        documents = source_manifest["counts"]["documents"]
        shared_documents_from = args.reuse_documents_from
    else:
        documents = write_parquet(
            (stringify_document(document) for document in dataset.docs_iter()),
            temporary / "documents.parquet",
            args.row_group_size,
        )
    queries = write_parquet(
        (stringify_query(query) for query in dataset.queries_iter()),
        temporary / "queries.parquet",
        args.row_group_size,
    )
    qrels = write_qrels(dataset, temporary / "qrels.tsv")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": args.dataset_id,
        "name": args.name,
        "counts": {"documents": documents, "queries": queries, "qrels": qrels},
        "files": {
            filename: {"sha256": sha256_file(temporary / filename)}
            for filename in ("documents.parquet", "queries.parquet", "qrels.tsv")
        },
        "retrieval_unit": "official corpus object; title + newline + text when title exists",
        "rechunked": False,
        "sampled": False,
    }
    if shared_documents_from:
        manifest["documents_shared_from"] = shared_documents_from
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing artifact: {output}")
    temporary.replace(output)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("cancelled", file=sys.stderr)
        raise SystemExit(130)
