#!/usr/bin/env python3
"""Prepare the five official TREC-COVID chronological snapshots.

Only CORD-19 metadata is needed because the frozen retrieval unit is title plus
abstract, matching the existing BEIR TREC-COVID export recipe.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROUND_RELEASES = {
    1: "2020-04-10",
    2: "2020-05-01",
    3: "2020-05-19",
    4: "2020-06-19",
    5: "2020-07-16",
}
EXPECTED_VALID_DOCUMENTS = {1: 51_103, 2: 59_851, 3: 128_492, 4: 157_817, 5: 191_175}
COMMON_TOPICS = frozenset(str(value) for value in range(1, 31))
NIST_ROOT = "https://ir.nist.gov/trec-covid/data"
CORD_ROOT = "https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "stratumind-bench/0.1"})
    with (
        urllib.request.urlopen(request, timeout=300) as response,
        temporary.open("xb") as output,
    ):
        shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(destination)


def atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def download_round(artifact_root: Path, round_id: int) -> tuple[Path, dict[str, dict[str, object]]]:
    release = ROUND_RELEASES[round_id]
    destination = artifact_root / "sources" / "trec-covid-chronological" / f"round-{round_id}"
    urls = {
        "metadata.csv": f"{CORD_ROOT}/{release}/metadata.csv",
        "docids.txt": f"{NIST_ROOT}/docids-rnd{round_id}.txt",
        "topics.xml": f"{NIST_ROOT}/topics-rnd{round_id}.xml",
        "qrels.txt": f"{NIST_ROOT}/qrels-covid_d{round_id}_j0.5-5.txt",
    }
    files: dict[str, dict[str, object]] = {}
    for name, url in urls.items():
        path = destination / name
        fetch(url, path)
        files[name] = {
            "url": url,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    atomic_json(
        destination / "source-manifest.json",
        {"round": round_id, "release": release, "files": files},
    )
    return destination, files


def read_valid_ids(path: Path, expected: int) -> set[str]:
    values = [
        line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if len(values) != expected or len(values) != len(set(values)):
        raise RuntimeError(
            f"unexpected valid-document list: rows={len(values)} unique={len(set(values))}"
        )
    return set(values)


def write_documents(
    metadata: Path,
    output: Path,
    valid_ids: set[str],
    *,
    batch_rows: int,
) -> tuple[int, int]:
    schema = pa.schema([("id", pa.string()), ("text", pa.string())])
    writer = pq.ParquetWriter(output, schema, compression="zstd")
    ids: list[str] = []
    texts: list[str] = []
    seen: set[str] = set()
    empty = 0

    def flush() -> None:
        if not ids:
            return
        writer.write_table(pa.table({"id": ids, "text": texts}, schema=schema))
        ids.clear()
        texts.clear()

    try:
        with metadata.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"cord_uid", "title", "abstract"}
            if reader.fieldnames is None or not required <= set(reader.fieldnames):
                raise RuntimeError("CORD-19 metadata lacks cord_uid/title/abstract")
            for row in reader:
                document_id = (row.get("cord_uid") or "").strip()
                if document_id not in valid_ids:
                    continue
                if document_id in seen:
                    raise RuntimeError(f"duplicate valid cord_uid in metadata: {document_id}")
                title = (row.get("title") or "").strip()
                abstract = (row.get("abstract") or "").strip()
                text = f"{title}\n{abstract}".strip() if title else abstract
                empty += not text
                ids.append(document_id)
                texts.append(text)
                seen.add(document_id)
                if len(ids) == batch_rows:
                    flush()
        flush()
    finally:
        writer.close()
    missing = valid_ids - seen
    if missing:
        sample = sorted(missing)[:5]
        raise RuntimeError(f"metadata omitted {len(missing)} valid doc IDs, e.g. {sample}")
    return len(seen), empty


def topic_rows(path: Path) -> list[tuple[str, str]]:
    root = ET.parse(path).getroot()
    rows: list[tuple[str, str]] = []
    for topic in root.findall(".//topic"):
        query_id = (topic.get("number") or "").strip()
        if query_id not in COMMON_TOPICS:
            continue
        question = (topic.findtext("question") or "").strip()
        if not question:
            raise RuntimeError(f"topic {query_id} has no question text")
        rows.append((query_id, question))
    if {query_id for query_id, _ in rows} != COMMON_TOPICS:
        raise RuntimeError("round topic file does not contain every common topic 1--30")
    return sorted(rows, key=lambda item: int(item[0]))


def write_text_rows(rows: Iterable[tuple[str, str]], output: Path) -> int:
    values = list(rows)
    pq.write_table(
        pa.table(
            {"id": [item[0] for item in values], "text": [item[1] for item in values]},
            schema=pa.schema([("id", pa.string()), ("text", pa.string())]),
        ),
        output,
        compression="zstd",
    )
    return len(values)


def write_qrels(source: Path, output: Path, valid_ids: set[str]) -> int:
    rows: list[tuple[str, str, int]] = []
    seen: set[tuple[str, str]] = set()
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 4:
            raise RuntimeError(f"invalid chronological qrel: {line!r}")
        query_id, _, document_id, relevance_text = fields
        if query_id not in COMMON_TOPICS:
            continue
        if document_id not in valid_ids:
            raise RuntimeError(f"chronological qrel references invalid document {document_id}")
        key = (query_id, document_id)
        if key in seen:
            raise RuntimeError(f"duplicate chronological qrel: {key}")
        relevance = int(relevance_text)
        if relevance not in {0, 1, 2}:
            raise RuntimeError(f"unsupported TREC-COVID relevance grade: {relevance}")
        rows.append((query_id, document_id, relevance))
        seen.add(key)
    with output.open("x", encoding="utf-8") as handle:
        handle.write("query_id\tdoc_id\trelevance\n")
        for query_id, document_id, relevance in rows:
            handle.write(f"{query_id}\t{document_id}\t{relevance}\n")
        handle.flush()
        os.fsync(handle.fileno())
    return len(rows)


def prepare_round(artifact_root: Path, round_id: int, batch_rows: int) -> dict[str, object]:
    raw, upstream = download_round(artifact_root, round_id)
    dataset = f"trec-covid-chrono-r{round_id}"
    output = artifact_root / "datasets" / dataset / "source"
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    temporary = output.with_name(output.name + ".building")
    if temporary.exists():
        raise RuntimeError(f"stale chronological build directory exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        valid_ids = read_valid_ids(raw / "docids.txt", EXPECTED_VALID_DOCUMENTS[round_id])
        documents, empty_documents = write_documents(
            raw / "metadata.csv",
            temporary / "documents.parquet",
            valid_ids,
            batch_rows=batch_rows,
        )
        queries = write_text_rows(topic_rows(raw / "topics.xml"), temporary / "queries.parquet")
        qrels = write_qrels(raw / "qrels.txt", temporary / "qrels.tsv", valid_ids)
        manifest: dict[str, object] = {
            "schema_version": 1,
            "dataset_id": f"trec-covid/chronological/round-{round_id}",
            "name": dataset,
            "round": round_id,
            "cord19_release": ROUND_RELEASES[round_id],
            "counts": {"documents": documents, "queries": queries, "qrels": qrels},
            "files": {
                name: {"sha256": sha256_file(temporary / name)}
                for name in ("documents.parquet", "queries.parquet", "qrels.tsv")
            },
            "upstream": upstream,
            "retrieval_unit": (
                "official CORD-19 object; title + newline + abstract when title exists"
            ),
            "query_text": "official topic question field",
            "topic_scope": "common topics 1--30",
            "empty_documents": empty_documents,
            "rechunked": False,
            "sampled": False,
        }
        atomic_json(temporary / "manifest.json", manifest)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(output)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--round", type=int, action="append", choices=range(1, 6))
    parser.add_argument("--batch-rows", type=int, default=25_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_rows <= 0:
        raise ValueError("batch rows must be positive")
    rounds = sorted(set(args.round or ROUND_RELEASES))
    results = [prepare_round(args.artifact_root, round_id, args.batch_rows) for round_id in rounds]
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
