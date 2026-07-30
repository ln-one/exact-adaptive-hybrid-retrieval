"""Verified access to frozen canonical query representations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

DENSE_PROFILE = "bge-small-en-v1.5-f32"
SPARSE_PROFILE = "bm25-impact-v1"
CANONICAL_COLLECTION_FORMAT = "qdrant-v1.18.2-exact-v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


@dataclass(frozen=True)
class QueryInput:
    query_id: str
    dense: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]


@dataclass(frozen=True)
class DatasetSnapshot:
    dataset: str
    split: str
    document_count: int
    queries: tuple[QueryInput, ...]
    source_manifest_sha256: str
    dense_manifest_sha256: str
    sparse_manifest_sha256: str


@dataclass(frozen=True)
class CollectionSnapshot:
    path: Path
    manifest_path: Path
    manifest_sha256: str
    snapshot_sha256: str
    collection_config_sha256: str
    points: int


def _read_manifest_rows(base: Path, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for shard in manifest["shards"]:
        path = base / shard["name"]
        if sha256_file(path) != shard["sha256"]:
            raise RuntimeError(f"manifest checksum mismatch: {path}")
        for row in pq.read_table(path).to_pylist():
            identity = row["id"]
            if identity in rows:
                raise RuntimeError(f"duplicate query identity in {path}: {identity}")
            rows[identity] = row
    return rows


def _read_sparse_query_rows(base: Path, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    files_by_path = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
    rows: dict[str, dict[str, Any]] = {}
    for name in manifest["shards"]["queries"]:
        path = base / name
        if sha256_file(path) != files_by_path[name]:
            raise RuntimeError(f"manifest checksum mismatch: {path}")
        for row in pq.read_table(path).to_pylist():
            identity = row["id"]
            if identity in rows:
                raise RuntimeError(f"duplicate query identity in {path}: {identity}")
            rows[identity] = row
    return rows


def load_dataset_snapshot(artifact_root: Path, dataset: str) -> DatasetSnapshot:
    base = artifact_root / "datasets" / dataset
    source_path = base / "source" / "manifest.json"
    dense_base = base / "dense" / DENSE_PROFILE
    sparse_base = base / "sparse" / SPARSE_PROFILE
    dense_path = dense_base / "queries-manifest.json"
    sparse_path = sparse_base / "manifest.json"

    source = read_json(source_path)
    dense_manifest = read_json(dense_path)
    sparse_manifest = read_json(sparse_path)

    if source.get("sampled") or source.get("rechunked"):
        raise RuntimeError(f"canonical runner rejects sampled/rechunked data: {dataset}")
    if dense_manifest.get("dimension") != 384 or dense_manifest.get("dtype") != "float32":
        raise RuntimeError(f"unexpected Dense representation: {dataset}")
    representation = sparse_manifest.get("representation", {})
    if representation.get("kind") != "bm25_impact_sparse_vector" or not representation.get(
        "nonnegative"
    ):
        raise RuntimeError(f"unexpected Sparse representation: {dataset}")

    dense_rows = _read_manifest_rows(dense_base, dense_manifest)
    sparse_rows = _read_sparse_query_rows(sparse_base, sparse_manifest)
    if dense_rows.keys() != sparse_rows.keys():
        missing_dense = sorted(sparse_rows.keys() - dense_rows.keys())[:5]
        missing_sparse = sorted(dense_rows.keys() - sparse_rows.keys())[:5]
        raise RuntimeError(
            f"Dense/Sparse query identity mismatch: dense={missing_dense}, sparse={missing_sparse}"
        )

    queries: list[QueryInput] = []
    for query_id, dense_row in dense_rows.items():
        sparse_row = sparse_rows[query_id]
        indices = list(sparse_row["indices"])
        values = list(sparse_row["values"])
        if len(indices) != len(values) or indices != sorted(set(indices)):
            raise RuntimeError(f"invalid canonical Sparse query: {query_id}")
        queries.append(
            QueryInput(
                query_id=query_id,
                dense=list(dense_row["vector"]),
                sparse_indices=indices,
                sparse_values=values,
            )
        )

    expected_queries = int(source["counts"]["queries"])
    if len(queries) != expected_queries:
        raise RuntimeError(
            f"query count mismatch for {dataset}: {len(queries)} != {expected_queries}"
        )
    dataset_id = str(source["dataset_id"])
    split = dataset_id.rsplit("/", 1)[-1] if "/" in dataset_id else "official"
    return DatasetSnapshot(
        dataset=dataset,
        split=split,
        document_count=int(source["counts"]["documents"]),
        queries=tuple(queries),
        source_manifest_sha256=sha256_file(source_path),
        dense_manifest_sha256=sha256_file(dense_path),
        sparse_manifest_sha256=sha256_file(sparse_path),
    )


def load_collection_snapshot(
    artifact_root: Path,
    snapshot: DatasetSnapshot,
    collection: str,
) -> CollectionSnapshot:
    base = artifact_root / "collections" / snapshot.dataset / CANONICAL_COLLECTION_FORMAT
    manifest_path = base / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("schema") != "canonical-qdrant-collection-snapshot-v1":
        raise RuntimeError(f"unsupported canonical Collection snapshot: {manifest_path}")
    expected_inputs = {
        "sourceManifestSha256": snapshot.source_manifest_sha256,
        "denseManifestSha256": snapshot.dense_manifest_sha256,
        "sparseManifestSha256": snapshot.sparse_manifest_sha256,
    }
    if manifest.get("inputs") != expected_inputs:
        raise RuntimeError("Collection snapshot inputs do not match canonical dataset manifests")
    if (
        manifest.get("dataset") != snapshot.dataset
        or manifest.get("collection") != collection
        or manifest.get("points") != snapshot.document_count
    ):
        raise RuntimeError("Collection snapshot dataset, collection, or point count mismatch")
    snapshot_spec = manifest.get("snapshot")
    if not isinstance(snapshot_spec, dict) or not isinstance(snapshot_spec.get("path"), str):
        raise RuntimeError("Collection snapshot manifest is missing snapshot.path")
    path = base / snapshot_spec["path"]
    expected_sha256 = snapshot_spec.get("sha256")
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise RuntimeError(f"Collection snapshot checksum mismatch: {path}")
    config_sha256 = manifest.get("collectionConfigSha256")
    if not isinstance(config_sha256, str) or not config_sha256:
        raise RuntimeError("Collection snapshot manifest is missing collectionConfigSha256")
    return CollectionSnapshot(
        path=path,
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        snapshot_sha256=expected_sha256,
        collection_config_sha256=config_sha256,
        points=snapshot.document_count,
    )
