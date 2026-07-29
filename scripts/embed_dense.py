#!/usr/bin/env python3
"""Create recoverable canonical float32 Dense embeddings from an exported corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer


MODEL_ID = "BAAI/bge-small-en-v1.5"
MODEL_REVISION = "01d3c3cd65ac9dc6bd0d702ed913366e7931097b"
DIMENSION = 384
MAX_LENGTH = 512
SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--kind", choices=("documents", "queries"), required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--shard-rows", type=int, default=100_000)
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps"))
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_shard(path: Path, ids: list[str], matrix: np.ndarray) -> None:
    if matrix.dtype != np.float32 or matrix.shape != (len(ids), DIMENSION):
        raise ValueError(f"invalid embedding matrix {matrix.shape} {matrix.dtype}")
    values = pa.array(matrix.reshape(-1), type=pa.float32())
    vectors = pa.FixedSizeListArray.from_arrays(values, DIMENSION)
    table = pa.table({"id": pa.array(ids, type=pa.string()), "vector": vectors})
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temporary, compression="zstd", row_group_size=len(ids))
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.shard_rows <= 0:
        raise ValueError("batch and shard sizes must be positive")
    source = args.artifact_root / "datasets" / args.dataset / "source" / f"{args.kind}.parquet"
    if not source.exists():
        raise FileNotFoundError(f"missing canonical source: {source}")
    output = args.artifact_root / "datasets" / args.dataset / "dense" / "bge-small-en-v1.5-f32"
    output.mkdir(parents=True, exist_ok=True)
    profile = output / "profile.json"
    profile_payload = {
        "schema_version": SCHEMA_VERSION,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "dimension": DIMENSION,
        "max_length": MAX_LENGTH,
        "dtype": "float32",
        "normalization": "l2",
        "backend": "sentence-transformers",
        "device": args.device,
    }
    if profile.exists():
        if json.loads(profile.read_text()) != profile_payload:
            raise RuntimeError("existing Dense profile differs; refusing to mix artifacts")
    else:
        profile.write_text(json.dumps(profile_payload, indent=2, sort_keys=True) + "\n")

    cache_folder = args.artifact_root / "model-cache" / "huggingface"
    cache_folder.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_folder=str(cache_folder),
        device=args.device,
        trust_remote_code=False,
    )
    model.max_seq_length = MAX_LENGTH
    source_file = pq.ParquetFile(source)
    shard_index = 0
    ids: list[str] = []
    texts: list[str] = []
    total = 0

    def flush() -> None:
        nonlocal shard_index, total
        if not ids:
            return
        shard = output / f"{args.kind}-{shard_index:06d}.parquet"
        if shard.exists():
            table = pq.read_table(shard, columns=["id"])
            if table.num_rows != len(ids) or table.column("id").to_pylist() != ids:
                raise RuntimeError(f"existing shard differs: {shard}")
        else:
            vectors = model.encode(
                texts,
                batch_size=args.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
                precision="float32",
            ).astype(np.float32, copy=False)
            write_shard(shard, ids, vectors)
        total += len(ids)
        shard_index += 1
        ids.clear()
        texts.clear()
        if shard_index % 10 == 0:
            print(f"{args.dataset}/{args.kind}: {total:,} vectors", flush=True)

    for batch in source_file.iter_batches(batch_size=args.shard_rows, columns=["id", "text"]):
        mapping = batch.to_pydict()
        ids.extend(mapping["id"])
        texts.extend(mapping["text"])
        flush()
    flush()
    shards = sorted(output.glob(f"{args.kind}-*.parquet"))
    manifest = {
        **profile_payload,
        "kind": args.kind,
        "source_sha256": sha256_file(source),
        "vectors": total,
        "shards": [
            {"name": shard.name, "sha256": sha256_file(shard), "rows": pq.ParquetFile(shard).metadata.num_rows}
            for shard in shards
        ],
    }
    temporary = output / f"{args.kind}-manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(output / f"{args.kind}-manifest.json")
    print(json.dumps({"dataset": args.dataset, "kind": args.kind, "vectors": total}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("cancelled", file=sys.stderr)
        raise SystemExit(130)
