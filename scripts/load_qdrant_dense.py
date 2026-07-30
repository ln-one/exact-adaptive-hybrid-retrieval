#!/usr/bin/env python3
"""Stream one verified canonical Dense artifact into a Qdrant collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from pathlib import Path

import httpx
import numpy as np
import pyarrow.parquet as pq
from dataset_gate import assert_dataset_eligible

POINT_NAMESPACE = uuid.UUID("a0541185-c167-51be-9665-4c5e739d75d3")


def point_id(dataset: str, external_id: str) -> str:
    """Stable UUID; original identifier remains available in payload."""
    return str(uuid.uuid5(POINT_NAMESPACE, f"{dataset}\x1f{external_id}"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--vector-name", default="dense")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--wait", action="store_true")
    return parser.parse_args()


def verify_manifest(base: Path) -> dict[str, object]:
    manifest = json.loads((base / "documents-manifest.json").read_text())
    if manifest["dtype"] != "float32" or manifest["normalization"] != "l2":
        raise RuntimeError("only canonical normalized float32 Dense artifacts are accepted")
    if manifest["dimension"] != 384:
        raise RuntimeError(f"unexpected Dense dimension: {manifest['dimension']}")
    for shard in manifest["shards"]:
        path = base / shard["name"]
        digest = hashlib.sha256()
        if not path.is_file():
            raise RuntimeError(f"Dense artifact is missing: {path}")
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != shard["sha256"]:
            raise RuntimeError(f"Dense artifact checksum mismatch: {path}")
    return manifest


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    assert_dataset_eligible(args.dataset)
    base = args.artifact_root / "datasets" / args.dataset / "dense" / "bge-small-en-v1.5-f32"
    manifest = verify_manifest(base)
    client = httpx.Client(base_url=args.url.rstrip("/"), timeout=120.0)
    sent = 0
    try:
        for spec in manifest["shards"]:
            shard = base / spec["name"]
            for record_batch in pq.ParquetFile(shard).iter_batches(
                batch_size=args.batch_size, columns=["id", "vector"]
            ):
                values = record_batch.to_pydict()
                vectors = np.asarray(values["vector"], dtype=np.float32)
                if (
                    vectors.shape != (len(values["id"]), manifest["dimension"])
                    or not np.isfinite(vectors).all()
                ):
                    raise RuntimeError(f"invalid vector batch: {shard.name}")
                payload = {
                    "points": [
                        {
                            "id": point_id(args.dataset, external_id),
                            "vector": {args.vector_name: vector.tolist()},
                            "payload": {"external_id": external_id, "dataset": args.dataset},
                        }
                        for external_id, vector in zip(values["id"], vectors, strict=True)
                    ]
                }
                response = client.put(
                    f"/collections/{args.collection}/points",
                    params={"wait": str(args.wait).lower()},
                    json=payload,
                )
                response.raise_for_status()
                sent += len(payload["points"])
                if sent % 10_000 == 0:
                    print(json.dumps({"dataset": args.dataset, "sent": sent}), flush=True)
    finally:
        client.close()
    print(
        json.dumps(
            {"collection": args.collection, "dataset": args.dataset, "points": sent}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
