#!/usr/bin/env python3
"""Create the minimal named-vector schema consumed by canonical ED-WRRF inputs."""

from __future__ import annotations

import argparse
import json

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--dense-vector-name", default="dense")
    parser.add_argument("--sparse-vector-name", default="sparse")
    parser.add_argument("--shards", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.shards <= 0:
        raise ValueError("--shards must be positive")
    payload = {
        "vectors": {args.dense_vector_name: {"size": 384, "distance": "Cosine"}},
        "sparse_vectors": {args.sparse_vector_name: {"index": {"on_disk": False}}},
        "shard_number": args.shards,
    }
    with httpx.Client(base_url=args.url.rstrip("/"), timeout=120.0) as client:
        response = client.put(f"/collections/{args.collection}", json=payload)
        response.raise_for_status()
    print(json.dumps({"collection": args.collection, "schema": payload}, sort_keys=True))


if __name__ == "__main__":
    main()
