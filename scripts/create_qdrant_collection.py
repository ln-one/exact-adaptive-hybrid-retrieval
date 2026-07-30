#!/usr/bin/env python3
"""Create the minimal named-vector schema consumed by canonical ED-WRRF inputs."""

from __future__ import annotations

import argparse
import json

import httpx


def collection_schema(
    *,
    dense_vector_name: str,
    sparse_vector_name: str,
    shards: int,
    exact_rank_profile: str,
) -> dict[str, object]:
    if shards <= 0:
        raise ValueError("shards must be positive")
    if exact_rank_profile not in {"disabled", "dense_sparse_v1"}:
        raise ValueError(f"unsupported exact-rank profile: {exact_rank_profile}")
    return {
        "vectors": {dense_vector_name: {"size": 384, "distance": "Cosine"}},
        "sparse_vectors": {sparse_vector_name: {"index": {"on_disk": False}}},
        "shard_number": shards,
        "exact_rank_config": {"profile": exact_rank_profile},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--dense-vector-name", default="dense")
    parser.add_argument("--sparse-vector-name", default="sparse")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument(
        "--exact-rank-profile",
        choices=("disabled", "dense_sparse_v1"),
        default="disabled",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = collection_schema(
        dense_vector_name=args.dense_vector_name,
        sparse_vector_name=args.sparse_vector_name,
        shards=args.shards,
        exact_rank_profile=args.exact_rank_profile,
    )
    with httpx.Client(base_url=args.url.rstrip("/"), timeout=120.0) as client:
        response = client.put(f"/collections/{args.collection}", json=payload)
        response.raise_for_status()
    print(json.dumps({"collection": args.collection, "schema": payload}, sort_keys=True))


if __name__ == "__main__":
    main()
