#!/usr/bin/env python3
"""Build one Qdrant Collection snapshot directly from canonical representations."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
from canonical_runner.artifacts import load_dataset_snapshot, sha256_file
from canonical_runner.provenance import canonical_hash, git_is_dirty, git_revision
from canonical_runner.server import ManagedQdrant
from create_qdrant_collection import collection_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--system-repo", type=Path, required=True)
    parser.add_argument("--system-binary", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--optimizer-timeout", type=float, default=900.0)
    return parser.parse_args()


def _run_loader(script: Path, args: argparse.Namespace, url: str) -> None:
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--artifact-root",
            str(args.artifact_root),
            "--dataset",
            args.dataset,
            "--collection",
            args.collection,
            "--url",
            url,
            "--wait",
        ],
        check=True,
        cwd=script.parents[1],
    )


def _collection_info(client: httpx.Client, collection: str) -> dict[str, object]:
    response = client.get(f"/collections/{collection}")
    response.raise_for_status()
    result = response.json().get("result")
    if not isinstance(result, dict) or not isinstance(result.get("config"), dict):
        raise RuntimeError("Qdrant collection response is missing result.config")
    return result


def main() -> None:
    args = parse_args()
    bench_repo = Path(__file__).resolve().parents[1]
    if git_is_dirty(bench_repo) or git_is_dirty(args.system_repo):
        raise RuntimeError("canonical Collection snapshot builder refuses dirty repositories")
    snapshot = load_dataset_snapshot(args.artifact_root, args.dataset)
    output = args.artifact_root / "collections" / args.dataset / "qdrant-v1.18.2"
    if output.exists():
        raise FileExistsError(f"canonical Collection snapshot already exists: {output}")
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary Collection snapshot directory exists: {temporary}")
    temporary.mkdir(parents=True)

    scripts = Path(__file__).resolve().parent
    try:
        with (
            ManagedQdrant(
                binary=args.system_binary,
                system_repo=args.system_repo,
                collection=args.collection,
                snapshot=None,
            ) as server,
            httpx.Client(base_url=server.url, timeout=120.0) as client,
        ):
            schema = collection_schema(
                dense_vector_name="dense",
                sparse_vector_name="sparse",
                shards=args.shards,
                exact_rank_profile="dense_sparse_v1",
            )
            response = client.put(f"/collections/{args.collection}", json=schema)
            response.raise_for_status()
            _run_loader(scripts / "load_qdrant_dense.py", args, server.url)
            _run_loader(scripts / "load_qdrant_sparse.py", args, server.url)

            deadline = time.monotonic() + args.optimizer_timeout
            stable_since: float | None = None
            while time.monotonic() < deadline:
                info = _collection_info(client, args.collection)
                ready = (
                    info.get("status") == "green"
                    and info.get("points_count") == snapshot.document_count
                    and info.get("indexed_vectors_count") == snapshot.document_count
                )
                if ready:
                    stable_since = stable_since or time.monotonic()
                    if time.monotonic() - stable_since >= 2.0:
                        break
                else:
                    stable_since = None
                time.sleep(0.25)
            else:
                raise TimeoutError("canonical Collection did not reach a stable indexed state")

            info = _collection_info(client, args.collection)
            response = client.post(
                f"/collections/{args.collection}/snapshots",
                params={"wait": "true"},
            )
            response.raise_for_status()
            description = response.json().get("result")
            if not isinstance(description, dict) or not isinstance(description.get("name"), str):
                raise RuntimeError("Qdrant snapshot response is missing result.name")
            snapshot_name = description["name"]
            destination = temporary / snapshot_name
            with client.stream(
                "GET",
                f"/collections/{args.collection}/snapshots/{snapshot_name}",
            ) as download:
                download.raise_for_status()
                with destination.open("xb") as handle:
                    for chunk in download.iter_bytes():
                        handle.write(chunk)

            manifest = {
                "schema": "canonical-qdrant-collection-snapshot-v1",
                "dataset": args.dataset,
                "collection": args.collection,
                "points": snapshot.document_count,
                "inputs": {
                    "sourceManifestSha256": snapshot.source_manifest_sha256,
                    "denseManifestSha256": snapshot.dense_manifest_sha256,
                    "sparseManifestSha256": snapshot.sparse_manifest_sha256,
                },
                "collectionConfigSha256": canonical_hash(info["config"]),
                "snapshot": {
                    "path": snapshot_name,
                    "bytes": destination.stat().st_size,
                    "sha256": sha256_file(destination),
                },
                "builder": {
                    "benchCommit": git_revision(bench_repo),
                    "systemCommit": git_revision(args.system_repo),
                    "systemBinarySha256": server.binary_sha256,
                },
                "createdAtUtc": datetime.now(UTC).isoformat(),
            }
            (temporary / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
