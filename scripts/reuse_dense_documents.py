#!/usr/bin/env python3
"""Hard-link verified Dense documents when two official tasks share one corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--from-dataset", required=True)
    parser.add_argument("--to-dataset", required=True)
    args = parser.parse_args()
    model_dir = Path("dense/bge-small-en-v1.5-f32")
    source_base = args.artifact_root / "datasets" / args.from_dataset / model_dir
    target_base = args.artifact_root / "datasets" / args.to_dataset / model_dir
    source_manifest = json.loads((source_base / "documents-manifest.json").read_text(encoding="utf-8"))
    source_documents = args.artifact_root / "datasets" / args.from_dataset / "source" / "documents.parquet"
    target_documents = args.artifact_root / "datasets" / args.to_dataset / "source" / "documents.parquet"
    if sha256_file(source_documents) != sha256_file(target_documents):
        raise RuntimeError("official document corpora differ; refusing Dense reuse")
    if source_manifest["source_sha256"] != sha256_file(target_documents):
        raise RuntimeError("source Dense manifest does not match target corpus")
    target_base.mkdir(parents=True, exist_ok=True)
    for shard in source_manifest["shards"]:
        source, target = source_base / shard["name"], target_base / shard["name"]
        if target.exists():
            if sha256_file(target) != shard["sha256"]:
                raise RuntimeError(f"existing target shard differs: {target}")
        else:
            os.link(source, target)
    target_manifest = target_base / "documents-manifest.json"
    if target_manifest.exists():
        existing = json.loads(target_manifest.read_text(encoding="utf-8"))
        if existing != source_manifest:
            raise RuntimeError("existing target manifest differs")
    else:
        target_manifest.write_text(json.dumps(source_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"from": args.from_dataset, "to": args.to_dataset, "vectors": source_manifest["vectors"]}))


if __name__ == "__main__":
    main()
