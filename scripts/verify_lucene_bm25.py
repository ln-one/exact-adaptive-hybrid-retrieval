#!/usr/bin/env python3
"""Verify the immutable Lucene BM25 reference-index manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    root = args.artifact_root / "datasets" / args.dataset / "sparse" / "lucene-bm25"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    source = args.artifact_root / "datasets" / args.dataset / "source" / "documents.parquet"
    if sha256_file(source) != manifest["source"]["documents_sha256"]:
        raise RuntimeError("canonical document source checksum mismatch")
    actual_bytes = 0
    for entry in manifest["index"]["files"]:
        path = root / "index" / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise RuntimeError(f"missing or size-mismatched index file: {path}")
        if sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"index checksum mismatch: {path}")
        actual_bytes += entry["bytes"]
    if actual_bytes != manifest["index"]["bytes"]:
        raise RuntimeError("index byte total mismatch")
    print(json.dumps({"dataset": args.dataset, "bytes": actual_bytes}, sort_keys=True))


if __name__ == "__main__":
    main()
