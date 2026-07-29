#!/usr/bin/env python3
"""Materialize and fingerprint an immutable Hugging Face model revision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    destination = args.artifact_root / "models" / args.name / args.revision
    snapshot_download(
        repo_id=args.repository,
        revision=args.revision,
        local_dir=destination,
    )

    files = {
        str(path.relative_to(destination)): sha256_file(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file() and ".cache" not in path.parts
    }
    payload = {
        "schema_version": 1,
        "repository": args.repository,
        "revision": args.revision,
        "files": files,
    }
    (destination / "model-manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"path": str(destination), "files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
