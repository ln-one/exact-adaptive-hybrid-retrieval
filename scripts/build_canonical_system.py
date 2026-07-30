#!/usr/bin/env python3
"""Build and attest the exact Qdrant executable used by canonical experiments."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from canonical_runner.provenance import git_is_dirty, git_revision
from canonical_runner.server import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--toolchain", default="nightly")
    parser.add_argument("--features", default="canonical-bench")
    parser.add_argument("--binary-name", default="qdrant")
    parser.add_argument("--rustflags", default="")
    return parser.parse_args()


def _version(command: list[str], cwd: Path) -> str:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    args = parse_args()
    repo = args.system_repo.resolve(strict=True)
    if git_is_dirty(repo):
        raise RuntimeError("canonical system builder refuses a dirty repository")
    if args.output.exists():
        raise FileExistsError(f"build manifest already exists: {args.output}")

    command = [
        "cargo",
        f"+{args.toolchain}",
        "build",
        "--release",
        "--bin",
        args.binary_name,
    ]
    if args.features:
        command.extend(["--features", args.features])
    environment = os.environ.copy()
    environment["RUSTFLAGS"] = args.rustflags
    subprocess.run(command, cwd=repo, env=environment, check=True)

    binary = repo / "target" / "release" / args.binary_name
    if not binary.is_file():
        raise RuntimeError(f"canonical build did not produce the expected binary: {binary}")
    manifest = {
        "schema": "canonical-qdrant-build-v1",
        "systemCommit": git_revision(repo),
        "binary": {
            "name": args.binary_name,
            "sha256": sha256_file(binary),
        },
        "build": {
            "command": command,
            "toolchain": args.toolchain,
            "features": args.features.split(",") if args.features else [],
            "rustflags": args.rustflags,
            "rustc": _version(["rustc", f"+{args.toolchain}", "-Vv"], repo),
            "cargo": _version(["cargo", f"+{args.toolchain}", "-V"], repo),
            "target": platform.machine(),
            "profile": "release",
        },
        "createdAtUtc": datetime.now(UTC).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
