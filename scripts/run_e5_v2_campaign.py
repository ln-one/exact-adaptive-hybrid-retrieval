#!/usr/bin/env python3
"""Run or safely resume all immutable shards in one E5-v2 campaign."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from canonical_runner.e5_v2 import E5V2ShardConfig, load_campaign, run_e5_v2_shard
from canonical_runner.server import sha256_file
from canonical_runner.validation import validate_log


def _scheduled(campaign: dict[str, object]) -> list[tuple[int, int]]:
    return [
        (round_spec["round"], shard["shard"])
        for round_spec in campaign["rounds"]
        for shard in round_spec["shards"]
    ]


def _assert_no_external_qdrant() -> None:
    if shutil.which("pgrep") is None:
        return
    completed = subprocess.run(
        ["pgrep", "-x", "qdrant"], check=False, capture_output=True, text=True
    )
    if completed.returncode == 0 and completed.stdout.strip():
        raise RuntimeError("another Qdrant process is already running")


@contextmanager
def _inhibit_sleep() -> Iterator[None]:
    process: subprocess.Popen[bytes] | None = None
    if shutil.which("caffeinate") is not None:
        process = subprocess.Popen(
            ["caffeinate", "-dimsu", "-w", str(os.getpid())],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    try:
        yield
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=10)


def _validate_existing(
    path: Path, campaign: dict[str, object], campaign_manifest: Path
) -> None:
    validate_log(path, require_clean=not campaign["dirty"])
    first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    if (
        first.get("campaignId") != campaign["campaignId"]
        or first.get("campaignManifestSha256")
        != sha256_file(campaign_manifest)
    ):
        raise RuntimeError(f"existing shard belongs to another campaign: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--failed-dir", type=Path, required=True)
    parser.add_argument(
        "--bench-repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--system-repo", type=Path, required=True)
    parser.add_argument("--system-binary", type=Path, required=True)
    parser.add_argument("--system-build-manifest", type=Path, required=True)
    parser.add_argument("--hardware-manifest", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    campaign = load_campaign(args.campaign_manifest)
    if sha256_file(Path(__file__)) != campaign["campaignDriverSha256"]:
        raise RuntimeError("campaign driver differs from the frozen campaign")
    if campaign["dirty"] and not args.allow_dirty:
        raise RuntimeError("dirty E5-v2 campaign requires --allow-dirty")
    _assert_no_external_qdrant()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.failed_dir.mkdir(parents=True, exist_ok=True)

    completed: list[str] = []
    with _inhibit_sleep():
        for round_number, shard_number in _scheduled(campaign):
            output = args.output_dir / f"r{round_number:02d}-s{shard_number:02d}.jsonl"
            if output.exists():
                _validate_existing(output, campaign, args.campaign_manifest)
                completed.append(output.name)
                continue
            summary = run_e5_v2_shard(
                E5V2ShardConfig(
                    artifact_root=args.artifact_root,
                    campaign_manifest=args.campaign_manifest,
                    round_number=round_number,
                    shard_number=shard_number,
                    output=output,
                    failed_dir=args.failed_dir,
                    bench_repo=args.bench_repo,
                    system_repo=args.system_repo,
                    system_binary=args.system_binary,
                    system_build_manifest=args.system_build_manifest,
                    hardware_manifest=args.hardware_manifest,
                    allow_dirty=args.allow_dirty,
                )
            )
            validate_log(output, require_clean=not campaign["dirty"])
            if summary["okQueries"] != summary["uniqueQueries"]:
                raise RuntimeError(f"E5-v2 shard did not complete cleanly: {output}")
            completed.append(output.name)
    print(
        json.dumps(
            {
                "campaignId": campaign["campaignId"],
                "completedShards": completed,
                "count": len(completed),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
