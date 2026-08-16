#!/usr/bin/env python3
"""Fetch and audit frozen BEIR archives without touching the Qdrant worktree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tomllib
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


MIB = 1024 * 1024
GIB = 1024 * MIB


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets.toml",
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--tier", action="append")
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--download-only", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(8 * MIB):
            hasher.update(chunk)
    return hasher.hexdigest()


def selected_datasets(config: dict, args: argparse.Namespace) -> list[tuple[str, dict]]:
    requested = set(args.dataset or [])
    tiers = set(args.tier or ["core"])
    selected = []
    for name, spec in config["datasets"].items():
        if "archive_url" not in spec:
            continue
        if requested and name not in requested:
            continue
        if not requested and spec["tier"] not in tiers:
            continue
        selected.append((name, spec))
    unknown = requested - {name for name, _ in selected}
    if unknown:
        raise SystemExit(f"unknown or non-archive datasets: {sorted(unknown)}")
    return selected


def assert_capacity(root: Path, required_bytes: int, minimum_free_after_gib: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    reserve = minimum_free_after_gib * GIB
    if free - required_bytes < reserve:
        raise SystemExit(
            f"capacity gate failed: free={free / GIB:.1f} GiB, "
            f"required={required_bytes / GIB:.1f} GiB, "
            f"required reserve={minimum_free_after_gib} GiB"
        )


def fetch(url: str, destination: Path, expected_bytes: int) -> None:
    if destination.exists() and destination.stat().st_size == expected_bytes:
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(
        url, headers={"User-Agent": "exact-adaptive-hybrid-retrieval/0.1"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        with temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=8 * MIB)
            output.flush()
            os.fsync(output.fileno())
    if temporary.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"download size mismatch for {url}: "
            f"{temporary.stat().st_size} != {expected_bytes}"
        )
    temporary.replace(destination)


def safe_extract(archive: Path, destination: Path) -> tuple[int, int]:
    marker = destination / ".stratumind-extracted.json"
    if marker.exists():
        metadata = json.loads(marker.read_text())
        return metadata["files"], metadata["uncompressed_bytes"]

    temporary = destination.with_name(destination.name + ".extracting")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    file_count = 0
    uncompressed_bytes = 0
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            relative = PurePosixPath(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"unsafe zip member: {member.filename}")
            file_count += not member.is_dir()
            uncompressed_bytes += member.file_size
        source.extractall(temporary)

    marker_payload = {
        "archive": archive.name,
        "archive_sha256": digest(archive, "sha256"),
        "files": file_count,
        "uncompressed_bytes": uncompressed_bytes,
    }
    (temporary / marker.name).write_text(
        json.dumps(marker_payload, indent=2, sort_keys=True) + "\n"
    )
    if destination.exists():
        shutil.rmtree(destination)
    temporary.replace(destination)
    return file_count, uncompressed_bytes


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    configured_root = Path(config["storage"]["default_root"])
    if not configured_root.is_absolute():
        configured_root = args.config.resolve().parent / configured_root
    root = (args.root or configured_root).resolve()
    archives = root / "archives"
    raw = root / "raw"
    manifests = root / "manifests"
    archives.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)

    datasets = selected_datasets(config, args)
    download_bytes = sum(spec["archive_bytes"] for _, spec in datasets)
    assert_capacity(
        root,
        required_bytes=download_bytes * 3,
        minimum_free_after_gib=config["storage"]["minimum_free_after_gib"],
    )

    for name, spec in datasets:
        archive = archives / f"{name}.zip"
        print(f"[{name}] downloading {spec['archive_bytes'] / MIB:.1f} MiB", flush=True)
        fetch(spec["archive_url"], archive, spec["archive_bytes"])
        actual_md5 = digest(archive, "md5")
        if actual_md5 != spec["archive_md5"]:
            raise RuntimeError(f"{name}: MD5 mismatch {actual_md5}")
        print(f"[{name}] archive verified", flush=True)

        file_count = 0
        uncompressed_bytes = 0
        if not args.download_only:
            file_count, uncompressed_bytes = safe_extract(archive, raw / name)
            print(
                f"[{name}] extracted {file_count} files, "
                f"{uncompressed_bytes / MIB:.1f} MiB",
                flush=True,
            )

        manifest = {
            "schema_version": 1,
            "dataset": name,
            "loader_id": spec["loader_id"],
            "split": spec["split"],
            "license_status": spec["license_status"],
            "expected_documents": spec["documents"],
            "expected_queries": spec["queries"],
            "source": {
                "url": spec["archive_url"],
                "bytes": archive.stat().st_size,
                "md5": actual_md5,
                "sha256": digest(archive, "sha256"),
            },
            "extraction": {
                "performed": not args.download_only,
                "files": file_count,
                "uncompressed_bytes": uncompressed_bytes,
            },
        }
        temporary = manifests / f"{name}.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary.replace(manifests / f"{name}.json")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("cancelled", file=sys.stderr)
        raise SystemExit(130)
