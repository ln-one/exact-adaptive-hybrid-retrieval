#!/usr/bin/env python3
"""Capture a reproducible hardware profile without machine identifiers."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sysctl(name: str) -> str:
    return subprocess.run(
        ["sysctl", "-n", name],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"hardware manifest already exists: {args.output}")
    raw = subprocess.run(
        ["system_profiler", "SPHardwareDataType", "-json"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    hardware = json.loads(raw)["SPHardwareDataType"][0]
    manifest = {
        "schema": "canonical-hardware-v1",
        "hardwareProfile": args.profile,
        "architecture": platform.machine(),
        "model": hardware["machine_model"],
        "chip": hardware["chip_type"],
        "memoryBytes": int(_sysctl("hw.memsize")),
        "logicalCpuCount": int(_sysctl("hw.logicalcpu")),
        "physicalCpuCount": int(_sysctl("hw.physicalcpu")),
        "kernel": {
            "system": platform.system(),
            "release": platform.release(),
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
