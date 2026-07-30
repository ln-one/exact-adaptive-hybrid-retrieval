"""Reproducibility metadata without leaking machine or account identifiers."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def git_revision(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_is_dirty(repo: Path) -> bool:
    return bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


def source_tree_sha256(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def runner_source_sha256(repo: Path) -> str:
    sources = [repo / "scripts" / "run_canonical.py"]
    sources.extend((repo / "scripts" / "canonical_runner").glob("*.py"))
    sources.extend([repo / "pyproject.toml", repo / "uv.lock"])
    return source_tree_sha256(sources, repo)


def runtime_metadata(hardware_profile: str) -> dict[str, Any]:
    return {
        "hardwareProfile": hardware_profile,
        "architecture": platform.machine(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executableKind": Path(sys.executable).name,
    }


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def verify_system_build_manifest(
    path: Path,
    *,
    binary_sha256: str,
    system_commit: str,
) -> str:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema") != "canonical-qdrant-build-v1":
        raise RuntimeError(f"unsupported canonical system build manifest: {path}")
    if manifest.get("systemCommit") != system_commit:
        raise RuntimeError("system build manifest commit does not match the frozen system commit")
    binary = manifest.get("binary")
    build = manifest.get("build")
    if not isinstance(binary, dict) or binary.get("sha256") != binary_sha256:
        raise RuntimeError("system build manifest does not match the managed binary")
    if (
        not isinstance(build, dict)
        or build.get("profile") != "release"
        or not isinstance(build.get("command"), list)
        or "canonical-bench" not in build.get("features", [])
        or not isinstance(build.get("rustflags"), str)
    ):
        raise RuntimeError("system build manifest is missing the frozen release configuration")
    return hashlib.sha256(path.read_bytes()).hexdigest()
