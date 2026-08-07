#!/usr/bin/env python3
"""Verify checksums, CSV metadata, provenance links, and path hygiene."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path, PurePosixPath

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unsafe_path(value: str) -> bool:
    path = PurePosixPath(value)
    return (
        value.startswith(("/", "~"))
        or WINDOWS_ABSOLUTE_RE.match(value) is not None
        or ".." in path.parts
    )


def walk_strings(value: object):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_strings(item)


def verify_package(root: Path) -> list[str]:
    errors: list[str] = []
    manifest_path = root / "manifest.json"
    evidence_path = root / "source-evidence.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [str(error)]

    if manifest.get("schema") != "eahr-public-paper-results-v1":
        errors.append("unexpected package-manifest schema")
    if evidence.get("schema") != "eahr-public-source-evidence-v1":
        errors.append("unexpected source-evidence schema")

    for document_name, document in (("manifest", manifest), ("source evidence", evidence)):
        leaked = sorted({value for value in walk_strings(document) if unsafe_path(value)})
        if leaked:
            errors.append(f"{document_name} contains unsafe paths: {leaked}")

    evidence_record = manifest.get("source_evidence", {})
    if not SHA256_RE.fullmatch(str(evidence_record.get("sha256", ""))):
        errors.append("invalid source-evidence digest in manifest")
    elif sha256(evidence_path) != evidence_record["sha256"]:
        errors.append("source-evidence digest mismatch")

    evidence_files = evidence.get("files", [])
    evidence_hashes: set[str] = set()
    evidence_paths: set[str] = set()
    for item in evidence_files:
        digest = str(item.get("sha256", ""))
        relative_path = str(item.get("relative_path", ""))
        if not SHA256_RE.fullmatch(digest):
            errors.append(f"invalid evidence digest: {relative_path}")
        else:
            evidence_hashes.add(digest)
        if not relative_path or unsafe_path(relative_path):
            errors.append(f"invalid evidence path: {relative_path}")
        if relative_path in evidence_paths:
            errors.append(f"duplicate evidence path: {relative_path}")
        evidence_paths.add(relative_path)
    if evidence_record.get("files") != len(evidence_files):
        errors.append("source-evidence file count mismatch")

    provenance = evidence.get("execution_provenance", [])
    for record in provenance:
        if record.get("source") not in evidence_paths:
            errors.append(f"provenance source is not indexed: {record.get('source')}")
    flagged = any(
        record.get("dirty") is True or record.get("publicationEligible") is False
        for record in provenance
    )
    if flagged and not (root / "PROVENANCE_NOTES.md").is_file():
        errors.append("flagged execution provenance requires PROVENANCE_NOTES.md")

    declared_derived = manifest.get("derived_files", [])
    declared_paths = {str(item.get("path", "")) for item in declared_derived}
    actual_paths = {
        path.relative_to(root).as_posix() for path in (root / "derived").glob("*.csv")
    }
    if declared_paths != actual_paths:
        declared = sorted(declared_paths)
        actual = sorted(actual_paths)
        errors.append(
            f"derived file set mismatch: declared={declared} actual={actual}"
        )

    for item in declared_derived:
        relative_path = str(item.get("path", ""))
        if unsafe_path(relative_path):
            errors.append(f"invalid derived path: {relative_path}")
            continue
        path = root / relative_path
        if not path.is_file():
            errors.append(f"missing derived file: {relative_path}")
            continue
        if sha256(path) != item.get("sha256"):
            errors.append(f"derived digest mismatch: {relative_path}")
        if path.stat().st_size != item.get("bytes"):
            errors.append(f"derived byte count mismatch: {relative_path}")
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        if reader.fieldnames != item.get("columns"):
            errors.append(f"derived columns mismatch: {relative_path}")
        if len(rows) != item.get("rows"):
            errors.append(f"derived row count mismatch: {relative_path}")
        unknown_hashes = set(item.get("source_sha256", [])) - evidence_hashes
        if unknown_hashes:
            errors.append(f"unindexed source hashes in {relative_path}: {sorted(unknown_hashes)}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parents[1] / "paper-results",
    )
    args = parser.parse_args()
    errors = verify_package(args.root.resolve())
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print(f"verified public paper-results package: {args.root.resolve()}")


if __name__ == "__main__":
    main()
