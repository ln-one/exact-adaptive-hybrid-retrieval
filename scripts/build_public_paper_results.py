#!/usr/bin/env python3
"""Build the small, public EAHR paper-results package from frozen evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path

PAPER_TITLE = "Exact Adaptive Hybrid Retrieval Without Fixed Top-L Cutoffs"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

DERIVED_FILES = {
    "figures/data/results-target-validity-static.csv": (
        "fixed-depth-effectiveness.csv",
        "Figure 2 and static retrieval-effectiveness claims",
    ),
    "figures/data/fig_eahr_chronological.csv": (
        "temporal-transfer-summary.csv",
        "Temporal fixed-depth transfer and aggregate adaptive access",
    ),
    "figures/data/fig_eahr_chronological_per_query.csv": (
        "query-snapshot-access.csv",
        "Figure 3 request-level access variation",
    ),
    "figures/data/fig_eahr_query_snapshot_variance.csv": (
        "query-snapshot-variance.csv",
        "Query, snapshot, and interaction variance decomposition",
    ),
    "tables/generated/results-e1-correctness.csv": (
        "ordered-correctness.csv",
        "Ordered-result parity and failure counts",
    ),
    "tables/generated/results-e2-efficiency.csv": (
        "aggregate-efficiency.csv",
        "Figure 4 aggregate latency and access comparisons",
    ),
    "figures/data/results-e2-scale-per-query.csv": (
        "per-query-latency.csv",
        "Query-level latency ratios and tail cases",
    ),
    "figures/data/results-e3-fixed-prefix.csv": (
        "fixed-prefix-agreement.csv",
        "Fixed-depth membership and ordered agreement",
    ),
    "figures/data/results-e4-regimes.csv": (
        "controlled-ranking-regimes.csv",
        "Controlled correlation-regime access behavior",
    ),
    "tables/generated/results-e5-producers.csv": (
        "rank-generator-ablation.csv",
        "Table 1 PVS and PBM rank-generator comparisons",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def relative_evidence_path(path_value: str, artifact_root: Path) -> str:
    path = Path(path_value).resolve()
    try:
        relative = path.relative_to(artifact_root.resolve())
    except ValueError as error:
        raise ValueError(f"evidence path is outside artifact root: {path}") from error
    return relative.as_posix()


def evidence_entry(
    *,
    path: Path,
    declared_sha256: str,
    artifact_root: Path,
    family: str,
    verify_source: bool,
) -> dict[str, str]:
    if not SHA256_RE.fullmatch(declared_sha256):
        raise ValueError(f"invalid SHA-256 for {path}")
    if verify_source and sha256(path) != declared_sha256:
        raise ValueError(f"source hash mismatch: {path}")
    return {
        "family": family,
        "relative_path": relative_evidence_path(str(path), artifact_root),
        "sha256": declared_sha256,
    }


def collect_manifest_triplet(
    *,
    manifest_path: Path,
    expected_manifest_sha256: str,
    artifact_root: Path,
    family: str,
    verify_sources: bool,
) -> list[dict[str, str]]:
    entries = [
        evidence_entry(
            path=manifest_path,
            declared_sha256=expected_manifest_sha256,
            artifact_root=artifact_root,
            family=family,
            verify_source=True,
        )
    ]
    manifest = load_json(manifest_path)
    for role in ("run", "summary"):
        item = manifest[role]
        entries.append(
            evidence_entry(
                path=manifest_path.parent / item["path"],
                declared_sha256=item["sha256"],
                artifact_root=artifact_root,
                family=family,
                verify_source=verify_sources,
            )
        )
    return entries


def collect_e5_campaign_evidence(
    aggregate_path: Path, artifact_root: Path, verify_sources: bool
) -> list[dict[str, str]]:
    aggregate = load_json(aggregate_path)
    campaign_path = aggregate_path.parent / "campaign.json"
    family = f"rank-generator-ablation:{aggregate['dataset']}"
    entries = [
        evidence_entry(
            path=campaign_path,
            declared_sha256=aggregate["campaignManifestSha256"],
            artifact_root=artifact_root,
            family=family,
            verify_source=True,
        )
    ]
    campaign = load_json(campaign_path)
    for round_spec in campaign["rounds"]:
        for shard in round_spec["shards"]:
            relative = Path("shards") / (
                f"r{round_spec['round']:02d}-s{shard['shard']:02d}.jsonl"
            )
            path = aggregate_path.parent / relative
            declared = path.with_suffix(path.suffix + ".sha256").read_text().split()[0]
            entries.append(
                evidence_entry(
                    path=path,
                    declared_sha256=declared,
                    artifact_root=artifact_root,
                    family=family,
                    verify_source=verify_sources,
                )
            )
    return entries


def collect_standard_evidence(
    source_manifest: dict[str, object], artifact_root: Path, verify_sources: bool
) -> list[dict[str, str]]:
    evidence: list[dict[str, str]] = []
    for source in source_manifest["sources"]:
        path = Path(source["path"])
        evidence.append(
            evidence_entry(
                path=path,
                declared_sha256=source["sha256"],
                artifact_root=artifact_root,
                family="canonical-run-evidence",
                verify_source=verify_sources,
            )
        )
        if "e5-v2-counterbalanced" in path.as_posix() and path.name == "aggregate.json":
            evidence.extend(collect_e5_campaign_evidence(path, artifact_root, verify_sources))

    for dataset in source_manifest["targetValiditySources"]:
        manifest = dataset["manifest"]
        family = f"target-validity:{dataset['dataset']}"
        evidence.append(
            evidence_entry(
                path=Path(manifest["path"]),
                declared_sha256=manifest["sha256"],
                artifact_root=artifact_root,
                family=family,
                verify_source=verify_sources,
            )
        )
        for source in dataset["files"]:
            evidence.append(
                evidence_entry(
                    path=Path(source["path"]),
                    declared_sha256=source["sha256"],
                    artifact_root=artifact_root,
                    family=family,
                    verify_source=verify_sources,
                )
            )
        run_manifest_path = (
            artifact_root
            / "experiments/target-validity/static-v3"
            / dataset["dataset"]
            / "manifest.json"
        )
        evidence.extend(
            collect_manifest_triplet(
                manifest_path=run_manifest_path,
                expected_manifest_sha256=dataset["sourceRunManifestSha256"],
                artifact_root=artifact_root,
                family=f"target-validity-run:{dataset['dataset']}",
                verify_sources=verify_sources,
            )
        )
    return evidence


def collect_chronological_evidence(
    artifact_root: Path, verify_sources: bool
) -> list[dict[str, str]]:
    base = artifact_root / "experiments/target-validity"
    quality = base / "chronological-v1-aggregate"
    depth = base / "chronological-v1-eahr-depth-aggregate"
    quality_manifest_path = quality / "manifest.json"
    depth_manifest_path = depth / "manifest.json"
    quality_manifest = load_json(quality_manifest_path)
    depth_manifest = load_json(depth_manifest_path)

    entries = [
        evidence_entry(
            path=quality_manifest_path,
            declared_sha256=sha256(quality_manifest_path),
            artifact_root=artifact_root,
            family="chronological-target-validity",
            verify_source=False,
        ),
        evidence_entry(
            path=quality / "aggregate.json",
            declared_sha256=quality_manifest["aggregateSha256"],
            artifact_root=artifact_root,
            family="chronological-target-validity",
            verify_source=verify_sources,
        ),
        evidence_entry(
            path=quality / "per-query.csv",
            declared_sha256=quality_manifest["perQuerySha256"],
            artifact_root=artifact_root,
            family="chronological-target-validity",
            verify_source=verify_sources,
        ),
        evidence_entry(
            path=depth_manifest_path,
            declared_sha256=sha256(depth_manifest_path),
            artifact_root=artifact_root,
            family="chronological-eahr-depth",
            verify_source=False,
        ),
    ]
    for item in depth_manifest["files"]:
        entries.append(
            evidence_entry(
                path=depth / item["path"],
                declared_sha256=item["sha256"],
                artifact_root=artifact_root,
                family="chronological-eahr-depth",
                verify_source=verify_sources,
            )
        )
    for round_id, declared in quality_manifest["sourceRunManifestSha256"].items():
        entries.extend(
            collect_manifest_triplet(
                manifest_path=base / f"chronological-v1/round-{round_id}/manifest.json",
                expected_manifest_sha256=declared,
                artifact_root=artifact_root,
                family=f"chronological-target-validity-run:R{round_id}",
                verify_sources=verify_sources,
            )
        )
    for round_id, declared in depth_manifest["sourceRunManifestSha256"].items():
        entries.extend(
            collect_manifest_triplet(
                manifest_path=(
                    base / f"chronological-v1-eahr-depth/round-{round_id}/manifest.json"
                ),
                expected_manifest_sha256=declared,
                artifact_root=artifact_root,
                family=f"chronological-eahr-depth-run:R{round_id}",
                verify_sources=verify_sources,
            )
        )
    return entries


def extract_provenance(
    evidence: list[dict[str, str]], artifact_root: Path
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    wanted = {
        "dataset",
        "experiment",
        "method",
        "benchCommit",
        "runnerSourceSha256",
        "systemCommit",
        "systemArtifact",
        "binarySha256",
        "dirty",
        "publicationEligible",
        "campaignDriverSha256",
        "aggregateScriptSha256",
    }
    for item in evidence:
        relative = item["relative_path"]
        path = artifact_root / relative
        if path.name == "campaign.json" or path.name == "run.json":
            value = load_json(path)
        elif relative.startswith("logs/e2/") and path.suffix == ".jsonl":
            value = json.loads(path.open(encoding="utf-8").readline())
        else:
            continue
        record = {"source": relative}
        record.update({key: value[key] for key in wanted if key in value})
        records.append(record)
    return sorted(records, key=lambda item: item["source"])


def csv_metadata(path: Path) -> tuple[list[str], int, set[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV has no data rows: {path}")
    source_hashes = {
        value
        for row in rows
        for column, value in row.items()
        if column.lower().endswith("sha256") and value
    }
    invalid = sorted(value for value in source_hashes if not SHA256_RE.fullmatch(value))
    if invalid:
        raise ValueError(f"invalid source hashes in {path}: {invalid}")
    return reader.fieldnames, len(rows), source_hashes


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_package(
    *, paper_root: Path, artifact_root: Path, output: Path, verify_sources: bool
) -> None:
    paper_root = paper_root.resolve()
    artifact_root = artifact_root.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    derived_dir = output / "derived"
    derived_dir.mkdir(exist_ok=True)

    expected_names = {destination for destination, _ in DERIVED_FILES.values()}
    for stale in derived_dir.glob("*.csv"):
        if stale.name not in expected_names:
            stale.unlink()

    source_manifest_path = paper_root / "figures/data/results-source-manifest.json"
    source_manifest = load_json(source_manifest_path)
    if source_manifest.get("schema") != "eahr-paper-results-source-manifest-v2":
        raise ValueError(f"unexpected source-manifest schema: {source_manifest_path}")

    evidence = collect_standard_evidence(source_manifest, artifact_root, verify_sources)
    evidence.extend(collect_chronological_evidence(artifact_root, verify_sources))
    evidence.sort(key=lambda item: (item["family"], item["relative_path"]))
    if len({item["relative_path"] for item in evidence}) != len(evidence):
        raise ValueError("duplicate source-evidence paths")

    evidence_document = {
        "schema": "eahr-public-source-evidence-v1",
        "access": "not redistributed in this package",
        "access_reason": (
            "Raw logs, indexes, model artifacts, and third-party corpora are excluded because of "
            "size and upstream redistribution terms. Relative logical paths and SHA-256 digests "
            "preserve the link to the frozen local evidence archive."
        ),
        "files": evidence,
        "execution_provenance": extract_provenance(evidence, artifact_root),
    }
    evidence_path = output / "source-evidence.json"
    write_json(evidence_path, evidence_document)

    evidence_hashes = {item["sha256"] for item in evidence}
    derived: list[dict[str, object]] = []
    referenced_source_hashes: set[str] = set()
    for source_relative, (destination_name, supports) in DERIVED_FILES.items():
        source = paper_root / source_relative
        destination = derived_dir / destination_name
        shutil.copyfile(source, destination)
        columns, row_count, source_hashes = csv_metadata(destination)
        referenced_source_hashes.update(source_hashes)
        derived.append(
            {
                "path": f"derived/{destination_name}",
                "sha256": sha256(destination),
                "bytes": destination.stat().st_size,
                "rows": row_count,
                "columns": columns,
                "supports": supports,
                "paper_source": source_relative,
                "source_sha256": sorted(source_hashes),
            }
        )

    missing_hashes = sorted(referenced_source_hashes - evidence_hashes)
    if missing_hashes:
        raise ValueError(f"derived data reference unindexed evidence hashes: {missing_hashes}")

    manifest = {
        "schema": "eahr-public-paper-results-v1",
        "title": PAPER_TITLE,
        "contents": "Processed source data for the reported figures, table, and result claims.",
        "derived_files": sorted(derived, key=lambda item: item["path"]),
        "source_evidence": {
            "path": "source-evidence.json",
            "sha256": sha256(evidence_path),
            "files": len(evidence),
        },
    }
    write_json(output / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "paper-results",
    )
    parser.add_argument(
        "--verify-sources",
        action="store_true",
        help="re-hash every frozen source file before building (requires the canonical archive)",
    )
    args = parser.parse_args()
    build_package(
        paper_root=args.paper_root,
        artifact_root=args.artifact_root,
        output=args.output,
        verify_sources=args.verify_sources,
    )
    print(f"built public paper-results package: {args.output.resolve()}")


if __name__ == "__main__":
    main()
