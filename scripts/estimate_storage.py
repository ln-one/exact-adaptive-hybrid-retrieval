#!/usr/bin/env python3
"""Estimate canonical vector footprint before any expensive encoding."""

from __future__ import annotations

import argparse
import shutil
import tomllib
from pathlib import Path


GIB = 1024**3


def gib(value: int) -> float:
    return value / GIB


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets.toml",
    )
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    with args.config.open("rb") as handle:
        config = tomllib.load(handle)
    root = args.root or Path(config["storage"]["default_root"])

    regular = [
        spec
        for spec in config["datasets"].values()
        if spec.get("tier") in {"core", "secondary", "large-archive"}
    ]
    regular_documents = sum(spec["documents"] for spec in regular)
    msmarco_documents = config["datasets"]["msmarco_passage_trec_dl"]["documents"]

    dense_regular = regular_documents * 384 * 4
    dense_msmarco = msmarco_documents * 384 * 4
    e5_robustness_documents = (
        config["datasets"]["scifact"]["documents"]
        + config["datasets"]["trec_covid"]["documents"]
        + config["datasets"]["cqadupstack"]["documents"]
    )
    dense_e5 = e5_robustness_documents * 768 * 4
    archive_bytes = sum(spec.get("archive_bytes", 0) for spec in regular)
    free = shutil.disk_usage(root.parent).free

    print(f"BEIR-family documents: {regular_documents:,}")
    print(f"MS MARCO documents:     {msmarco_documents:,}")
    print(f"Known BEIR archives:    {gib(archive_bytes):8.2f} GiB")
    print(f"BGE f32 BEIR vectors:   {gib(dense_regular):8.2f} GiB")
    print(f"BGE f32 MS MARCO:       {gib(dense_msmarco):8.2f} GiB")
    print(f"E5 f32 robustness:      {gib(dense_e5):8.2f} GiB")
    print(f"Current free space:     {gib(free):8.2f} GiB")
    print()
    print("Planning envelope (includes raw text, sparse impacts, indexes, results, temp):")
    print("  expected canonical working set: 100-160 GiB")
    print(f"  enforced hard budget:           {config['storage']['hard_budget_gib']} GiB")
    print(
        f"  minimum free space afterwards:  "
        f"{config['storage']['minimum_free_after_gib']} GiB"
    )


if __name__ == "__main__":
    main()

