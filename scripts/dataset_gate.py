"""Dataset eligibility gates shared by canonical artifact builders."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path


ELIGIBLE_LICENSE_STATUSES = {
    "research-audited",
    "research-only",
    "research-only-mixed-cord19-rights",
    "research-only-source-terms-required",
}


def dataset_key(dataset: str) -> str:
    if dataset.startswith("msmarco-passage-trec-dl-"):
        return "msmarco_passage_trec_dl"
    if dataset.startswith("cqadupstack-"):
        return "cqadupstack"
    return dataset.replace("-", "_")


def assert_dataset_eligible(dataset: str, config: Path | None = None) -> str:
    config = config or Path(__file__).resolve().parents[1] / "datasets.toml"
    with config.open("rb") as handle:
        datasets = tomllib.load(handle)["datasets"]
    key = dataset_key(dataset)
    if key not in datasets:
        raise RuntimeError(f"dataset is absent from canonical registry: {dataset}")
    status = datasets[key]["license_status"]
    if status not in ELIGIBLE_LICENSE_STATUSES:
        raise RuntimeError(
            f"dataset {dataset!r} has license status {status!r}; "
            "canonical artifact generation is blocked pending original-source review"
        )
    return status


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail closed unless a dataset is eligible for canonical artifacts."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    status = assert_dataset_eligible(args.dataset, args.config)
    print(f"{args.dataset}: {status}")


if __name__ == "__main__":
    main()
