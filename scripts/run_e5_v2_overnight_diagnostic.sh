#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 ARTIFACT_ROOT STRATUMIND_REPO" >&2
  exit 2
fi

bench_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
artifact_root="$(cd "$1" && pwd)"
system_repo="$(cd "$2" && pwd)"
system_binary="${system_repo}/target/release/qdrant"
build_manifest="${artifact_root}/manifests/build/qdrant-70f4943d-canonical-bench.json"
hardware_manifest="${artifact_root}/manifests/hardware/apple-m4-pro-24gb-v1.json"
system_artifact="sha256:28c352630bb6bad140a51fabc56e358da1c2e992b983152067843ba2823fe980"
run_root="${artifact_root}/runs/e5-v2-overnight-full-coverage-20260803"

run_dataset() {
  local dataset="$1"
  local collection="$2"
  local shard_size="$3"
  local dataset_root="${run_root}/${dataset}"
  local campaign="${dataset_root}/campaign.json"
  local shard_dir="${dataset_root}/shards"
  local failed_dir="${dataset_root}/failed"
  local aggregate="${dataset_root}/aggregate.json"

  mkdir -p "${dataset_root}" "${shard_dir}" "${failed_dir}"
  if [[ ! -f "${campaign}" ]]; then
    "${bench_repo}/.venv/bin/python" "${bench_repo}/scripts/run_canonical.py" \
      e5-v2-plan \
      --artifact-root "${artifact_root}" \
      --dataset "${dataset}" \
      --collection "${collection}" \
      --output "${campaign}" \
      --bench-repo "${bench_repo}" \
      --system-repo "${system_repo}" \
      --system-artifact "${system_artifact}" \
      --system-binary "${system_binary}" \
      --system-build-manifest "${build_manifest}" \
      --hardware-manifest "${hardware_manifest}" \
      --rounds 1 \
      --shard-size "${shard_size}" \
      --warmups 4 \
      --repetitions 2 \
      --request-timeout-seconds 1200 \
      --startup-timeout-seconds 1800 \
      --shard-wall-timeout-seconds 21600 \
      --allow-dirty
  fi

  "${bench_repo}/.venv/bin/python" "${bench_repo}/scripts/run_e5_v2_campaign.py" \
    --artifact-root "${artifact_root}" \
    --campaign-manifest "${campaign}" \
    --output-dir "${shard_dir}" \
    --failed-dir "${failed_dir}" \
    --bench-repo "${bench_repo}" \
    --system-repo "${system_repo}" \
    --system-binary "${system_binary}" \
    --system-build-manifest "${build_manifest}" \
    --hardware-manifest "${hardware_manifest}" \
    --allow-dirty

  if [[ ! -f "${aggregate}" ]]; then
    "${bench_repo}/.venv/bin/python" "${bench_repo}/scripts/aggregate_e5_v2.py" \
      --campaign-manifest "${campaign}" \
      --output "${aggregate}" \
      --failed-dir "${failed_dir}" \
      --allow-dirty \
      "${shard_dir}"/*.jsonl
  fi
}

# The scale dataset runs first because it dominates wall-clock time.
run_dataset \
  "msmarco-passage-trec-dl-2020" \
  "ed-wrrf-msmarco-passage-trec-dl-2019-canonical-v2" \
  9
run_dataset "trec-covid" "ed-wrrf-trec-covid-canonical-v2" 50
run_dataset "scifact" "ed-wrrf-scifact-canonical-v2" 300
run_dataset "nfcorpus" "ed-wrrf-nfcorpus-canonical-v2" 323
