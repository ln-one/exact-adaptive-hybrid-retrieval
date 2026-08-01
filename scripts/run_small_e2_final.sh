#!/bin/zsh
# Run clean causal and engineering E2 pairs on the three small/medium datasets.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  print -u2 "usage: $0 <artifact-root> <system-repo>"
  exit 2
fi

artifact_root=${1:A}
system_repo=${2:A}
script_dir=${0:A:h}
bench_repo=${script_dir:h}
system_commit=70f4943d9604cd2b5fe2df60e93521015d87fa74
binary_sha=28c352630bb6bad140a51fabc56e358da1c2e992b983152067843ba2823fe980
binary="$system_repo/target/release/qdrant"
build_manifest="$artifact_root/manifests/build/qdrant-70f4943d-canonical-bench.json"
hardware_manifest="$artifact_root/manifests/hardware/apple-m4-pro-24gb-v1.json"
output_dir="$artifact_root/logs/e2/final-small"

cd "$bench_repo"
[[ -z $(git status --porcelain) ]] || { print -u2 "bench repository is dirty"; exit 1; }
[[ -z $(git -C "$system_repo" status --porcelain) ]] || {
  print -u2 "system repository is dirty"
  exit 1
}
[[ $(git -C "$system_repo" rev-parse HEAD) == $system_commit ]] || {
  print -u2 "unexpected system commit"
  exit 1
}
[[ -f $binary && -f $build_manifest && -f $hardware_manifest ]] || {
  print -u2 "binary or provenance manifest is missing"
  exit 1
}
[[ $(shasum -a 256 "$binary" | awk '{print $1}') == $binary_sha ]] || {
  print -u2 "binary SHA-256 mismatch"
  exit 1
}
if pgrep -x qdrant >/dev/null; then
  print -u2 "another qdrant process is running"
  exit 1
fi

run_pair() {
  local dataset=$1
  local baseline=$2
  local collection="ed-wrrf-${dataset}-canonical-v2"
  local output="$output_dir/${dataset}-${baseline}-final-v1.jsonl"
  if [[ -f $output ]]; then
    .venv/bin/python scripts/validate_canonical_log.py "$output"
    print "reuse validated output: $output"
    return
  fi
  [[ ! -e ${output}.sha256 ]] || {
    print -u2 "orphan checksum exists: ${output}.sha256"
    exit 1
  }
  .venv/bin/python scripts/run_canonical.py e2 \
    --artifact-root "$artifact_root" \
    --dataset "$dataset" \
    --collection "$collection" \
    --bench-repo "$bench_repo" \
    --system-repo "$system_repo" \
    --system-commit "$system_commit" \
    --system-binary "$binary" \
    --system-build-manifest "$build_manifest" \
    --hardware-manifest "$hardware_manifest" \
    --system-artifact "sha256:$binary_sha" \
    --baseline "$baseline" \
    --output "$output" \
    --warmups 2 \
    --repetitions 5 \
    --request-timeout-seconds 900
  .venv/bin/python scripts/validate_canonical_log.py "$output"
}

mkdir -p "$output_dir"
for dataset in nfcorpus scifact trec-covid; do
  run_pair "$dataset" same-producer-exhaustive
  run_pair "$dataset" native-bulk-exhaustive
done
print "small/medium E2 matrix complete"
