#!/bin/zsh
# Final deterministic gate for the research-eligible canonical v1 suite.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  print -u2 "usage: $0 <artifact-root>"
  exit 2
fi

artifact_root=$1
script_dir=${0:A:h}
repo_root=${script_dir:h}
cd "$repo_root"

[[ -f "$artifact_root/models/bge-small-en-v1.5/01d3c3cd65ac9dc6bd0d702ed913366e7931097b/model-manifest.json" ]] || {
  print -u2 "frozen BGE model manifest is missing"
  exit 1
}

for dataset in nfcorpus scifact trec-covid msmarco-passage-trec-dl-2019 msmarco-passage-trec-dl-2020; do
  .venv/bin/python scripts/dataset_gate.py --dataset "$dataset"
  .venv/bin/python scripts/verify_source.py --artifact-root "$artifact_root" --dataset "$dataset"
  for kind in documents queries; do
    .venv/bin/python scripts/verify_dense.py --artifact-root "$artifact_root" --dataset "$dataset" --kind "$kind"
  done
  .venv-sparse/bin/python scripts/verify_bm25_impacts.py --artifact-root "$artifact_root" --dataset "$dataset"
done

for dataset in nfcorpus scifact trec-covid msmarco-passage-trec-dl-2019; do
  .venv-sparse/bin/python scripts/verify_lucene_bm25.py --artifact-root "$artifact_root" --dataset "$dataset"
done

free_gib=$(df -Pk "$artifact_root" | awk 'NR == 2 { printf "%.1f", $4 / 1024 / 1024 }')
awk -v free="$free_gib" 'BEGIN { if (free < 120) exit 1 }' || {
  print -u2 "capacity audit failed: only ${free_gib} GiB free; require >=120 GiB"
  exit 1
}
print "canonical-v1 audit passed; free_gib=${free_gib}"
