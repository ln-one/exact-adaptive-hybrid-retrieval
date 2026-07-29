#!/bin/zsh
# Run only after a separately-started MS MARCO encoder succeeds. Designed for
# overnight use: a failed predecessor stops the queue at verification time.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  print -u2 "usage: $0 <artifact-root> <wait-for-pid>"
  exit 2
fi

artifact_root=$1
wait_pid=$2
script_dir=${0:A:h}
repo_root=${script_dir:h}
cd "$repo_root"

while kill -0 "$wait_pid" 2>/dev/null; do
  sleep 60
done

uv run python scripts/verify_dense.py --artifact-root "$artifact_root" --dataset msmarco-passage-trec-dl-2019 --kind documents
uv run python scripts/reuse_dense_documents.py --artifact-root "$artifact_root" --from-dataset msmarco-passage-trec-dl-2019 --to-dataset msmarco-passage-trec-dl-2020
uv run python scripts/verify_dense.py --artifact-root "$artifact_root" --dataset msmarco-passage-trec-dl-2020 --kind documents

for dataset in touche-2020 scidocs arguana scifact; do
  uv run python scripts/embed_dense.py --artifact-root "$artifact_root" --dataset "$dataset" --kind documents --batch-size 128 --shard-rows 25000 --device mps
  uv run python scripts/verify_dense.py --artifact-root "$artifact_root" --dataset "$dataset" --kind documents
  uv run python scripts/embed_dense.py --artifact-root "$artifact_root" --dataset "$dataset" --kind queries --batch-size 128 --shard-rows 25000 --device mps
  uv run python scripts/verify_dense.py --artifact-root "$artifact_root" --dataset "$dataset" --kind queries
done
