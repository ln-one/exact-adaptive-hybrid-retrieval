#!/bin/zsh
# Start only after the Dense queue writes its successful completion marker.
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

[[ -f "$artifact_root/logs/remaining-dense.success" ]] || {
  print -u2 "Dense queue did not complete successfully; refusing Sparse generation"
  exit 1
}

java_home=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home
export JAVA_HOME="$java_home"
export PATH=/opt/homebrew/opt/openjdk@21/bin:$PATH

for dataset in trec-covid touche-2020 quora msmarco-passage-trec-dl-2019; do
  .venv-sparse/bin/python scripts/build_bm25_impacts.py --artifact-root "$artifact_root" --dataset "$dataset" --row-batch-size 10000 --shard-rows 100000
  .venv-sparse/bin/python scripts/verify_bm25_impacts.py --artifact-root "$artifact_root" --dataset "$dataset"
done

.venv-sparse/bin/python scripts/build_bm25_impacts.py --artifact-root "$artifact_root" --dataset msmarco-passage-trec-dl-2020 --reuse-documents-from msmarco-passage-trec-dl-2019 --row-batch-size 10000 --shard-rows 100000
.venv-sparse/bin/python scripts/verify_bm25_impacts.py --artifact-root "$artifact_root" --dataset msmarco-passage-trec-dl-2020

for dataset in trec-covid touche-2020 quora msmarco-passage-trec-dl-2019; do
  .venv-sparse/bin/python scripts/build_lucene_bm25.py --artifact-root "$artifact_root" --dataset "$dataset" --threads 8 --row-batch-size 10000
  .venv-sparse/bin/python scripts/verify_lucene_bm25.py --artifact-root "$artifact_root" --dataset "$dataset"
done

touch "$artifact_root/logs/remaining-sparse.success"
