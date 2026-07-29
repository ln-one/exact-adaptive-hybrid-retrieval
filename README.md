# Stratumind Bench

Reproducible data and experiment pipeline for ED-WRRF.

The source repository contains only configuration, manifests, and scripts.
Generated artifacts live under:

```text
/Users/ln1/Projects/stratumind-artifacts/canonical-v1
```

Estimate the working set:

```bash
python3 scripts/estimate_storage.py
```

Fetch the core BEIR archives with archive and capacity checks:

```bash
python3 scripts/fetch_beir.py --tier core
```

Large archives are always explicit:

```bash
python3 scripts/fetch_beir.py --dataset cqadupstack
```

MS MARCO is prepared through its fixed `ir_datasets` identifiers after the
core archive and representation gates pass.

The lexical reference and portable Sparse impacts use an isolated Pyserini
runtime so its Transformer dependency cannot alter the frozen Dense encoder:

```bash
JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home \
PATH=/opt/homebrew/opt/openjdk@21/bin:$PATH \
.venv-sparse/bin/python scripts/build_lucene_bm25.py \
  --artifact-root /Users/ln1/Projects/stratumind-artifacts/canonical-v1 \
  --dataset nfcorpus
```

`scripts/build_bm25_impacts.py` materializes the corresponding portable,
non-negative Sparse vectors. Both builders write atomic, checksum-backed
artifacts outside this Git repository.

To load a verified Dense/Sparse pair later, create a matching named-vector
collection, then run `load_qdrant_dense.py` followed by `load_qdrant_sparse.py`.
The sparse loader uses Qdrant's vector-update endpoint so it does not overwrite
the Dense vector already stored for the same deterministic point identity.
