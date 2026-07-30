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

## Canonical experiment runner

`run_canonical.py` is the thin publication harness. Its first executable family
is E1 ordered parity: Qdrant exact channel queries produce the exhaustive Dense
and Sparse orders, the runner applies Stratumind's frozen WRRF formula and
identity tie rule, and the Production `exact-rrf` response must match.

Create the Collection with the explicit Production profile:

```bash
.venv/bin/python scripts/create_qdrant_collection.py \
  --url http://127.0.0.1:6333 \
  --collection ed-wrrf-nfcorpus \
  --exact-rank-profile dense_sparse_v1
```

After loading both representations, run E1:

```bash
.venv/bin/python scripts/run_canonical.py e1 \
  --artifact-root /Users/ln1/Projects/stratumind-artifacts/canonical-v1 \
  --dataset nfcorpus \
  --collection ed-wrrf-nfcorpus \
  --system-repo /path/to/frozen/Stratumind \
  --system-artifact sha256:<container-or-binary-digest> \
  --output /path/to/results/e1-nfcorpus.jsonl
```

Canonical execution refuses dirty source repositories. `--allow-dirty` exists
only for development dry runs; such logs carry `"dirty": true` and fail the
publication validator by default.

E1 latency is correctness-instrumentation latency: the exhaustive oracle is
queried before the method and the record is marked `correctness-validation`.
It must not be copied into the E2 performance table.

```bash
.venv/bin/python scripts/validate_canonical_log.py \
  /path/to/results/e1-nfcorpus.jsonl
```

The live exhaustive HTTP oracle is intentionally bounded to small corpora.
Larger datasets require a separately frozen exhaustive-oracle artifact rather
than returning millions of identities through one HTTP response.
