# Reproduce EAHR

Run commands from the repository root after installing the locked environment with uv sync --frozen.

## Prepare an artifact root

Full reproduction additionally requires JDK 21, the separately locked
Pyserini environment, a clean checkout of the recorded StratuMind revision,
and sufficient local storage.

```bash
export EAHR_ARTIFACT_ROOT=/path/to/eahr-artifacts/canonical-v1
mkdir -p "$EAHR_ARTIFACT_ROOT"

uv run python scripts/estimate_storage.py --root "$EAHR_ARTIFACT_ROOT"
uv run python scripts/fetch_beir.py \
  --root "$EAHR_ARTIFACT_ROOT" \
  --dataset nfcorpus
```

Large archives are never selected implicitly. BEIR availability is not treated
as a license grant.

The portable non-negative sparse representation is built in the isolated
sparse environment:

```bash
JAVA_HOME=/path/to/jdk-21 \
PATH=/path/to/jdk-21/bin:$PATH \
.venv-sparse/bin/python scripts/build_lucene_bm25.py \
  --artifact-root "$EAHR_ARTIFACT_ROOT" \
  --dataset nfcorpus
```

## Run a canonical check

After preparing and loading a collection, E1 obtains exact dense and sparse
rankings, applies the frozen weighted-RRF and tie-order contract, and compares
that oracle with the StratuMind response:

```bash
uv run python scripts/run_canonical.py e1 \
  --artifact-root "$EAHR_ARTIFACT_ROOT" \
  --dataset nfcorpus \
  --collection ed-wrrf-nfcorpus \
  --system-repo /path/to/frozen/StratuMind \
  --system-artifact sha256:<binary-or-container-digest> \
  --output /path/to/results/e1-nfcorpus.jsonl

uv run python scripts/validate_canonical_log.py \
  /path/to/results/e1-nfcorpus.jsonl
```

Canonical execution rejects dirty source repositories by default.
`--allow-dirty` is a development escape hatch; it marks records dirty and makes
them ineligible for publication validation. E1 includes correctness-oracle
work and must not be reported as E2 performance latency.

The clean E5-v2 counterbalanced campaign also requires a unique run label:

```bash
bash scripts/run_e5_v2_counterbalanced_overnight.sh \
  "$EAHR_ARTIFACT_ROOT" \
  /path/to/frozen/StratuMind \
  e5-v2-clean-YYYYMMDD
```

## Rebuild the compact result package

Readers normally use the checked-in package. Maintainers with the private paper
tree and canonical evidence archive can reproduce it deterministically:

```bash
uv run python scripts/build_public_paper_results.py \
  --paper-root /path/to/stratumind-paper \
  --artifact-root /path/to/eahr-artifacts/canonical-v1 \
  --verify-sources
```

`--verify-sources` re-hashes every indexed frozen input. It is separate from
reader-side verification because the large inputs are not redistributed.

## Provenance status

The checked-in E5-v2 measurements retain their original dirty-source flag;
they are not silently relabeled. Their recorded runner hashes match the later
committed harness, while a new clean-checkout rerun is maintained as a separate
campaign. The exact scope and interpretation are documented in
[`paper-results/PROVENANCE_NOTES.md`](../paper-results/PROVENANCE_NOTES.md).
