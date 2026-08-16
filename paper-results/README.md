# EAHR paper result data

This directory contains the compact processed source data behind the figures,
table, and numerical result claims in **Exact Adaptive Hybrid Retrieval Without
Fixed Top-L Cutoffs**.

## Contents

- `derived/`: ten CSV files used for paper figures, tables, or stated result
  checks;
- `manifest.json`: file checksums, byte and row counts, columns, and
  manuscript-role mappings;
- `source-evidence.json`: sanitized relative paths and SHA-256 digests for the
  frozen upstream aggregates, logs, and run files;
- `DATA_DICTIONARY.md`: file-level and column-level interpretation.
- `PROVENANCE_NOTES.md`: audit notes for non-obvious execution-provenance
  fields that must not be silently normalized away.
- `LICENSE.md`: CC BY 4.0 terms for author-generated processed data; third-party
  inputs are excluded.

Run `uv run python scripts/verify_public_paper_results.py` from the repository
root to detect changed files, missing files, unindexed source hashes, malformed
CSVs, or leaked absolute paths.

## Evidence boundary

The checked-in CSVs are processed source data, not replacements for the
third-party benchmark corpora. BEIR collections, MS MARCO, TREC-COVID source
releases, model caches, vector indexes, and raw execution logs are not
redistributed here. They are excluded because of their size and upstream data
terms. `source-evidence.json` retains content hashes and logical paths so a
maintainer with the frozen canonical archive can audit every derived row.

The experiment harness is in this repository. The corresponding retrieval
implementation is maintained in
[StratuMind](https://github.com/ln-one/StratuMind). The paper is identified by
[DOI: 10.48550/arXiv.2608.07152](https://doi.org/10.48550/arXiv.2608.07152).
The software and processed-data package is preserved under the stable Zenodo
concept DOI
[`10.5281/zenodo.21968866`](https://doi.org/10.5281/zenodo.21968866); release
`v0.1.1` is archived as
[`10.5281/zenodo.21968867`](https://doi.org/10.5281/zenodo.21968867).

## Data Availability

The processed source data underlying the reported figures, table, and
numerical result claims are included in the `paper-results/derived` directory,
together with file-level checksums, column metadata, and a content-addressed
provenance index. The study reuses publicly available benchmark collections,
including BEIR, MS MARCO passage ranking, and TREC-COVID resources, under their
respective upstream access and use terms; these third-party corpora are not
redistributed. Model artifacts, vector indexes, and raw execution logs are also
excluded from this compact package because of their size. Their frozen source
files are identified by relative logical path and SHA-256 digest in
`paper-results/source-evidence.json`. Code for data preparation, validation,
and experiment execution is provided in this repository, and the corresponding
retrieval implementation is available from the StratuMind repository linked
above. The frozen public package is available from the Zenodo record linked
above.
