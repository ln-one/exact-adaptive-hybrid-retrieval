# Canonical benchmark preparation status

This file records only artifacts that have completed checksum-backed export or
embedding verification. Active transfers are intentionally not presented as
complete.

## Complete source exports

| Dataset | Documents | Queries | Qrels | Status |
|---|---:|---:|---:|---|
| NFCorpus | 3,633 | 323 | 12,334 | manifest checksum verified |
| SciFact | 5,183 | 300 | 339 | manifest checksum verified |
| ArguAna | 8,674 | 1,406 | 1,406 | manifest checksum verified |
| SciDocs | 25,657 | 1,000 | 29,928 | manifest checksum verified |
| TREC-COVID | 171,332 | 50 | 66,336 | manifest checksum verified |
| Touche-2020 | 382,545 | 49 | 2,962 | manifest checksum verified |
| Quora | 522,931 | 10,000 | 15,675 | manifest checksum verified |
| MS MARCO Passage / TREC-DL 2019 | 8,841,823 | 43 | 9,260 | manifest checksum verified |
| MS MARCO Passage / TREC-DL 2020 | shared with 2019 | 54 | 11,386 | manifest checksum verified |
| CQADupStack / Android | 22,998 | 699 | 1,696 | manifest checksum verified |

All exports preserve official document/query identities and qrels. Documents
with an official empty retrieval body are retained with empty text rather than
removed or filled using metadata.

## Complete canonical Dense artifacts

| Dataset | Documents | Queries | Encoder | Verification |
|---|---:|---:|---|---|
| NFCorpus | 3,633 | 323 | BAAI/bge-small-en-v1.5 f32 | IDs/checksums/vectors/norms verified |
| Quora | 522,931 | 10,000 | BAAI/bge-small-en-v1.5 f32 | IDs/checksums/vectors/norms verified |
| CQADupStack / Android | 22,998 | 699 | BAAI/bge-small-en-v1.5 f32 | IDs/checksums/vectors/norms verified |
| TREC-COVID | 171,332 | 50 | BAAI/bge-small-en-v1.5 f32 | IDs/checksums/vectors/norms verified |
| MS MARCO Passage / TREC-DL 2019 | pending documents | 43 | BAAI/bge-small-en-v1.5 f32 | query artifact verified; MPS document run active |
| MS MARCO Passage / TREC-DL 2020 | shared documents | 54 | BAAI/bge-small-en-v1.5 f32 | query artifact verified |

The frozen model revision is
`01d3c3cd65ac9dc6bd0d702ed913366e7931097b`; its local immutable snapshot and
file checksums live under `models/bge-small-en-v1.5/`.

## Complete Sparse artifacts

| Dataset | Artifact | Verification |
|---|---|---|
| NFCorpus | Pyserini/Anserini Lucene BM25 reference index | index and source checksums verified |
| NFCorpus | `bm25-impact-v1` portable non-negative vectors | source, shard, vocabulary and vector checks verified |

The isolated JDK 21 / Pyserini runtime is pinned in `sparse-requirements.lock`.

## Active work

- MS MARCO Passage official corpus + TREC-DL 2019 judged qrels;
- Dense/Sparse representation generation for remaining canonical corpora.
