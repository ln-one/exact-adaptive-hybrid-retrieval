# Canonical data protocol v1

This repository prepares publication-eligible inputs for ED-WRRF. It is
independent from the Qdrant fork and never writes generated data into a source
worktree.

## Evidence layers

1. Standard IR datasets preserve their official retrieval objects, queries,
   splits, and qrels.
2. Frozen Dense and Sparse representations define the two exact channel
   rankings.
3. Exhaustive execution produces the ordered WRRF oracle and per-query
   execution traces.
4. Controlled synthetic workloads cover correlation, anti-correlation, ties,
   and worst-case full expansion.

## Non-negotiable rules

- No re-chunking of a dataset that has document-level qrels.
- No custom corpus sampling in the main benchmark.
- CQADupStack subforums remain independent datasets.
- TREC-COVID uses the complete corpus in the main benchmark.
- Model repository revisions and every consumed model file are checksummed.
- Canonical Dense vectors use original float32 model weights. Quantized ONNX
  embeddings are separate engineering ablations.
- Raw archives, extracted corpora, vectors, indexes, and results are stored
  outside Git under the configured artifact root.
- A result is manuscript-eligible only when its dataset, representation,
  executable, parameters, hardware, and output checksums are recorded.

## Capacity gates

The artifact root has a 180 GiB working-set budget and must leave at least
120 GiB free. CQADupStack and MS MARCO are separate gates. Full per-query
rankings are not archived by default; the pipeline stores ordered Top-K
oracles, trace summaries, and reproducible inputs instead.

