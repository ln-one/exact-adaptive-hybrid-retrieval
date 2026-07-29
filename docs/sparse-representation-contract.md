# Canonical Sparse representation contract v1

The Sparse channel has two distinct artifacts. They must not be conflated.

## 1. Lexical reference ranking

The external, publication-facing lexical baseline is a Lucene BM25 index built
by Pyserini/Anserini from the canonical `id` / `text` corpus. Its analyzer,
BM25 parameters, Lucene/Pyserini version, Java runtime and complete index
checksum are recorded in the index manifest. This is the reference definition
for a conventional lexical ranking and is evaluated with the official qrels.

The Pyserini project documents custom JSONL collection indexing and an
embeddable Lucene BM25 indexer. This benchmark pins Pyserini in its own Python
environment because its current Transformers dependency is intentionally kept
separate from the frozen Dense runtime. It uses the host's recorded JDK 21
installation. The index manifest records both environments rather than
silently swapping tokenizer or Java stacks.

## 2. Stratumind sparse channel input

Stratumind/Qdrant consumes a non-negative sparse representation. It is an
execution input, not a claim that Qdrant's physical posting layout equals the
Lucene baseline. Its manifest must record:

- frozen tokenizer/encoder identifier and all parameters;
- vocabulary or model revision and checksum;
- document and query source checksums;
- vector dtype, non-negativity check and dimension/term-id domain;
- artifact byte size and shard checksums.

The benchmark reports both layers. A Dense/Sparse ED-WRRF run uses the frozen
Stratumind channel representations; the Lucene BM25 run remains an independent
mature lexical reference. A representation cannot be called canonical merely
because its output is accepted by an index.

## 3. Gates

- No arbitrary text re-chunking or corpus sampling.
- No on-the-fly vocabulary fitting during evaluation.
- Every generated vector must be finite; Sparse weights must be non-negative.
- Index build and query runtime are separate measurements.
- A missing or mismatched Sparse manifest invalidates a manuscript result.
