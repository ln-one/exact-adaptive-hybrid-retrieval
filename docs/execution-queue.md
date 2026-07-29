# Canonical v1 execution queue

This queue prevents the benchmark preparation jobs from competing for the same
MPS device, memory bandwidth, or artifact disk.

## Current critical path

1. Finish and verify `msmarco-passage-trec-dl-2019` document Dense vectors.
   The 2020 task reuses those document vectors and needs only its own queries.
2. Encode remaining missing canonical Dense artifacts, one MPS job at a time:
   SciFact. Touche-2020, SciDocs and ArguAna remain outside this queue until
   their original-source license rows are audited.
3. Materialize `bm25-impact-v1` for remaining eligible corpora. Run these
   CPU/Java jobs only after the MPS job is idle; process MS MARCO last. The TREC-DL 2020
   task hard-links the verified 2019 document impacts and materializes only its
   distinct query vectors.
4. Build Lucene BM25 reference indexes, also CPU/Java only; small datasets
   first, then TREC-COVID, Quora and MS MARCO.
5. Run the all-artifact checksum, source identity, vector and capacity gates
   before producing any rank oracle or performance result.

## Resource policy

- Never run a large MPS Dense encoder and a large Java Sparse build together.
- Dense encoding writes atomic 25,000-row shards, so interruption is safe and
  completed shards are never recomputed.
- Every Sparse builder writes a separate atomic directory. A stale `.building`
  directory is an error requiring inspection, never an invitation to overwrite.
- Do not start a job that could violate the 120 GiB free-space floor.
