# Canonical benchmark data-use register

This register is a reproducibility and scope record, not legal advice. A
dataset is eligible for an experiment only when its row below is marked
`accepted-for-research` and the experiment follows its stated restriction.
Generated vectors and indexes inherit the source dataset's restriction.

| Source group | Evidence | Status | Restriction carried into artifacts |
|---|---|---|---|
| MS MARCO Passage / TREC-DL | [Official MS MARCO terms](https://microsoft.github.io/msmarco/) | accepted-for-research | Non-commercial research only; no redistribution of source text or derived artifacts without checking the terms. Project operator explicitly accepted the official terms before acquisition. |
| TREC-COVID / CORD-19 | [NIST TREC-COVID Complete](https://ir.nist.gov/covidSubmit/data.html), [task guidance](https://ir.nist.gov/trec-covid/round2.html) | accepted-for-research-with-source-rights | Use only as the frozen benchmark collection; retain its CORD-19 source-rights notice and do not claim that NIST grants rights to every underlying article. |
| BEIR-hosted corpora | [BEIR's license disclaimer](https://github.com/beir-cellar/beir) | source-specific-review-required | BEIR explicitly says that use permission remains the user's responsibility. A corpus is not licensed merely by its BEIR archive. |

## Operational rules

- All public result tables identify the exact dataset and official split.
- Canonical artifacts are local research materials, not source-text releases.
- Any dataset still marked `pending-source-audit` in `datasets.toml` is excluded
  from a public release or a claimed canonical evaluation suite until its
  original-source terms have been recorded here.
- MS MARCO and TREC-COVID remain research-only. They are never used to justify
  a commercial deployment claim.
