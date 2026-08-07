# Data dictionary

All ratios are unitless unless noted. Latencies are milliseconds. A field named
`source_sha256` (or ending in `_sha256`) identifies the frozen upstream file
from which the row was derived; every such digest must occur in
`source-evidence.json`.

Boolean values are serialized as `True` or `False`. Empty cells mean the field
is not applicable to that row, not zero. Confidence-interval fields ending in
`ci95_low` and `ci95_high` are the lower and upper endpoints of a 95% interval.

## Files

### `fixed-depth-effectiveness.csv`

Static effectiveness and ranking-agreement results for dense-only,
sparse-only, fixed Top-L, and complete-list weighted RRF. `depth` is the
per-channel fixed cutoff. `mean` is the collection mean for `metric`.
`paired_difference_from_full` is method minus complete-list weighted RRF.
Agreement fields compare the returned ranking with that complete-list target.

### `temporal-transfer-summary.csv`

One row per TREC-COVID snapshot. It records full-WRRF and cross-fitted fixed-L
nDCG@10, the paired difference, the hindsight-only diagnostic best depth, EAHR
median channel depths and read ratios, exhaustion counts, and ordered
mismatches. `diagnostic_best_l` is descriptive and is not a deployment rule.

### `query-snapshot-access.csv`

One row per query and corpus snapshot for Figure 3. `denseDepth` and
`sparseDepth` are ranks exposed to fusion; `denseFullDepth` and
`sparseFullDepth` are complete support lengths. Read ratios are exposed depth
divided by complete length. `orderedMismatch` indicates disagreement with the
complete-list ordered Top-20.

### `query-snapshot-variance.csv`

Descriptive two-factor sum-of-squares decomposition of log10 read ratios.
`query_share`, `snapshot_share`, and `query_snapshot_interaction_share` sum to
one up to floating-point rounding; they are descriptive variance shares, not
causal effects.

### `ordered-correctness.csv`

Counts of queries, measured observations, ordered-result mismatches, timeouts,
and errors for the exactness checks. A zero count means none were observed in
the frozen campaign; it is not a universal guarantee for all deployments.

### `aggregate-efficiency.csv`

Collection-level latency and access summaries. `dynamic_*` fields describe
EAHR; `baseline_*` fields describe the named exhaustive baseline.
`paired_ratio` is EAHR divided by baseline, so values below one favor EAHR.
`paired_win_rate` is the fraction of paired query comparisons won by EAHR.

### `per-query-latency.csv`

Per-query medians for the large TREC-DL collections. `paired_ratio` is EAHR
divided by native exhaustive batch execution. `dynamic_wins` is one when that
ratio is below one. Values in this file support the paper's query-tail caveats.

### `fixed-prefix-agreement.csv`

Agreement between fixed-depth fusion and complete-list weighted RRF. The file
separates candidate omission (`candidate_missing`), a present candidate with a
wrong result (`candidate_present_but_wrong`), and order-only disagreement.
Rates are query fractions; depth and candidate counts are ranks/items.

### `controlled-ranking-regimes.csv`

Synthetic rank-stream results by collection `size` and cross-channel `regime`.
Certificate depth is the first checked depth at which the ordered Top-20 is
fixed. `work_ratio_*` is certificate depth divided by list size.
`exhaustive_cases` counts seeds that reached list exhaustion.

### `rank-generator-ablation.csv`

PVS/PBM rank-generator comparisons. `ratio_vs_reference` is the alternative
plan divided by the PVS+PBM reference, so values above one favor the reference.
Counter fields are medians per query. `measured_observations` equals queries ×
rounds × repetitions per round for complete campaigns.
