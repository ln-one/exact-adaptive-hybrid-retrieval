# Provenance notes

## Common system implementation

All 27 extracted execution-provenance records identify the same system commit
and binary artifact:

- StratuMind commit: `70f4943d9604cd2b5fe2df60e93521015d87fa74`;
- binary SHA-256: `28c352630bb6bad140a51fabc56e358da1c2e992b983152067843ba2823fe980`.

The source records themselves remain authoritative. These values are written
here only to make a cross-campaign consistency check visible to readers.

## E5-v2 dirty-source flag

The four counterbalanced E5-v2 campaign manifests used for the rank-generator
table record `dirty: true` and `publicationEligible: false`. This flag must be
disclosed rather than edited after the fact.

The cause is recoverable: the campaigns were launched while the new E5-v2
bench harness was not yet committed. The recorded campaign-driver and
aggregation-script hashes exactly match commit
`6e355554edfc5237b6cd379bc7a2209200364771` ("Add counterbalanced E5-v2
experiment workflow"), committed shortly after the runs. The recorded combined
runner-source hash also matches that commit's runner source tree. The system
commit and binary digest are fixed and identical to the other campaigns.

This evidence supports auditability, but it does not turn the original campaign
flag into `publicationEligible: true`. For the strongest final archival record,
rerun E5-v2 from a clean tagged bench revision and compare the resulting table
with `derived/rank-generator-ablation.csv`. Until then, describe these results
as checksum-backed frozen measurements, not as a clean-checkout reproduction.

## Integrity rule

Do not rewrite old manifests, change `dirty`, or replace hashes to make the
record look cleaner. A clean rerun should be stored as a new version with its
own campaign identifier and manifest chain.
