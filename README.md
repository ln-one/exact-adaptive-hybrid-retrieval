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

