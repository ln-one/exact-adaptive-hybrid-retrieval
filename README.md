# Exact Adaptive Hybrid Retrieval (EAHR)

Code and processed results for **Exact Adaptive Hybrid Retrieval Without Fixed
Top-L Cutoffs**.

[Paper](https://arxiv.org/abs/2608.07152) ·
[Archived release](https://doi.org/10.5281/zenodo.21968866)

## Verify results

Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/) are required.

```sh
uv sync --frozen
uv run python scripts/verify_public_paper_results.py
uv run python -m unittest discover -s tests
```

These checks use the included result records; no models or datasets are needed.

## Reproduce experiments

Follow [the reproduction guide](docs/reproduce.md) to prepare external artifacts
and run the experiments. Retrieval uses the separately maintained
[StratuMind implementation](https://github.com/ln-one/StratuMind).

- [Paper results](paper-results/): CSVs, definitions, checksums, and provenance.
- [Data preparation](docs/canonical-data-protocol.md): inputs and experiment protocol.
- [Data access](docs/data-use-register.md): upstream sources and terms.

Code is Apache-2.0 licensed. Author-generated processed data is CC BY 4.0;
third-party datasets and models retain their original terms.
See [LICENSE](LICENSE), [data license](paper-results/LICENSE.md), and
[CITATION.cff](CITATION.cff).
