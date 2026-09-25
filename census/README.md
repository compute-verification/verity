# Census

The fixed quantities benchmarks are computed against, each with a stable id, its sources and its methodology: GPU
hardware lines (device, dtype, dense peak), networks (the 1 ms / 100 Gb/s reference and the sensitivity profiles),
workloads and frozen instance sets. It is meant to grow into the knowledge base of hardware, ML models and workloads.

- Data: `src/verity_census/data/*.json`, one file per registry. Every entry has an `id` and `sources`.
- Loader: `verity_census` (typed, stdlib only), e.g. `from verity_census import HARDWARE, REFERENCE_NETWORK, line`.
- Dependency direction: the census imports nothing from Verity, and Verity core never imports the census. The renderer,
  benchmarks and tools read it. `tests/test_census_registry.py` and `packages/verity/tests/test_boundaries.py` enforce both.
- A changed quantity is a new entry with a new id; ids already cited are never renamed.
