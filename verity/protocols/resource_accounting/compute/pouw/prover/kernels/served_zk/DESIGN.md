# Served Pearl-C under zero knowledge: design

The replay commands and the Python modules, beside [the README](README.md).

## Other commands

`benchmarks/pouw/served_zk/`'s `replay_core.py OUT --runs DIR...` and `replay_window.py OUT --runs DIR...` replay
recorded windows through PoUW's check and the window's Lean port with their Python references, each case against the
verdict the run recorded.

## The Python modules

`served_commit.py` is `job`'s commitment, challenger and check, which `job` runs as
`benchmarks/pouw/served_zk/served_commit_run.py`; `stage_tile.py` stages one sampled tile as a Flock `--zk` unit, run as
`benchmarks/pouw/hidden_zk/served_stage.py` with its Flock staging; `statement.py` commits a served pass's rows in the
statement's format: `pearl_c_vllm/e2e.py` loads it with its device tensors and gate
(`benchmarks/pouw/served_zk/statement_torch.py`), and the device adapter's
Pearl-C graphs take its `Statement`; `rows_bench.py` is the GPU
row commitments `statement.py` loads; `of_record.py` runs each of PoUW's checks on the challenger's path in Lean and in
its Python reference on one input, its Lean side from `benchmarks/pouw/served_zk/of_record_lean.py`. What verity does
not import (`verity_difftest`, the device library, `fast_tile`, the evidence store) the benchmark-side files pass in or supply.
`rows_bench.py` and `statement.py` list in `__all__` the names other files read, the tests' included. Their tests are
the `test_served_zk*.py` files under `tests/benchmarks/pouw/`.
