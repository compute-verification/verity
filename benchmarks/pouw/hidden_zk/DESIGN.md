# Pearl-C4's hidden checks under zero knowledge: design

The Python modules, beside [the README](README.md).

## The Python modules

`stage.py ROLE K N OUT` stages one size class (`row_a`, `row_b`, `tile`) through `verity_flock.class_statement`;
`fast_stage.py` and `compact_stage.py` make that staging take less time and memory while writing the same bytes;
`summarize.py OUT` folds what `job` leaves into its summaries; `timed.py OUT_FILE CMD...` records a command's wall
seconds, peak memory and CPU seconds. `core3_scratch.py` and `hash_scratch.py` are development bindings, Flock pieces
for primitives Flock has none for yet, and every result staged through them is labelled so. `h1_prove/stage.py` and
`served_stage.py` (`../../../verity/protocols/resource_accounting/compute/pouw/prover/kernels/served_zk/stage_tile.py` with its Flock
staging) import this directory's modules, and `compact_stage.py`, `fast_stage.py`, `core3_scratch.py`
and `hash_scratch.py` list in `__all__` the names other files read, the tests' included. Their tests are
`test_hidden_zk_stage.py` and `test_hidden_zk_summarize.py` under `tests/benchmarks/pouw/`.
