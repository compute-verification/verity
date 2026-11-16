# H1 receipts beside vLLM's int8 GEMM: design

The Python modules, beside [the README](README.md).

## The Python modules

`serve.py`, `j10r_time.py`, `step_time.py` and `h1_gpu.py` (the vLLM adapter's `pouw_h1_gpu.py` with `producer.py`
handed to it) are the jobs' programs. `coins.py` (the live coins and their check),
`commit.py` (the window's tables), `credit.py` (the padded-window credit), `producer.py` (`h1u.cu`'s units kernel),
`partials.py` and `lane_refs.py` (option A's lane partials) are what they import. Other directories load some of them by
path: `verity/protocols/proofs/spot/verifier/challenger/challenger_session.py` reads `coins.py`, and `benchmarks/hash/dither/` reads `coins.py`, `commit.py`, `credit.py`, `serve.py` and `lane_refs.py`.
Each of these modules' `__all__` lists the names other files read through it, the tests' included, except underscore
names; anything else in it may change without notice. Their tests are the `test_served_h1*.py` files,
`test_partials.py` and `test_lane_refs.py` under `tests/benchmarks/pouw/`.
