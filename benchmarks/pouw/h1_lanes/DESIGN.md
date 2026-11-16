# H1 leaves proved lane by lane: design

The Python modules, beside [the README](README.md).

## The Python modules

`leaf.py` holds the leaf's protocol and its negatives, and the verifier's `verity/protocols/resource_accounting/compute/pouw/verifier/leaf_lanes.py`
names it by its path, while the Lean window's tests and `benchmarks/unit_lowering/` load it, reading the names its
`__all__` lists; `j10r.py` is the same protocol for a J10-R leaf; `served.py` is the served proof block's extract, prepare and
summary. Their test is `tests/benchmarks/pouw/test_h1_lanes.py`.
