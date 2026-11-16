# PoUW's receipt folds

A receipt is how a PoUW kernel folds the words it computes into a small running state as it goes, with keyed 32-bit
multiply, xor and rotate steps. This directory is the one definition of those folds: each fold's step and key rule, the
start states and keys of the schedules that run them, and what is proved about each. Every implementation (pfill's
filler, EGEMM's kernels, E2E's PouwSegCheck statements, the r9 twin, the filler scorer's fold table) is held to it by
its published vectors. It is a definition and its checks, not code a prover or verifier runs.

## Try it

In this directory:

~~~sh
uv run python receipt_conform.py
~~~

This runs every check on the Python reference and on the C++ header, and prints how many agree for each:

~~~text
receipt_folds.py: 659 checks agree
receipt_folds.cuh (g++): 397 operations agree
~~~

The header's line needs g++.

The folds are duo, xin13, li, ram and the retired xin7.

## Conforming

A Python implementation calls the checks with its own functions:

```python
import receipt_conform as C
C.check_steps({"xin13": my_step, "duo": my_duo_step})   # receipt_folds' signatures
C.check_key_rules({"key_ok_r9": my_rule})
C.check_header("my_folds.cuh")                          # a header defining receipt_folds.cuh's rf_ functions
```

`p2ff_r9`, `pfill_ref`, `egemm_ref`, `e2e_segment` and `scorer_folds` check the existing implementations whole, each
against the schedule it runs. `pfill_cuh` builds pfill.cuh for the host with g++, as pfill_host.cpp does, once a fold build
(default, `-DPFILL_LI`, `-DPFILL_XIN13`) and runs `pf_absorb`, the build's key rule, `pf_keys` and the lane starts over
the vectors. `e2e_definitions` evaluates E2E's Boolean Definitions (PouwDuoAbsorb_v2, PouwXinAbsorb_v2
and their finals) on bits over the same vectors. `tests/benchmarks/pouw/test_receipts_adopters.py` runs them on the ones under
`benchmarks/pouw` and on E2E's, and skips `pfill.cuh`'s where there is no g++. For any module, run
`python receipt_conform.py --adopter NAME PATH`. A kernel is held to its host twin by its own gate, and the twin is held
to these vectors.
