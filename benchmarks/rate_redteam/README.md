# benchmarks/rate_redteam

An earlier red team of the store codec's squaring rate, the adversary's side of the bandwidth audit's (PoUB,
`verity/core/protocols/remote_bandwidth`) rebuild term: the host CPUs' squaring rate with GMP (`host_sq.py`), completing a
k = 1 Sloth word from its top bits to ask whether a partial rebuild is priced per bit (`partial_root.py`), γ for p512 at
k = 2 from its inputs (`node_gamma.py`), and the GPU's squaring rate at 768, 1536 and 2048 bits (`sqfloor.cu`, run by
`sqfloor.sh` and gated by `sqfloor_gate.py`).

**Status: retire candidate.** `rg rate_redteam` over `verity/`, `benchmarks/`, `examples/`, `tests/`, `integrations/`
and `Security/` finds no importer or runner, and it has no suite. The one outside mention,
`verity/core/protocols/pouw/tests/test_pouw_supplied_inputs.py`, names a file this directory no longer has.
`benchmarks/rate_squaring` carries the work on: its `sq.cu`'s FOLD 0 is `sqfloor.cu`'s column form verbatim,
`sq_gate.py` is `sqfloor_gate.py`, and `gamma.py` restates `node_gamma.py`'s formula, its test pinning that table.

**Run:** `uv run --no-project --with gmpy2 --with python-flint python benchmarks/rate_redteam/partial_root.py` prints
one JSON line a field and reducer. `host_sq.py` (`[CORELIST [SQUARINGS]]`) runs the same way, and
`node_gamma.py` needs only the standard library. `sqfloor.sh` needs one GPU.

**Reads and writes:** nothing but its arguments and constants; the Python scripts print, and `sqfloor.sh` writes its
gate output, timings, SASS opcode mix and Nsight Compute's counts into its output directory. **Backs:** no Lean
guarantee or claim.
