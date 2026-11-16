# benchmarks/rate_squaring: design

`sq.cu` (run by `sq.sh`) squares mod 2^512 − 569 (or another width, `BITS` and `C`) on the GPU's CUDA cores in every
(FOLD, ILP) form, each gated against Python's `pow` by `sq_gate.py` before it is timed, with its SASS opcode mix
(`sass_mix.py`) and Nsight Compute's counts. `ifma.c` (run by `ifma.sh`) is the host's AVX-512 IFMA rate, and `gamma.py`
turns the rates into x₁ and γ tables.

## Who reads it

**Status: active.** `verity/protocols/resource_accounting/memory/poub/prover/word_codec/kv.cu` and `benchmarks/codec_tensor/wt.cu` take their FOLD
6 form from `sq.cu`, and the audit's `timing.py` cites `gamma.py`. Decided by `rg rate_squaring` over `verity/`,
`benchmarks/`, `demos/`, `tests/` and `adapters/`. `gamma.py` is a finished study.
