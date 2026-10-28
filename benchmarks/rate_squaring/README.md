# benchmarks/rate_squaring

How fast a node can square modulo the store codec's Sloth primes, which bounds how fast it can rebuild an encoded store:
the evidence for the bandwidth audit's (PoUB, `verity/core/protocols/remote_bandwidth`) stated squaring ceiling.
`sq.cu` (run by `sq.sh`) squares mod 2^512 − 569 (or another width, `BITS` and `C`) on the GPU's CUDA cores in every
(FOLD, ILP) form, each gated against Python's `pow` by `sq_gate.py` before it is timed, with its SASS opcode mix
(`sass_mix.py`) and Nsight Compute's counts. `ifma.c` (run by `ifma.sh`) is the host's AVX-512 IFMA rate. `gamma.py`
turns the rates into x₁ and γ.

**Status: active.** `benchmarks/word_codec/encode_rate.sh` builds and runs `sq.cu`, `sq_gate.py` and `sass_mix.py`;
`benchmarks/word_codec/kv.cu` and `benchmarks/codec_tensor/wt.cu` take their FOLD 6 form from `sq.cu`; and the audit's
`timing.py` cites `gamma.py`. Decided by `rg rate_squaring` over `verity/`, `benchmarks/`, `examples/`, `tests/` and
`integrations/`.

**Run:** `uv run tools/verity/check/suites.py benchmarks/rate_squaring` runs the GPU forms' CPU twins, the IFMA build's
gate (with `gcc`), and `gamma.py`. `sq.sh` needs one GPU and `ifma.sh` an AVX-512 IFMA host.

**Reads and writes:** `sq.sh` and `ifma.sh` write `time.jsonl`, the gate outputs, `sass-mix.json`, Nsight Compute's
CSVs and the build logs into their output directory. **Backs:** these rates are the evidence for
`timing.GPU_SQUARINGS_PER_S` and so for the stated ceiling C_sq (`timing.SQUARINGS_PER_S`), the `s` of the named
assumption `Accounting.Remote.Held.Assumptions.SquaringCeiling`, which `Accounting.Guarantees.Remote.Held` and
`HeldOverTime` take.
