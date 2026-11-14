# benchmarks/rate_squaring

How fast a node can square modulo the store codec's Sloth primes, which bounds how fast it can rebuild an encoded store:
the evidence for the bandwidth audit's (PoUB, `verity/protocols/accounting/poub/verifier`) stated squaring ceiling.
`sq.cu` (run by `sq.sh`) squares mod 2^512 − 569 (or another width, `BITS` and `C`) on the GPU's CUDA cores in every
(FOLD, ILP) form, each gated against Python's `pow` by `sq_gate.py` before it is timed, with its SASS opcode mix
(`sass_mix.py`) and Nsight Compute's counts. `ifma.c` (run by `ifma.sh`) is the host's AVX-512 IFMA rate, and `gamma.py`
turns the rates into x₁ and γ tables.

**Status: active.** `verity/protocols/accounting/poub/prover/word_codec/kv.cu` and `benchmarks/codec_tensor/wt.cu` take their FOLD
6 form from `sq.cu`, and the audit's `timing.py` cites `gamma.py`. Decided by `rg rate_squaring` over `verity/`,
`benchmarks/`, `demos/`, `tests/` and `adapters/`. `gamma.py` is a finished study.

**Run:** `verity bench rate_squaring sq|ifma [ARGS]` (`run.sh`) runs one probe's script as the command of a
`research run`: `sq` (`sq.sh`) needs one GPU and `ifma` (`ifma.sh`) an AVX-512 IFMA host.
`uv run tools/verity/check/suites.py benchmarks/rate_squaring` runs the GPU forms' CPU twins, the IFMA build's gate
(with `gcc`), `gamma.py` and the runner, on any machine.

**Writes:** a probe's records are its result, in its output directory: `time.jsonl`, the gate outputs,
`sass-mix.json`, Nsight Compute's CSVs and the build logs. No code reduces a probe's records to one number, so no
probe writes a `result.json`. **Backs:** these rates are
the evidence for `timing.GPU_SQUARINGS_PER_S` and so for the stated ceiling C_sq (`timing.SQUARINGS_PER_S`), the `s` of
the named assumption `Accounting.Remote.Held.Assumptions.SquaringCeiling`, which `Accounting.Guarantees.Remote.Held`
and `HeldOverTime` take.
