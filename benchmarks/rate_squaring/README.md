# benchmarks/rate_squaring

How fast a node can square modulo the store codec's Sloth primes, which bounds how fast it can rebuild an encoded store:
the evidence for the bandwidth audit's (PoUB, `verity/protocols/poub/verifier`) stated squaring ceiling.
`sq.cu` (run by `sq.sh`) squares mod 2^512 − 569 (or another width, `BITS` and `C`) on the GPU's CUDA cores in every
(FOLD, ILP) form, each gated against Python's `pow` by `sq_gate.py` before it is timed, with its SASS opcode mix
(`sass_mix.py`) and Nsight Compute's counts. `ifma.c` (run by `ifma.sh`) is the host's AVX-512 IFMA rate. `cycle_fit.py`
fits a form's cycles to its instruction classes, and `gamma.py` and `wide.py` turn the rates into x₁ and γ tables.

## Interface

What code outside this directory may use. Everything else here is internal.

| Entry point | What it is |
|---|---|
| `uv run tools/verity/check/suites.py benchmarks/rate_squaring` | The suite, on the CPU: the GPU forms' CPU twins, the IFMA build's gate (with `gcc`), and `gamma.py`, `wide.py` and `cycle_fit.py`. It writes no result. |
| `bash benchmarks/rate_squaring/sq.sh`, with `BITS`, `C`, `FORMS`, `PROBES` and `TPB` | `sq.cu` on one GPU: squarings mod `2^BITS − C` in every (FOLD, ILP) form of `FORMS`, each gated against Python's `pow` before it is timed, with every kernel's SASS opcode mix and Nsight Compute's counts. It writes to `$RESEARCH_RUN_DIR/out`. |
| `bash benchmarks/rate_squaring/ifma.sh`, with `BITS` and `C` | `ifma.c` on the host's CPUs, those of the affinity mask: each compiler's build gated against Python's `pow`, then timed on one thread, on one thread a physical core and on every CPU. It writes to `$RESEARCH_RUN_DIR/out`. |
| `sq.cu`, with `sq_gate.py` and `sass_mix.py` | The squaring kernels, their gate and the opcode mix of a build, for a script that builds and gates `sq.cu` itself, as `benchmarks/word_codec/encode_rate.sh` does. |

Nothing outside `benchmarks/` imports or runs this directory. PoUB's `timing.py` names `gamma.py` by its path once, for
the host's core count.

**Status: active.** `benchmarks/word_codec/encode_rate.sh` builds and runs `sq.cu`, `sq_gate.py` and `sass_mix.py`;
`benchmarks/word_codec/kv.cu` and `benchmarks/codec_tensor/wt.cu` take their FOLD 6 form from `sq.cu`; and the audit's
`timing.py` cites `gamma.py`. Decided by `rg rate_squaring` over `verity/`, `benchmarks/`, `workloads/`, `tests/` and
`integrations/`. `gamma.py`, `wide.py` and `cycle_fit.py` are a finished study beside the parts others run.

**Run:** `uv run tools/verity/check/suites.py benchmarks/rate_squaring` runs the GPU forms' CPU twins, the IFMA build's
gate (with `gcc`), and `gamma.py`, `wide.py` and `cycle_fit.py`. `sq.sh` needs one GPU and `ifma.sh` an AVX-512 IFMA host.

**Reads and writes:** `sq.sh` and `ifma.sh` write `time.jsonl`, the gate outputs, `sass-mix.json`, Nsight Compute's
CSVs and the build logs into their output directory; `cycle_fit.py` reads `sq.sh`'s outputs. **Backs:** these rates are
the evidence for `timing.GPU_SQUARINGS_PER_S` and so for the stated ceiling C_sq (`timing.SQUARINGS_PER_S`), the `s` of
the named assumption `Accounting.Remote.Held.Assumptions.SquaringCeiling`, which `Accounting.Guarantees.Remote.Held`
and `HeldOverTime` take.
