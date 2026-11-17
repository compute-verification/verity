# How fast a node squares

How fast a node can square modulo the store codec's Sloth primes bounds how fast it can rebuild an encoded store. These
measurements are the evidence for the bandwidth audit's (PoUB, `verity/protocols/resource_accounting/memory/poub/verifier`)
stated squaring ceiling: the GPU's CUDA cores in every form, each gated against Python's `pow` before it is timed, and
the host's AVX-512 IFMA rate.

**Backs:** these rates are
the evidence for `timing.GPU_SQUARINGS_PER_S` and so for the stated ceiling C_sq (`timing.SQUARINGS_PER_S`), the `s` of
the named assumption `ResourceAccounting.Remote.Held.Assumptions.SquaringCeiling`, which `ResourceAccounting.Guarantees.Remote.Held`
and `HeldOverTime` take.

## Try it

~~~sh
uv run python benchmarks/rate_squaring/gamma.py
~~~

It turns the measured rates into x₁ and γ, one row per squaring form and host: no GPU needed.

| Command | Needs | Writes into its output directory |
| --- | --- | --- |
| `uv run tools/check/suites.py benchmarks/rate_squaring` | the CPU, with `gcc` | nothing: the GPU forms' CPU twins, the IFMA build's gate, and `gamma.py` |
| `sq.sh` | one GPU | `time.jsonl`, the gate outputs, `sass-mix.json`, Nsight Compute's CSVs and the build logs |
| `ifma.sh` | an AVX-512 IFMA host | the same |
