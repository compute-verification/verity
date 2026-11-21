# How fast a node squares

How fast a node can square modulo the store codec's Sloth primes bounds how fast it can rebuild an encoded store. These
measurements are the evidence for the bandwidth audit's (PoUB, `verity/protocols/accounting/memory/poub/verifier`)
stated squaring ceiling: the GPU's CUDA cores in every form, each gated against Python's `pow` before it is timed, and
the host's AVX-512 IFMA rate.

**Backs:** these rates are
the evidence for `timing.Figures.gpu_squarings_per_s` (`benchmarks/poub_session/sheets.py`'s `squaring_ceilings`, which
takes at 512 bits the fastest of `gamma.GPU`'s variants, measured or modelled, over `gamma.py`'s reference at its clock,
scales it to the card's highest clock reading and adds the host's rate once, and refuses a width or a card with no
established variant) and
so for the stated maximum squaring rate C_sq (`timing.Figures.squarings_per_s`), the `s` of
the named assumption `ResourceAccounting.Remote.Held.Assumptions.SquaringCeiling`, which `ResourceAccounting.Guarantees.Remote.Held`
and `HeldOverTime` take. The handed rate comes from a modelled variant and is above every timed one, so the rebuild
share it gives is a bound from above, not a measurement.

## Try it

~~~sh
uv run python calibration/rate_squaring/gamma.py
~~~

It turns the measured rates into x₁ and γ, one row per squaring form and host: no GPU needed.

| Command | Needs | Writes into its output directory (`run.sh`, each the command of a `research run`) |
| --- | --- | --- |
| `uv run tools/check/suites.py calibration/rate_squaring` | the CPU, with `gcc` | nothing: the GPU forms' CPU twins, the IFMA build's gate, and `gamma.py` |
| `verity calibrate rate_squaring sq` | one GPU | `time.jsonl`, the gate outputs, `sass-mix.json`, Nsight Compute's CSVs and the build logs, and the gates' validation as the run's `result.json` |
| `verity calibrate rate_squaring ifma` | an AVX-512 IFMA host | the same, and the run's `result.json` with metrics.toml's number, `memory-accounting/host-squaring-rate` |
