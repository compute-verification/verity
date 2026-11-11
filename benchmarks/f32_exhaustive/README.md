# f32 reference primitives, on every input

`run.py` evaluates a primitive of `verity.definitions.registry.ref_prims` on all 2^32 binary32 words, in one of two modes.
A pass is 0 mismatches over `[0, 2^32)`.

- `log`, `exp`: the correctly rounded evaluator against MPFR's correctly rounded result (gmpy2 at 24 bits, emin -148, emax
  128, subnormalized, nearest even; a NaN is 0x7FC00000). It runs the registered evaluator itself, uncached: `log` is
  `f32_log_rn_bits` (`F32LogRn_v1`), `exp` `f32_exp_rn_bits` (`F32ExpRn_v1`).
- `boolean WORD`: the Boolean version of the word primitive `WORD` (`boolean_version`, so `F32Sqrt` checks `F32Sqrt_v2`),
  evaluated bit-sliced, against the word primitive's own evaluator. `WORD` maps one 32-bit word to one.

~~~text
uv run --with gmpy2 benchmarks/f32_exhaustive/run.py log --out DIR [--procs N] [--lo 0x0 --hi 0x100000000]
uv run benchmarks/f32_exhaustive/run.py boolean F32Sqrt --out DIR [--procs N] [--lo 0x0 --hi 0x100000000]
~~~

`DIR/exhaustive-<fn>.json` (`exhaustive-boolean-WORD.json`) holds the range, the inputs checked, the mismatch count and up to
1000 mismatches (the input and the two words). `--procs` defaults to every CPU the process may run on. The 100-digit
`decimal` evaluators make a whole range a CPU node's work, so split it with `--lo` and `--hi` across runs.

## Interface

- **Provides:** `run.py`, a command, and the result file it writes.
- **Loaded by:** nothing.
- **Loads:** `verity.definitions` (`registry`, `boolean`), `verity.circuits.evaluation` and gmpy2.
