# Correctly rounded f32 primitives, on every input

`run.py` evaluates a reference primitive of `verity.ml.definitions.registry.ref_prims` on all 2^32 binary32 words and compares
each result with MPFR's correctly rounded one (gmpy2 at 24 bits, emin -148, emax 128, subnormalized, nearest even; a NaN is
0x7FC00000). It runs the registered evaluator itself, uncached: `log` is `f32_log_rn_bits` (`F32LogRn_v1`), `exp`
`f32_exp_rn_bits` (`F32ExpRn_v1`). A pass is 0 mismatches over `[0, 2^32)`.

~~~text
uv run --with gmpy2 benchmarks/f32_exhaustive/run.py log --out DIR [--procs N] [--lo 0x0 --hi 0x100000000]
~~~

`DIR/exhaustive-<fn>.json` holds the range, the inputs checked, the mismatch count and up to 1000 mismatches (input,
evaluator's word, MPFR's word). `--procs` defaults to every CPU the process may run on, and the 100-digit `decimal`
evaluators make the whole range a CPU node's work: split it with `--lo` and `--hi` across runs.
