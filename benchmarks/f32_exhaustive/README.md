# f32 reference primitives, on every input

A binary32 primitive has only 2^32 inputs, so it can be checked on all of them. This benchmark does that for a
primitive of the vLLM adapter's reference primitives (`verity_vllm.definitions.registry.ref_prims`), in one of two
modes. A pass is 0 mismatches over `[0, 2^32)`.

- `log`, `exp`: the correctly rounded evaluator against MPFR's correctly rounded result.
- `boolean WORD`: the Boolean version of the word primitive `WORD`, evaluated bit-sliced, against the word primitive's
  own evaluator.

## Try it

~~~text
verity bench f32_exhaustive boolean F32Sqrt --out DIR --lo 0x3f800000 --hi 0x3f900000
uv run --with gmpy2 benchmarks/f32_exhaustive/run.py log --out DIR [--procs N] [--lo 0x0 --hi 0x100000000]
~~~

The first checks the Boolean square root on 2^20 inputs in about a second, with no GPU. Each prints a line with the
inputs checked and the mismatches, and writes `DIR/exhaustive-<fn>.json`, which also holds up to 1000 mismatches.

The 100-digit `decimal` evaluators make a whole range a CPU node's work, so split it with `--lo` and `--hi` across
runs. `run.py`'s docstring gives MPFR's settings and every option.
