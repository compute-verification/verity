# sha512_rows: design

The compression variants the engine builds and what measures it on a GPU. `README.md` says what the engine
hashes and how to run it.

## Compression variants

A variant is four digits, the pipe of each sigma function's shifts (S0, S1, s0, s1; `verity/commitments/cuda/sha512.cuh`), and
an optional `s` for straight-line rounds (`engine.variant_flags`):

- 0: funnel shifts on the integer ALU;
- 1: the high half of each rotation as mul.hi/mad.lo on the FMA pipe;
- 2: both halves on the FMA pipe;
- 3: none on the ALU (`rotr3`).

The default is `0000`; `pouw` is commit_gpu's `sha::compress`. `rows.cu` builds the row kernel in every variant in
`VARIANTS`, and `Engine(..., default=)` sets the other kernels' variant. The engine reads no settings: builds are cached
under the caller's `build` directory, and the benchmarks read both from their `SHA512_ROWS_BUILD` and `SHA512_ROWS_V`
(`benchmarks/torch_gpu/settings.py`) and pass them in.

## Measuring

What measures it is in `benchmarks/torch_gpu/sha512_rows/`:

- `torch_engine.py`: the engine with the benchmarks' build directory and variant.
- `check_gpu.py`: `check.py`'s checks on a CUDA device, into check.json.
- `bench.py`: the variant and leaf-kernel sweeps, and whole steps at the H200 demo's shape.
- `sass.py`: opcode counts per variant.
- `gpu.sh`: one leased GPU run.

