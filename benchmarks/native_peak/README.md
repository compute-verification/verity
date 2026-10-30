# native_peak: a GPU's measured dense peak

`measure.py` measures what a card sustains on dense tensor-core work for one dtype, beside the data-sheet peak for that card
and dtype. Verity takes hardware peak rates from data sheets (`benchmarks/census/hardware.json`); this is their measured
counterpart, and the native anchor that overhead figures such as `overhead.vs_native_peak` compare against. Nothing here is
a proof.

- `bf16`: square GEMMs through `torch.matmul` (cuBLAS, FP32 accumulate).
- `e4m3`: FP8 GEMMs through `torch._scaled_mm` with unit scales and FP32 output, skipped where the card has no FP8 tensor
  cores.
- `nvfp4` (sm_120 only): the block-scaled FP4 `mma.sync` instruction itself (`mma_peak_fp4.cu`), register-resident on every
  warp of every SM, so an instruction peak rather than a GEMM; a cuBLASLt NVFP4 GEMM is attempted beside it.

Each size runs warm-up GEMMs, then a sustained phase so that clocks and power settle (sampled by `nvidia-smi`), then timed
repetitions; the rate is 2n³ over their median. For `bf16` and `e4m3` a sub-block of every timed product is checked against
an FP32 reference. `nvfp4`'s check compares no reference (determinism + SASS only): its outputs are finite and bit-identical
across two launches and the kernel's SASS holds the expected E2M1 MMAs, and it records `max_rel_err` and `tolerance` as null.

On a CUDA machine with this package's `torch` extra (and nvcc for `nvfp4`), from the repository root:

~~~text
python benchmarks/native_peak/measure.py --dtype bf16 [--sizes 8192,16384] [--reps 5] [--sustain-s 2.0] --out DIR
~~~

It writes `DIR/result.json` (`native-peak/v1`: device, dtype, method, measured and data-sheet FLOP/s and their ratio, kernel,
SASS opcode, clocks, power and per-size times) and `DIR/native_peak.json` (every size's samples and checks); with `ncu` on the
machine it also keeps the SASS capture. DIR defaults to `$RESEARCH_RUN_DIR`, else `out/` beside `measure.py`.

Status: current. `tests/` holds the data-sheet table to every census line and checks the result document and the store
declaration (`tool.py`) without a GPU; one test measures a small GEMM where CUDA is present.
