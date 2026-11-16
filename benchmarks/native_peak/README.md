# native_peak: a GPU's measured dense peak

Verity takes a card's peak rates from its data sheet (`benchmarks/census/hardware.json`). This benchmark measures what
the card really sustains on dense tensor-core work for one datatype, beside the data-sheet peak for that card and
datatype. It is the native anchor that overhead figures such as `overhead.vs_native_peak` compare against. Nothing
here is a proof.

It measures one datatype a run:

- `bf16`: square GEMMs through `torch.matmul` (cuBLAS, FP32 accumulate).
- `e4m3`: FP8 GEMMs through `torch._scaled_mm`, skipped where the card has no FP8 tensor cores.
- `nvfp4` (sm_120 only): the FP4 `mma.sync` instruction itself, so an instruction peak rather than a GEMM; a cuBLASLt
  NVFP4 GEMM is attempted beside it. Its check compares no reference: its outputs are finite and bit-identical across
  two launches and the kernel's SASS holds the expected E2M1 MMAs, and it records `max_rel_err` and `tolerance` as
  null.

## Try it

On a CUDA machine with this package's `torch` extra (and nvcc for `nvfp4`), from the repository root:

~~~text
python benchmarks/native_peak/measure.py --dtype bf16 [--sizes 8192,16384] [--reps 5] [--sustain-s 2.0] --out DIR
~~~

It writes `DIR/result.json`, whose top level holds the measured and the data-sheet FLOP/s and their ratio, and
`DIR/native_peak.json`, every size's samples and checks. `measure.py`'s docstring says how each size is warmed,
sustained and timed, and how each product is checked.

`tool.py` is the evidence store's declaration of the result, and the store's tool registry loads it by name. The
tests hold the data-sheet table to every census line and check the result document without a GPU.
