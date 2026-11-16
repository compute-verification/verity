# torch_gpu

This directory measures GPU code that serving runs beside the model: the serve-time commit call, the SHA-512
engine that call runs on, and the store's load, which decodes the encoded weights store into GPU memory. The code
itself lives elsewhere, under `verity/commitments/cuda/` and in PoUB's and PoUS's provers; only its measurement is
here, and nothing outside this directory loads a file of it.

Each kernel is held bit-exact to its Definition or reference by gates of three kinds before it is timed:

- a SASS gate on the build: the instructions the kernel must compile to, and none it must not (no FTZ, fast-math, FMA
  contraction, spill or local memory);
- every output word against a CPU twin or reference, with the output buffers filled with poison first, so a word the
  kernel failed to write, or wrote outside its output, shows;
- sweeps over every operand word of the instructions an exact result rests on.

## Try it

`p2_stage` is also a test suite, run on the CPU from the tree's root:

```bash
uv sync --all-packages --extra torch-cpu
uv run tools/check/suites.py benchmarks/torch_gpu/p2_stage
```

The GPU runs need CUDA's nvcc (12.8 or later), a GPU taken through `gpu-lease`, and an `OUT` directory for what they
write; each script's header gives its command and its hardware. `p2_stage` builds on a CPU node first, into a ship the
GPU run unpacks, with the ship frame `benchmarks/kernels/ship.sh`
([`benchmarks/kernels/DESIGN.md`](../kernels/DESIGN.md) states it).

| Directory | What it measures |
| --- | --- |
| `commit_slots/` | The serve-time commit call: `gpu.sh` holds the GPU's roots to the CPU reference's and times the call. |
| `sha512_rows/` | The SHA-512 engine: `gpu.sh` holds every variant to `verity.commitments`' reference, bit for bit, and times it. |
| [`p2_stage/`](p2_stage/README.md) | The store's load, with PoUB's receipts beside it, on an H200. |
