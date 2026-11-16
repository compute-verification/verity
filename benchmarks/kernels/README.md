# benchmarks/kernels

This is where the served path's exact kernels are measured. Each is held bit for bit to its CPU reference or its
Definition before anything is timed. Nothing here changes a committed format, and nothing a verifier runs reads this code.

It also holds what a kernel build ships with: the ship frame, which lets a GPU build made off the card, on a CPU node,
be rebuilt byte for byte.

## Try it

Every directory here but `moe_align` is also a test suite, run on the CPU from the tree's root. It needs no GPU:

```bash
uv sync --all-packages --extra torch-cpu
uv run tools/check/suites.py benchmarks/kernels/DIR
```

DIR is `exactlin`, `numabind` or `s8_linear`. The GPU scripts each give their command and the hardware they need in
their header: CUDA's nvcc (12.8 or later), a GPU taken through `gpu-lease`, and an `OUT` directory for what they write.

| Directory | What it measures |
| --- | --- |
| [`s8_linear/`](s8_linear/README.md) | The exact W8A8 linear that `ScaledMmS8Coordinate_v1` defines, in three backends held to one gate. |
| [`exactlin/`](exactlin/README.md) | The int8-exact linear as the vLLM adapter serves it, two Triton kernels, timed against stock vLLM int8. |
| [`moe_align/`](moe_align/README.md) | vLLM's `moe_align_block_size` made a function of its inputs on the Marlin MoE path. |
| `numabind/` | The NUMA binding every timed kernel job runs its command under, and why. |

## Ships

A build made off the card goes into a ship that the GPU run unpacks. Those builds (`s8_linear`, and
`benchmarks/torch_gpu/p2_stage`) source `ship.sh`, the one statement of a ship's toolkit, cubin flags and layout. A
rebuild at the same commit, with the same toolkit at the same path and the same host compiler, reproduces every cubin
byte for byte. [`DESIGN.md`](DESIGN.md) says how the frame makes that hold, and which files here code outside this
directory loads.
