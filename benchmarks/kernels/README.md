# benchmarks/kernels

The served path's exact kernels and what a kernel build ships with. Each kernel is held bit for bit to its CPU reference or
Definition before anything is timed; nothing here changes a committed format, and nothing a verifier runs reads this code.

| Directory or file | What it holds |
|---|---|
| `s8_linear/` | the exact W8A8 linear that `ScaledMmS8Coordinate_v1` defines, three backends under one gate; `run.sh "run.py"` holds every cell to the twin, on an H200 |
| `exactlin/` | the int8-exact linear as two Triton kernels, the vLLM integration's (`verity_vllm.engine.int8_fused`: the exact quantizer, and the GEMM with the exact epilogue and per-card configs; no copy of a kernel or a table here, only the three Triton helpers `dither/dither_a.py` inlines, in `exactlin.py`), gated bit-exact against `int8_exact` (`kgate.py`) and the int8 Definitions' replay evaluators (`replay.py`), timed against stock vLLM int8 at kernel level (`kbench.py`, `ksweep.py`) and served (`served.sh`, through `dither/sbench.py`'s exact and exact-fused arms, one engine, after the kernel gate), with the served y checked bit for bit by the vLLM integration's `int8_exact_gpu` (`values.py`), the per-step kernel table (`steps.py`) and the share of a served step PoUW doesn't credit (`share.py`) |
| `moe_align/` | the Marlin MoE path's `moe_align_block_size` at the integration's vLLM pin, made a function of its inputs by `verity_vllm.engine.moe_align`: outputs run to run, each expert's pairs against the pin's op, and its cost against it |
| `numabind/` | the NUMA binding every timed kernel job runs its command under: the leased GPU's node, within the run's own CPUs, or unbound, and why (`numabind.py`, a script with no import outside the standard library); its suite holds the plan on a fake sysfs |
| `ship.sh`, `ship_clock.c`, `ship_nvcc.sh`, `ship_probe.sh` | the ship frame, below |
| `job_nvcc.sh` | the nvcc a job's own library compiles with off the card, outside a ship |

Each directory is also a test suite, run on the CPU from the tree's root:

```bash
uv sync --all-packages --extra torch-cpu
uv run tools/verity/check/suites.py benchmarks/kernels/DIR
```

Each GPU script's header gives its command and the hardware it needs: CUDA's nvcc (12.8 or later), a GPU taken through
`gpu-lease`, and an `OUT` directory for what it writes. Some build on a CPU node first, into a ship the GPU run unpacks:
those builds (`s8_linear`, and `benchmarks/torch_gpu/p2_stage`) source `ship.sh`, the one statement of a
ship's toolkit, cubin flags and layout (`toolchain.txt`, `source.txt`, `MANIFEST.sha256`). nvcc's PTX can depend on its
heap layout, on the lengths of the paths it reads and on how long its passes take, so they build with address randomization
off, from a copy of their sources and with TMPDIR at paths of one length, and with a counted clock inside nvcc's front end
(`ship_clock.c`), and the manifest names each shared library's cubins: a rebuild at the same commit, from any checkout,
with the same toolkit at the same path and the same host compiler (a `research run --cwd source` on a node with
`/usr/local/cuda-13.0`), reproduces every cubin byte for byte, on a busy machine too. They compile with `ship_nvcc.sh`,
nvcc with the three things its device compiler reads that follow a file's place in the tree fixed (the file names in the
preprocessed text's line markers, the source's path, and its directory and environment): wherever a source or a header
it includes sits, the device compiler is handed the same text and the same arguments, in the same directory, so a commit
that only moves one leaves each cubin they compile with it as it was. The program's own text is compiled as written,
so `ship_nvcc.sh` refuses one that names a file (`__FILE__`, or an assert, which expands to it, gives the name as its
`#include` spells it, which a move changes). Each build first holds its compiler to that (`ship_probe.sh`): a probe that
nvcc itself compiles to two cubins at two places must be one cubin with `ship_nvcc.sh`, and a toolkit or a wrapper with
which it isn't builds no ship. A library a job builds off the card outside a ship (`benchmarks/hash/dither/job.sh`'s)
compiles with `job_nvcc.sh`, that wrapper without the frame and its probe: its cubin
doesn't follow its place either, and nothing holds a rebuild to the same bytes from run to run.

## Interface

What code outside `benchmarks/kernels` loads. Nothing else here is an interface: any other file may change or go without a
word to another directory.

| File | What is loaded | Loaded by |
|---|---|---|
| `ship.sh` | sourced: the ship's toolkit, cubin flags and layout | `benchmarks/torch_gpu/p2_stage/build.sh` |
| `job_nvcc.sh` | run as a script: nvcc for a library a job builds off the card | `benchmarks/hash/dither/job.sh` |
| `numabind/numabind.py` | run as a script, from the tree or from a copy sent with a job | `benchmarks/hash/dither/sbench.sh`; `benchmarks/pouw/pearl_c_sm120/card.sh`; `benchmarks/pouw/pearl_c_vllm/served_step.sh` |
| `s8_linear/sass_gate.py` | loaded by file, which fixes its path, nothing more | `benchmarks/pouw/tests/test_mma_gate.py` (declared in `benchmarks/pouw/pyproject.toml`) |
| `exactlin/kbench.py` | `timed`, `copies_for`, `nvml`, `ITEM_MS` | `benchmarks/kernels/s8_linear` |
| `exactlin/kgate.py` | `MODELS` | `benchmarks/kernels/s8_linear` |
