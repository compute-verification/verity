# benchmarks/kernels: design

## The ship frame

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
| `ship.sh` | sourced: the ship's toolkit, cubin flags and layout | `verity/protocols/resource_accounting/memory/pous/prover/p2/p2_store/build.sh` |
| `job_nvcc.sh` | run as a script: nvcc for a library a job builds off the card | `benchmarks/hash/dither/job.sh` |
| `numabind/numabind.py` | run as a script, from the tree or from a copy sent with a job | `benchmarks/hash/dither/sbench.sh`; `benchmarks/pouw/pearl_c_sm120/card.sh`; `benchmarks/pouw/pearl_c_vllm/served_step.sh` |
| `s8_linear/sass_gate.py` | loaded by file, which fixes its path, nothing more | `tests/benchmarks/pouw/test_mma_gate.py` (declared in `tests/benchmarks/pouw/pyproject.toml`) |
| `exactlin/kbench.py` | `timed`, `copies_for`, `nvml`, `ITEM_MS` | `benchmarks/kernels/s8_linear` |
| `exactlin/kgate.py` | `MODELS` | `benchmarks/kernels/s8_linear` |
