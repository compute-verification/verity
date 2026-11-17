# torch_gpu: the H200 demo's GPU kernels

The kernels the H200 demo serves and commits with. Each is held bit-exact to its Definition by its gates, of three
kinds:

- a SASS gate on the build: the instructions the kernel must compile to, and none it must not (no FTZ, fast-math, FMA
  contraction, spill or local memory);
- every output word against a CPU twin or reference of the Definition, with the output buffers filled with poison
  first, so a word the kernel failed to write, or wrote outside its output, shows;
- sweeps over every operand word of the instructions an exact result rests on.

A directory's README says which gates hold its kernels. `p2_stage` is also a test suite, run on the CPU from the
tree's root:

```bash
uv sync --all-packages --extra torch-cpu
uv run tools/check/suites.py benchmarks/torch_gpu/p2_stage
```

Each GPU script's header gives its command and the hardware it needs: CUDA's nvcc (12.8 or later), a GPU taken through
`gpu-lease`, and an `OUT` directory for what it writes. `p2_stage` builds on a CPU node first, into a ship the GPU run
unpacks, with the ship frame `benchmarks/kernels/ship.sh` (`benchmarks/kernels/README.md` states it).

- `commit_slots/`: what measures the serve-time commit call (`verity/commitments/cuda/commit_slots/`). `gpu.sh` holds
  the GPU's roots to the CPU reference's and times the call.
- `sha512_rows/`: what measures the SHA-512 engine the call runs on (`verity/commitments/cuda/sha512_rows/`). `gpu.sh`
  holds every variant to `verity.commitments`' reference, bit for bit, and times it.
- `p2_stage/`: what times the store's load, P2's decode of the encoded weights store into GPU memory
  (`verity/protocols/resource_accounting/memory/poub/prover/p2_stage/`'s `stage` and `layout`), with PoUB's receipts beside it.
  `full.sh` holds every loaded tensor to the weights manifest's SHA-256 on an H200.
## Interface

Nothing outside `benchmarks/torch_gpu` loads a file of it: any file here may change or go without a word to another
directory. The store's load and P2's library's build are PoUB's and PoUS's provers' (`verity/protocols/resource_accounting/memory/poub/prover/p2_stage/`,
`verity/protocols/resource_accounting/memory/pous/prover/p2/p2_store/build.sh`), and the seeded weights' generator PoUS's
(`verity/protocols/resource_accounting/memory/pous/prover/served_weights.py`).

