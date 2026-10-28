# torch_gpu: the H200 demo's GPU kernels

The kernels the H200 demo serves and commits with. Each is held bit-exact to its Definition by its gates, of three
kinds:

- a SASS gate on the build: the instructions the kernel must compile to, and none it must not (no FTZ, fast-math, FMA
  contraction, spill or local memory);
- every output word against a CPU twin or reference of the Definition, with the output buffers filled with poison
  first, so a word the kernel failed to write, or wrote outside its output, shows;
- sweeps over every operand word of the instructions an exact result rests on.

A directory's README says which gates hold its kernels. Each directory is also a test suite, run on the CPU from the
tree's root:

```bash
uv sync --all-packages --extra torch-cpu
uv run tools/verity/check/suites.py benchmarks/torch_gpu/DIR
```

Each GPU script's header gives its command and the hardware it needs: CUDA's nvcc (12.8 or later), a GPU taken through
`gpu-lease`, and an `OUT` directory for what it writes. Some build on a CPU node first, into a ship the GPU run unpacks.

- `commit_slots/`: the serve-time commit call, a step's proof-block outputs as hm96-sha512 rows under hiding leaves in
  frame-v3-sha512 trees, one subtree per sequence slot. `gpu.sh` holds the GPU's roots to the CPU reference's.
- `sha512_rows/`: the SHA-512 compression engine that `commit_slots` runs on: row digests, hm96-sha512 leaves and
  frame-v3-sha512 levels. `gpu.sh` holds every variant to `verity.ml.commitments`' reference, bit for bit.
- `p2_stage/`: the store's load: P2's decode of the encoded weights store into staging buffers in GPU memory, the one
  decode the served path and `demos/h200` stage through, with the remote bandwidth audit's receipts beside it.
  `full.sh` holds every loaded tensor to the weights manifest's SHA-256 on an H200, and `bw.sh` the receipts to their
  fixture.
- `units/`: the frozen decoder's non-matmul proof blocks, each word for word its CPU twin. `session.sh` sweeps their
  helpers over every operand word and holds each kernel's words to its twin's, on an H200 or an RTX PRO 6000.
- `attn_bf16/`: attention's QK^T and PV, a fast path and a reference path, each word `GemmCoordinate_v2`'s. `run.sh`
  runs every case under poison in each tiling of both paths, on an H200.
- `attn_ref/`: a shim over `attn_bf16`'s reference path for `demos/h200` and `commit_slots/overlap.sh`.
- `s8_linear/`: the exact W8A8 linear that `ScaledMmS8Coordinate_v1` defines, three backends under one gate.
  `run.sh "run.py"` holds every cell to the twin, on an H200.
- `j10r_sm90/`: J10-R's proof blocks on the H200: fixtures from the Definition, a check that holds a build to them bit
  for bit, a bench and an exact fallback kernel. `job.sh cpu` computes the fixtures, then `job.sh gpu` runs the check
  and the bench on an H200.
