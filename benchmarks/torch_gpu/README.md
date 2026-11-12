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
`gpu-lease`, and an `OUT` directory for what it writes. Some build on a CPU node first, into a ship the GPU run unpacks:
those builds (`attn_bf16`, `s8_linear` and `p2_stage`) source `ship.sh`, the one statement of a
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
which it isn't builds no ship. That leaves out the row tables' cubins in `s8_linear`'s ship, which the adapter's own
build compiles. A library a job builds off the card outside a ship (`j10r_sm90/job.sh`'s, and
`benchmarks/hash/dither/job.sh`'s) compiles with `job_nvcc.sh`, that wrapper without the frame and its probe: its cubin
doesn't follow its place either, and nothing holds a rebuild to the same bytes from run to run.

- `commit_slots/`: the serve-time commit call, a step's proof-block outputs as hm96-sha512 rows under hiding leaves in
  frame-v3-sha512 trees, one subtree per sequence slot. `gpu.sh` holds the GPU's roots to the CPU reference's.
- `sha512_rows/`: the SHA-512 compression engine that `commit_slots` runs on: row digests, hm96-sha512 leaves and
  frame-v3-sha512 levels. `gpu.sh` holds every variant to `verity.commitments`' reference, bit for bit.
- `p2_stage/`: the store's load: P2's decode of the encoded weights store into GPU memory, with PoUB's
  receipts beside it. `full.sh` holds every loaded tensor to the weights manifest's SHA-256 on an H200, and
  `bw.sh` the receipts to their fixture.
  The served path loads through it.
- `attn_bf16/`: attention's QK^T and PV, a fast path and a reference path, each word `GemmCoordinate_v2`'s. `run.sh`
  runs every case under poison in each tiling of both paths, on an H200.
- `s8_linear/`: the exact W8A8 linear that `ScaledMmS8Coordinate_v1` defines, three backends under one gate.
  `run.sh "run.py"` holds every cell to the twin, on an H200.
- `j10r_sm90/`: J10-R's proof blocks on the H200: fixtures from the Definition, a check that holds a build to them bit
  for bit, a bench and an exact fallback kernel. `job.sh cpu` computes the fixtures, then `job.sh gpu` runs the check
  and the bench on an H200.

## Interface

What code outside `benchmarks/torch_gpu` loads. Nothing else here is an interface: any other file may change or go
without a word to another directory.

| File | What is loaded | Loaded by |
|---|---|---|
| `p2_stage/stage.py` | the store's load on the GPU: `EncodedRows` (`decode_into`, `check`), `decode_matrix`, and the modules it binds as `L` (`layout.py`) and `gpu` | `adapters/vllm/verity_vllm/protocols/pous.py`, as the package `p2_stage`; `benchmarks/pous/p2_v1/bandwidth.py` |
| `p2_stage/layout.py` | `StoreLayout`, `open_store`, `segment_rows`, `R`, `ROW` | the same two |
| `p2_stage/full.py` | `Segments`, `sync`, `DEVICE` | `benchmarks/pous/p2_v1/bandwidth.py` |
| `p2_stage/build.sh` | run as a script: `libp2v1.so` for one architecture, with its manifest | `benchmarks/one_stage/served_bandwidth.sh`; `benchmarks/pous/p2_v1/bw_session.sh`; `benchmarks/pous/served_piece.sh`; `adapters/vllm/tests/protocols/pous_gpu.sh` |
| `benchmarks/torch_gpu/p2_stage/served_weights.py` | the seeded weights' generator, by its module name `benchmarks.torch_gpu.p2_stage.served_weights`: `SHAPES`, `ALIASES`, `tensors`, `generate`, `stream_bytes`, `tensor_bytes` | `benchmarks/network_traces/demo_endpoint.py`; `verity/protocols/pous/prover/p2/p2_store/store.py` |
| `commit_slots/slots.py` | the commit call's layout, domains and trees: `CommitShape`, `Group`, `Part`, `Row`, `ROW_SCHEMA`, `layout_digest`, `row_salts`, `slot_domain`, `step_domain`, `unit_domain`, `step_tree` | `benchmarks/one_stage/interfaces/verity_e2e_demo/replay_bundle.py` |

That bundle check also reads two private names of `slots.py`, `_unit_tree` and `_strings`; a private name is not an
interface. `verity/commitments/layout.py` names `commit_slots/README.md` as the commit call's specification. A test
of `benchmarks/pouw` loads `attn_bf16/sass_gate.py` and `s8_linear/sass_gate.py` by file, which fixes their paths,
nothing more. `commit_slots/fixtures/families-shared.layout.json` is the path a registered fixture is fetched to; it is
not tracked.
