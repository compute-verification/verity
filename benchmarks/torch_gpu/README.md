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
`gpu-lease`, and an `OUT` directory for what it writes. `p2_stage` builds on a CPU node first, into a ship the GPU run
unpacks, with the ship frame `benchmarks/kernels/ship.sh` (`benchmarks/kernels/README.md` states it).

- `commit_slots/`: what measures the serve-time commit call (`verity/commitments/cuda/commit_slots/`). `gpu.sh` holds
  the GPU's roots to the CPU reference's and times the call.
- `sha512_rows/`: what measures the SHA-512 engine the call runs on (`verity/commitments/cuda/sha512_rows/`). `gpu.sh`
  holds every variant to `verity.commitments`' reference, bit for bit, and times it.
- `p2_stage/`: the store's load: P2's decode of the encoded weights store into GPU memory, with PoUB's
  receipts beside it. `full.sh` holds every loaded tensor to the weights manifest's SHA-256 on an H200, and
  `bw.sh` the receipts to their fixture.
  The served path loads through it.
## Interface

What code outside `benchmarks/torch_gpu` loads. Nothing else here is an interface: any other file may change or go
without a word to another directory.

| File | What is loaded | Loaded by |
|---|---|---|
| `p2_stage/stage.py` | the store's load on the GPU: `EncodedRows` (`decode_into`, `check`), `decode_matrix`, and the modules it binds as `L` (`layout.py`) and `gpu` | `adapters/vllm/verity_vllm/protocols/pous.py`, as the package `p2_stage`; `benchmarks/pous/p2_v1/bandwidth.py` |
| `p2_stage/layout.py` | `StoreLayout`, `open_store`, `segment_rows`, `R`, `ROW` | the same two |
| `p2_stage/full.py` | `Segments`, `sync`, `DEVICE` | `benchmarks/pous/p2_v1/bandwidth.py` |
| `p2_stage/build.sh` | run as a script: `libp2v1.so` for one architecture, with its manifest | `benchmarks/one_stage/served_bandwidth.sh`; `benchmarks/pous/p2_v1/bw_session.sh`; `benchmarks/pous/served_piece.sh`; `adapters/vllm/tests/protocols/pous_gpu.sh` |
| `benchmarks/torch_gpu/p2_stage/served_weights.py` | the seeded weights' generator, by its module name `benchmarks.torch_gpu.p2_stage.served_weights`: `SHAPES`, `ALIASES`, `tensors`, `generate`, `stream_bytes`, `tensor_bytes` | `benchmarks/network_traces/demo_endpoint.py`; `benchmarks/pous/p2_v1/store.py` |

