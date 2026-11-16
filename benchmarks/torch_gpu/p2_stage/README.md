# p2_stage: the store's load

The store's load is how weights get from the P2-encoded weights store, which a GEMM can't read, to the W8A8 GEMM. This
directory holds its simplest exact path, on one H200: decode a layer's matrix with P2's decode, unchanged, into a
plaintext load buffer, then run the exact GEMM on that buffer. It times the load. It holds every loaded tensor to the
weights manifest's SHA-256, and the GEMM's every output word to an independent exact path and, on samples, to the
Definition. PoUB's receipts can run beside the decode, so the load is timed with them and without.

## Try it

The CPU tests are a suite, run from the tree's root:

```bash
uv sync --all-packages --extra torch-cpu
uv run tools/check/suites.py benchmarks/torch_gpu/p2_stage
```

On an H200, `session.sh` loads one layer from a development store or from PoUS's encoded store, and `full.sh` loads the
whole encoded store, every tensor of the 52 layers and the head, read only. Each runs under `gpu-lease` from P2's
library built on a CPU node, and its header gives its command line. [`DESIGN.md`](DESIGN.md) says what each part of
the load does and checks.
