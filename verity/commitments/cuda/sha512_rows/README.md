# sha512_rows: the serve-time commit's SHA-512 engine

Committing a serving step's values means hashing them with SHA-512, and this is the device engine that does it, bit for
bit the CPU reference's (`verity.commitments`). It computes:

- the sha512/row/v2 digest of each row: x = SHA-512(P(8 nbytes) ‖ row), `rowleaf.sha512_row_v2_digest`;
- its hm96-sha512 hiding leaf: b = x ⊕ M·y, c = SHA-512(salt prefix ‖ y), h = SHA-512(leaf prefix(key) ‖ b ‖ c), `hm96.SHA512`;
- that leaf's frame-v3-sha512 leaf, and frame-v3-sha512 nodes and pads a level at a time, `merkle.CommitmentDomain`.

`commit_slots`, beside this directory, runs on it: the engine is its inner "hash these rows" and "hash this level"
steps. `engine.py`'s docstrings are the call; `Engine("host", build=DIR)` is a host twin (g++) of the same kernels. The
compression is `sx::compress<V, ROLL>` from the GPU's one SHA-512 header (`verity/commitments/cuda/sha512.cuh`), and
[`DESIGN.md`](DESIGN.md) lists its variants.

## Try it

~~~sh
uv run tools/check/suites.py verity/commitments/cuda/sha512_rows
~~~

It holds the host twin to the reference in a few seconds; every variant on a CUDA device runs where there is one.

On a CUDA GPU (an H200 for the H200 layer) with CUDA torch, CUDA's nvcc and `gpu-lease`, the checks and benchmarks
run from the tree's root as one leased run:

```bash
OUT=DIR bash benchmarks/torch_gpu/sha512_rows/gpu.sh
```
