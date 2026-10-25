# p2_stage: the staging decode

The simplest exact path from the P2-encoded weights store to the W8A8 GEMM on one H200 (Linear HASH-12): decode a
layer's matrix with memory-accounting's P2 decode, unchanged, into a plaintext staging buffer, then run the exact GEMM
on that buffer.

- `layout.py`: where a layer's matrices lie in W (`StoreLayout`: PoUS's `Layout`, the key prefix, the tensor names),
  read from a manifest by `layout_of`. The layout stays a parameter until CMP-95 item 2's fixture lands. Until then it
  is the seeded weights stream (`verity_torch_loop.weights`) back to back in p2_v1's own block format. `decode_block`
  is P2's reference decode of one block, the check's ground truth.
- `stage.py`: `decode_matrix(store, layer, tensor, out_buf)`, built on `libp2v1.so` through `p2_v1.gpu.Kernels` (keys
  by `p2v1_keys` v4, then `p2v1_decode`). A chunk of blocks is decoded at a time, and each row's payload bytes are
  scattered to their place in `out_buf`. Rows that aren't codewords or aren't below p are refused (`StoreError`).
- `gemm.py`: `scaled_mm`, which is `torch._int_mm` plus the Definition's epilogue and stands in for HASH-11's
  KS8-SIMPLE. `twin` checks every output word by an independent exact path, and `definition_words` evaluates
  `ScaledMmS8Coordinate_v1{K, GEMM_SM90_S8}` on samples.
- `layer.py`: the GPU driver for one layer (encode a development store across leases, stage, gemm). `check.py` runs on
  the CPU afterwards: every encoded block against the reference, and the samples against the Definition.
- `build.sh` builds `libp2v1.so` for sm_90a on a CPU node by P2's own Makefile. `session.sh` runs one session on
  vy-dc-h200 under `gpu-lease`. Each script's header gives its command line.
