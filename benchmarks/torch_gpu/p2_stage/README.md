# p2_stage: the staging decode

The simplest exact path from the P2-encoded weights store to the W8A8 GEMM on one H200 (Linear HASH-12): decode a
layer's matrix with memory-accounting's P2 decode, unchanged, into a plaintext staging buffer, then run the exact GEMM
on that buffer.

- `layout.py`: where a layer's matrices lie in the store (`StoreLayout`), read by `store_layout` from CMP-95 item 2's
  layout JSON (memory-accounting's `p2_v1/store.py layout`, or its toy's `toy.json`): each tensor from a block boundary,
  zero past its end, and at its `staging_offset` in its layer's staging buffer. `Weights` is the plaintext, the seeded
  weights stream (`verity_torch_loop.weights`), and `payloads` lays it out as the store's W. `decode_block` is P2's
  reference decode of one block, the check's ground truth. `open_store` takes the store: memory-accounting's encoded
  store (`p2_v1/store.py encode`'s segment files, read by `segment_rows`), under `store_salt`, `setup.draw_salt` of the
  coin and commitment its `setup.json` is held to; or a development store under a public test salt.
- `stage.py`: `decode_matrix(store, layer, tensor, out_buf)` and the fixture's `decode_layer(store, layer, staging)`,
  built on `libp2v1.so` through `p2_v1.gpu.Kernels` (keys by `p2v1_keys` v4, then `p2v1_decode`). A chunk of blocks is
  decoded at a time, and each row's payload bytes are scattered to their place in the buffer. What the reference refuses
  raises `StoreError` with the fixture's flag bits: 2 for a row that isn't a codeword or isn't below p, 3 for a last
  block nonzero past its tensor's end, 0 for the library's own flag.
- `gemm.py`: `scaled_mm`, which is `torch._int_mm` plus the Definition's epilogue and stands in for HASH-11's
  KS8-SIMPLE. `twin` checks every output word by an independent exact path, and `definition_words` evaluates
  `ScaledMmS8Coordinate_v1{K, GEMM_SM90_S8}` on samples.
- `layer.py`: the GPU driver for one layer (the fixture's toy, encode a development store across leases, stage, gemm at
  each batch M). Every staged tensor is held to the stream's bytes and to the weights manifest's SHA-256 (the digests
  memory-accounting held W to). `check.py` runs on the CPU afterwards: every block of the layer against the reference,
  and the samples against the Definition.
- `build.sh` builds `libp2v1.so` for sm_90a on a CPU node by P2's own Makefile. `session.sh` runs one session on
  vy-dc-h200 under an untimed `gpu-lease`, from memory-accounting's store (`ENCODED=1`, the layer's segment files sent
  with the run) or a development store. Each script's header gives its command line.

KP2-DEC (#4453, f254735dd2) staged a layer with its own kernel and measured it beside this path: its `decode_p2v1` arm
is the same libp2v1 decode plus a payload copy. This directory uses that library unchanged and none of KP2-DEC's kernels. The
GEMM stays `torch._int_mm` plus the epilogue until KS8-SIMPLE (#4472) is on `main`.
