# p2_stage: the staging decode

The simplest exact path from the P2-encoded weights store to the W8A8 GEMM on one H200: decode a
layer's matrix with memory-accounting's P2 decode, unchanged, into a plaintext staging buffer, then run the exact GEMM
on that buffer.

- `layout.py`: where a layer's matrices lie in the store (`StoreLayout`), read by `store_layout` from the store's
  layout JSON (memory-accounting's `p2_v1/store.py layout`, or its toy's `toy.json`): each tensor from a label boundary,
  zero past its end, and at its `staging_offset` in its layer's staging buffer. `Weights` is the plaintext, the seeded
  weights stream (`e2e_h200.served_weights`), and `payloads` lays it out as the store's W. `check_blocks` holds
  rows to it by P2's reference decode of one label (`p2_v1/store.py`'s `decode_block`), the check's ground truth.
  `open_store` takes the store: memory-accounting's encoded store (`p2_v1/store.py encode`'s segment files, read by
  `segment_rows`), under `store_salt`, `setup.draw_salt` of the coin and commitment its `setup.json` is held to; or a
  development store under a public test salt.
- `stage.py`: `decode_matrix(store, layer, tensor, out_buf)` and the fixture's `decode_layer(store, layer, staging)`,
  built on `libp2v1.so` through `p2_v1.gpu.Kernels` (keys by `p2v1_keys` v4, then `p2v1_decode`). A chunk of labels is
  decoded at a time, and each row's payload bytes are scattered to their place in the buffer. What the reference refuses
  raises `StoreError` with the fixture's flag bits: 2 for a row that isn't a codeword or isn't below p, 3 for a last
  block nonzero past its tensor's end, 0 for the library's own flag. `timed` is the one timer the drivers below use.
  `EncodedRows.decode_into` by W offset is also the GPU decode of `examples/h200`'s `P2Store` and of `verity_vllm`'s PoUS
  store, each under a layout of no tensors.
- `gemm.py`: the GEMM's check. `twin` checks every output word by an independent exact path, and `definition_words`
  evaluates `ScaledMmS8Coordinate_v1{K, GEMM_SM90_S8}` on samples.
- `layer.py`: the GPU driver for one layer (the fixture's toy, encode a development store across leases, stage, gemm at
  each batch M). Every staged tensor is held to the stream's bytes and to the weights manifest's SHA-256 (the digests
  memory-accounting held W to). `check.py` runs on the CPU afterwards: every label of the layer against the reference,
  and the samples against the Definition.
- `full.py`: the whole store a step: every tensor of the 52 layers and the head staged one at a
  time from the store's segment files (each held to its SHA-512), the last timed decode of each held to the weights
  manifest's SHA-256, the step's decode summed tensor by tensor with the peak HBM, then one layer's exact GEMM by
  `int_mm.linear` against the twin. `check.py --sample N` checks N blocks of every tensor of the store
  against the reference afterwards.
- `sketch.py`: the remote bandwidth audit's receipts beside the decode (Demo B's bandwidth leg; the interface is
  `verity/core/protocols/remote_bandwidth/PROTOCOL.md` §11, the fixture `benchmarks/pous/p2_v1/serve.py fixture`). With
  a pass attached (`store.sketch = sketch.Pass(seed, units, device)`, `units = sketch.plan_units(layout.tensors)`: the
  units the store's tensors lie in, so the encoded padding past the head is in no receipt), `EncodedRows.decode_rows`
  sketches the units of each chunk before decoding it, from the same rows on the device: σ of each unit's 8,224 stored
  bytes against the pass's 16 sign rows in one int8 GEMM (its four blocks' partials π summed in the accumulators; `Pass.partials` gives each
  block's π). At the step's end `Pass.finish` takes y (`receipts.cu`, one SHA-512 compression a unit from the seed
  block's midstate) and the root (sha512_rows' engine: each leaf's 16 receipts as a sha512/row/v2 row, its hm96-sha512
  leaf under its salt, the frame-v3-sha512 tree). `full.py --receipts` runs a pass over those units beside the
  stage pass, timed with the sketch and without; `bw_ref.py` checks it on the CPU afterwards (every y and the root
  from the pass's σ, and sampled units' σ from C*); `bw_check.py` holds the GPU to the fixture and times the receipts
  at the store's size (`bw.sh`).
- `build.sh` builds `libp2v1.so` for sm_90a on a CPU node by P2's own Makefile. `session.sh` is one run on
  vy-dc-h200 under an untimed `gpu-lease`, from memory-accounting's store (`ENCODED=1`, the layer's segment files sent
  with the run) or a development store. `full.sh` runs `full.py` on the demo card over memory-accounting's finished
  store in place, read only. Each script's header gives its command line.

The decode is libp2v1's, unchanged. Both GEMMs are `int_mm.linear` (`benchmarks/torch_gpu/s8_linear/int_mm.py`, the
exact W8A8 linear's int_mm backend): `layer.py`'s with its eager epilogue, `full.py`'s with its fused one.
