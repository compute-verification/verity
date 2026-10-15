"""The node's store: one registry of everything the prover registers for the remote bandwidth audit, the hooks that keep
it current while vLLM serves, and the small interface a protocol driver talks to.

Terms are the design's (store, block, tier, registration, receipt, pass, window, commitment, opening):

- `interface`: the vocabulary and the two interfaces.  `StoreView` is all a driver (challenges, windows, chained
  seeds, openings) sees; `Lifecycle` is what the engine hooks call (weight load, the KV pool, a page's alloc, append and
  free, a pass's boundaries).  A `Codec` and a `Receipt` are per-region plugins, so the protocol can change underneath:
  the identity codec and Sloth over 2^512 − 569 are codecs, sketch-then-hash and a receipt bound to the prior tile's
  output are receipts, and none of them is the registry's.
- `registry`: `Store`, the reference implementation of both interfaces over any `Backing` (host bytes, or a CUDA
  tensor read back for an opening).  Tests drive it with a fake codec and receipt.
- `binder`: installs the hooks into a vLLM engine (`verity_vllm.engine.hooks`), all of them undone by `uninstall`.
"""
