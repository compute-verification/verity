"""The bandwidth audit v0: a window certifies how much of a PoUS store a device read from its memory before a deadline.

The verifier sends a fresh seed `σ`; before `T` the prover tags every block of the store `E` (P2's `C*`, under PoUS's
`vk`), `y_j = Tag(σ, j, E_j)`, and commits the tags; the verifier then samples `s` blocks and checks each opened tag
against the block and its path to `vk`'s root.  `PROTOCOL.md` is the protocol and what it certifies.

- `tags`: the three tag constructions, `ro-blake3/v1` and `ro-sha512/v1` (the random-oracle tags) and `gf31-poly/v1`
  (`hbm_audit`'s polynomial hash over GF(2^31 − 1), keyed per block, uncertified), and the tags' tree.
- `audit`: the window (`Verifier`, `Window`), the prover's reference (`Store`, `tag_pass`, `TagCommitment.open`) and
  the bound an accepted window certifies (`bound`).

It reuses PoUS's store, setup and `vk` as they are (`verity.core.protocols.pous`).  The GPU tagger is
`benchmarks/bandwidth`.  The bandwidth audit on the served path is `verity.core.protocols.remote_bandwidth`; this v0
stays while `verity_vllm.protocols.remote_bandwidth`'s tag receipt reads `tags`.
"""
