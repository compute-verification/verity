"""P3, the two-layer invertible labelling with a trusted encoder, as a spec-level reference: Lean's `p3Scheme`
(`lean/submissions/p3-meets/P3Meets.lean` over `p3Model'`), parametrized.

Import from the owning module: `params`, `labels` (the bit layout), `oracle` (queries, programs and their rounds),
`primitives` (`H`, `P_t` and their choices), `scheme` (encode, decode, per-segment decode), `layout` (tensors in `W`,
per-tensor decode), `setup` (commit to `W`, the verifier's salt and `vk`), `verifier` (`vk`: per-block digests and the
Merkle root), `audit` (the challenger, responders and the memory challengers) and `adversary` (partial holdings replayed
against the real labels).  `protocol` is the generic protocol over any scheme (abstract `Scheme`, `Certificate`,
`MemoryChallenger`, `StorageProfile`), and `schemes` the registered schemes behind it (the band first).
`PROTOCOL.md` says what is here, and `DISCREPANCIES.md` where the Lean model and the prose spec differ.
"""
