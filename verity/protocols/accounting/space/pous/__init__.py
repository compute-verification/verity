"""PoUS, proof of useful space for model weights, as a spec-level reference: the schemes behind one codec API (the band
first, then dense, P2 and P3), setup, the timed audit, and the memory challenger that signs its rounds.  It matches the
Lean models' definitions and is the oracle for the GPU kernels.

Import from the owning module: `params`, `labels` (the bit layout), `oracle` (queries, programs and their rounds),
`primitives` (`H`, `P_t` and their choices), `scheme` (encode, decode, per-segment decode), `layout` (tensors in `W`,
per-tensor decode), `setup` (commit to `W`, the verifier's salt and `vk`), `verifier` (`vk`: per-block digests and the
Merkle root), `audit` (the challenge key, responders and the unsigned timed rounds) and `adversary` (partial holdings
replayed against the real labels).  `protocol` is the generic protocol over any scheme (abstract `Scheme`,
`Certificate`, `TimedAudit`, `StorageProfile`), and `schemes` the registered schemes behind it (the band first).
`memory_challenger` is the memory challenger, which signs its rounds (`MemoryChallenger`).
`PROTOCOL.md` says what is here, and `DISCREPANCIES.md` where the Lean model and the prose spec differ.
"""
