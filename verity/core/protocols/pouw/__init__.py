"""Proof of useful work: one protocol, named schemes behind a small interface.

Import from the owning module: `protocol` (the `PoUWScheme` interface and its data), `audit` (the lifecycle: commitments,
salt, prover, draw, verifier, profile), `salt` (the network certifier's per-window salt, as PoUW reads and applies it),
`identifier` (the proof's identifier: the scheme, salt and workload a proof is for) and `schemes` (the named schemes,
`schemes.get`).  `PROTOCOL.md` is the spec.
"""
