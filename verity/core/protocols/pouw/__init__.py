"""Proof of useful work: one protocol, named schemes behind a small interface.

Import from the owning module: `protocol` (the `PoUWScheme` interface and its data), `audit` (the commitments, the epoch
and its salt, and the tile order) and `identifier` (the proof's identifier: the scheme, salt and workload a proof is
for).  The named schemes (`schemes.get`) and the circuits are the prover's, in `verity.ml.protocols.pouw`, which core
doesn't import.  `README.md` is the outline and `PROTOCOL.md` the spec.  The clear lifecycle (prover, draw, tile
check, profile) is a reference tool, not a verifier: ``benchmarks/pouw/clear``.
"""
