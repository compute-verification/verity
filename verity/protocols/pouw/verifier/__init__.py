"""Proof of useful work: one protocol, named schemes behind a small interface.

Import from the owning module: `protocol` (the `PoUWScheme` interface and its data), `audit` (the commitments, the execution phase
and its salt, and the tile order) and `identifier` (the proof's identifier: the scheme, salt and workload a proof is
for).  The named schemes (`schemes.get`) and the circuits are the prover's, in `verity.protocols.pouw.prover`, which core
doesn't import.  `README.md` is the outline; the verifier is the Lean under `lean/`.  The clear lifecycle (prover, sample,
tile check, profile) is a reference tool, not a verifier: ``verity/protocols/pouw/clear``.
"""
