"""Proof of useful work: one protocol, named schemes behind a small interface.

Import from the owning module: `protocol` (the `PoUWScheme` interface and its data), `audit` (the commitments, the execution phase
<<<<<<< 94ca7fd0d18
and its salt, and the tile order), `identifier` (the proof's identifier: the scheme, salt and workload a proof is for)
and `schemes` (the named schemes, `schemes.get`).  `README.md` is the outline; the verifier is the Lean under `lean/`.  The clear lifecycle (prover, sample, tile
check, profile) is a reference tool, not a verifier: ``benchmarks/pouw/clear``.
||||||| e10cf2c8559
and its salt, and the tile order), `identifier` (the proof's identifier: the scheme, salt and workload a proof is for)
and `schemes` (the named schemes, `schemes.get`).  `README.md` is the outline and `PROTOCOL.md` the spec.  The clear lifecycle (prover, sample, tile
check, profile) is a reference tool, not a verifier: ``benchmarks/pouw/clear``.
=======
and its salt, and the tile order) and `identifier` (the proof's identifier: the scheme, salt and workload a proof is
for).  The named schemes (`schemes.get`) and the circuits are the prover's, in `verity.ml.protocols.pouw`, which core
doesn't import.  `README.md` is the outline; the verifier is the Lean under `lean/`.  The clear lifecycle (prover, sample,
tile check, profile) is a reference tool, not a verifier: ``benchmarks/pouw/clear``.
>>>>>>> 7af4d24fca3e
"""
