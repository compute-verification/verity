"""Proof of useful work: one protocol, named schemes behind a small interface.

Import from the owning module: `protocol` (the `PoUWScheme` interface and its data), and `audit` (the commitments, the execution phase
and its salt, and the tile order).  The kernel schemes are named once, in `kernel_schemes`; their builders (`schemes.get`) and the circuits are the prover's, in `verity.protocols.accounting.compute.pouw.prover`, which core
doesn't import.  `README.md` is the outline; the verifier is the Lean under `lean/`.  The clear lifecycle (prover, sample,
tile check, profile) is a reference tool, not a verifier: ``verity/protocols/accounting/compute/pouw/prover/clear``.
"""
