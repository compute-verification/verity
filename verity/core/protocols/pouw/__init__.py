"""Proof of useful work: one protocol, named schemes behind a small interface.

Import from the owning module: `protocol` (the `PoUWScheme` interface and its data), `audit` (the commitments, the epoch
and its salt, and the tile order) and `schemes` (the named schemes, `schemes.get`).  `README.md` is the outline, and the
Lean verifier is `lean/`.  The clear lifecycle (prover, draw, tile check, profile) is a reference tool, not a verifier:
``benchmarks/pouw/clear``.
"""
