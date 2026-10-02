"""Sampled proofs (decision 41): the verifier draws proof units from its own randomness after the prover registers its
commitments, the drawn units are proved and checked, and an accepted audit emits an `IntegrityProfile`.

Import from the owning module: `one_stage` (the one-stage audit: registration, the draw's checks, the audit record and the
profile's consumers), `law` (the two-stage law and its draws) and `plan` (verification plans and obligations over IR
Programs).  `README.md` says what is here, where the Lean states it, and what the Lean does not state yet.
"""
