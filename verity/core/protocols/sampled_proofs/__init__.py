"""Sampled proofs: the verifier draws proof blocks from its own randomness after the prover registers its
commitments, the drawn proof blocks are proved and checked, and an accepted audit emits an `IntegrityProfile`.

Import from the owning module: `one_stage` (the one-stage audit: registration, the draw's checks, the audit record and the
profile's consumers), `coins` (the draws from the verifier's own randomness) and `consumers` (what
applications read off the audit's integrity profile).  `README.md` says what is here, where the Lean states it, and what
the Lean does not state yet.
"""
