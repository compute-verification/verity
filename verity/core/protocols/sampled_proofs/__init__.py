"""Sampled proofs: the verifier samples proof blocks from its own randomness after the prover registers its
commitments, the sampled proof blocks are proved and checked, and an accepted audit emits an `IntegrityProfile`.

Import from the owning module: `one_stage` (the one-stage audit: registration, the sample's checks, the audit record and the
profile's consumers), `coins` (the samples from the verifier's own randomness), `plan` (verification plans and
obligations over IR programs) and `consumers` (what applications read off the audit's integrity profile).  `README.md`
says what is here, where the Lean states it, and what the Lean does not state yet.
"""
