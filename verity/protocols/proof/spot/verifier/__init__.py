"""SPOT: the verifier samples proof blocks on the auditor's challenger's coins after the prover registers its
commitments, the sampled proof blocks are proved and checked, and an accepted audit emits an `IntegrityProfile`.

Import from the owning module: `one_stage` (the one-stage audit: docket, the sample's checks, the audit record and the
profile's consumers), `coins` (the samples from the verifier's own randomness) and `consumers` (what
applications read off the audit's integrity profile).  `README.md` says where each rule's docstring and Lean
are.
"""
