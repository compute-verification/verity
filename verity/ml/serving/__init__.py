"""The stack-neutral serving interface (Daniel, 24 Oct): a serving stack builds its circuit from Verity's Definitions and
silicon semantics, commits in a `verity.ml.commitments` scheme, and closes and registers windows with the proof service.
vLLM is its first implementation, and another stack implements it without touching `verity/`.

It imports `verity.ml.circuits` and `verity.ml.commitments` and never `verity.core`: what core's records need from it
(the registration record, the Lean verifier's registered-reads format) comes from the caller, who passes core's module.
Nothing in core imports it, so it is off the soundness path: the verifier recomputes every domain, partition and root it
checks from the registration and its own Program.
"""
