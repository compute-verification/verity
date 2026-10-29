"""verity.ml.definitions: kernel Definitions (`registry`), their vectorised twins and row evaluators (`numerics`), their
Boolean versions (`boolean`), the word replay (`replay`), the proof-block cut (`word`, `cross_call`, `call_scope`), a
Definition's encoding and digest (`encoding`), and the evidence ids their conformance texts and provenance records cite
(`evidence`).  It builds on the catalog (`verity.ml.circuits.ops`, `.silicon`) and the kernels (`verity.ml.kernels`); no
verifier reads it, so it is off the soundness path.
"""
