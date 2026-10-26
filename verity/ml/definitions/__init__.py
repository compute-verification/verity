"""verity.ml.definitions: kernel Definitions (`registry`), their vectorised twins and row evaluators (`numerics`), their
Boolean versions (`boolean`), the word replay (`replay`), the proof-unit cut (`word`, `cross_call`, `call_scope`), and a
Definition's encoding and digest (`encoding`).  It builds on the catalog (`verity.ml.circuits.ops`, `.silicon`) and the
kernels (`verity.ml.kernels`); no verifier reads it, so it is off the soundness path.
"""
