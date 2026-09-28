"""Trusted code: the verifiers, every part with proven properties or on its way to them. C-Flock's executable verifier of
record (`service/flock/`, Lean), the Python verifiers (`protocols`, with the integrity profile, and `claims`) and their Lean
ports (`primitives`, `protocols`, `service`, `devices`, each held to today's Python by `difftest`). `Security/Models/` names
the verifiers whose properties aren't proven yet. Core imports the parts of `verity.ml` that `tests/test_boundaries.py`'s
`ALLOWED` lists, which are on the soundness path because core's verifiers read them; `verity.ml` never imports core."""
