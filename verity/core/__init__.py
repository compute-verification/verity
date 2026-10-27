"""Trusted code: the verifiers, every part with proven properties or on its way to them. Lean is the definition and the
verifier of record (AGENTS.md "Design rules"): C-Flock's executable verifier (`service/flock/`), and the
Lean in `primitives`, `protocols`, `service` and `devices`, which takes over each protocol from its Python verifier
(`protocols`, with the integrity profile, and `claims`) only on evidence (AGENTS.md); `difftest` holds the Python that mirrors
each one to the Lean. `Security/Definitions/` names the verifiers whose properties aren't proven yet. A verifier imports no
`verity.ml` code: until its protocol switches, a Python verifier keeps only the edges into `verity.ml` that
`tests/test_boundaries.py` froze (`CORE_ML_EDGES`), and what they reach (`SOUNDNESS_PATH`) only shrinks. `verity.ml` never
imports core."""
