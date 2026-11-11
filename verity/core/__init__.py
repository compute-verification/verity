"""Trusted code: the verifiers, every part with proven properties or on its way to them. Lean is the definition and the
verifier (AGENTS.md "Design rules"): Flock's executable verifier (`service/flock/`), and the Lean in
`primitives`, `protocols`, `service` and `devices`, which takes over each protocol from its Python verifier
(`protocols`, with the integrity profile, and `claims`) on evidence (`.agents/skills/friction/SKILL.md`); `difftest`
holds the Python that mirrors each one to the Lean. `security/Definitions/` names the verifiers whose properties aren't
proven yet. A verifier imports no `verity.ml` code: until its protocol switches, a Python verifier keeps only the edges
into `verity.ml` that `tests/test_boundaries.py` froze (`CORE_ML_EDGES`), and what they reach (`SOUNDNESS_PATH`) only
shrinks. `verity.ml` never imports core."""
