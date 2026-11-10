"""Trusted code: the verifiers, every part with proven properties or on its way to them. Lean is the definition and the
verifier (AGENTS.md "Design rules"): Flock's executable verifier (`service/flock/`), and the Lean in
`primitives`, `protocols`, `service` and `devices`, which takes over each protocol from its Python verifier
(`protocols`, with the integrity profile, and `claims`) on evidence (`.agents/skills/friction/SKILL.md`); `difftest`
holds the Python that mirrors each one to the Lean. `Security/Definitions/` names the verifiers whose properties aren't
proven yet. Until its protocol switches, a Python verifier loads and reaches only what `tests/import_order.toml` froze
for it (`trusted`), which only shrinks."""
