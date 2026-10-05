"""verity_quarantine — the vLLM registry's quarantine lanes that core does not bind, moved out of
`verity_vllm.program.registry.quarantine` (note:20261004T2058Z-draft-repo-organization-principles): `act`, `collective`, `dense`, `ov_moe`.
The lanes core bound (`ln`, `ov_sampling`) were promoted into `verity_vllm.program.registry`.

Each lane is a sub-package of provisional numerical Definitions pending admission, or of re-exports of Definitions already promoted
to `verity_vllm.program.registry`, with a `*_difftest.py` adapter for `verity-vllm properties-admission` beside each. Admission
is as in core's quarantine (`verity_vllm.properties.quarantine_lint`, `verity_vllm.properties.admission`): a body is a function of
its operands and statics only, each module names its `SEMANTIC_ID = "<lane>/<Definition>@v<k>"`, a re-derived body takes a new
version, and promotion into the registry is an integrator move. Nothing in core imports this package.
"""
