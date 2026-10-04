"""verity_vllm.program.registry.quarantine — provisional numerical Definitions that core binds, one sub-package per lane, pending
admission: `ln`, `ov_sampling`. Existing registered bodies (`b1.py`, `prims.py`, ...) are immutable; a lane that
needs different exact semantics adds a NEW versioned Definition and never edits an old one.

Admission (gate G4 + `verity_vllm.properties.quarantine_lint` + `verity_vllm.properties.admission`): the body is a function of its
operands and statics only; `SEMANTIC_ID = "<lane>/<Definition>@v<k>"` at module level; the lane's pattern has positive and negative
recognizer tests; `difftest produce` + `difftest check` are bit-exact on randomized and edge inputs; every specialization the run
uses is replayed. Promotion into `registry` is an integrator move.

The lanes core does not bind (`act`, `collective`, `dense`, `ov_moe`) are the experimental package `verity_quarantine`
(`experimental/vllm/quarantine`).
"""
