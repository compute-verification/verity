"""verity_vllm.program.registry.quarantine — provisional numerical definitions added by generalization lanes, one sub-package per
lane (`quarantine/<lane>/`), pending admission.  Existing registered bodies (`b1.py`, `prims.py`, ...) are immutable; a
lane that needs different exact semantics — a GELU, a different reduction order, a quantized GEMM, another attention kernel's
arithmetic — adds a NEW versioned definition here and never edits an old one.

Admission (gate G4 + `verity_vllm.properties.quarantine_lint` + `verity_vllm.properties.admission`), all mechanical:
  1. purity: the body is a function of its declared operands and statics only — no os/env/time/random/torch/file/network
     imports, no `open`, no `global`, no reads of capture objects (lint);
  2. identity: `SEMANTIC_ID = "<lane>/<Definition>@v<k>"` at module level and a definition version >= 1; a re-derived
     body gets a new version, the old one stays (lint);
  3. recognizer tests: the lane's pattern has positive AND negative tests (dtype / rank / alpha / layout / constexpr
     guards each demonstrably reject) in `tests/observe/test_gen_<lane>.py` (the sealed verifier runs them);
  4. differential: `difftest produce` (production op, GPU) + `difftest check` (Verity body) bit-exact on randomized inputs
     including the edge kinds zeros / max / subnormal / alternating -> `quarantine.json`;
  5. every specialization of the definition that appears in the target run is compared at least once by Tier A (replay);
  6. only then may an accepted profile use the definition.  Promotion out of quarantine into `b1.py` is an integrator move.

Lane sub-packages must import only `verity.ml.circuits.*` and the standard library's math / dataclasses / typing.
"""
