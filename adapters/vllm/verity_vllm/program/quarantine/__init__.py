"""verity_vllm.program.quarantine — provisional numerical definitions, one sub-package per kind of op
(`collective/`), pending admission.  Existing registered bodies (`b1.py`, `prims.py`, ...) are immutable; code that
needs different exact semantics — a GELU, a different reduction order, a quantized GEMM, another attention kernel's
arithmetic — adds a NEW versioned definition here and never edits an old one.

Admission (gate G4 + `verity_vllm.properties.quarantine_lint`), all mechanical:
  1. purity: the body is a function of its declared operands and statics only — no os/env/time/random/torch/file/network
     imports, no `open`, no `global`, no reads of capture objects (lint);
  2. identity: `SEMANTIC_ID = "<lane>/<Definition>@v<k>"` at module level and a definition version >= 1; a re-derived
     body gets a new version, the old one stays (lint);
  3. recognizer tests: the sub-package's pattern has positive AND negative tests (dtype / rank / alpha / layout / constexpr
     guards each demonstrably reject) in `tests/engine/test_gen_<lane>.py` (the sealed verifier runs them);
  4. differential: a `quarantine.json` whose difftest record (`verity-gen/difftest/v1`) shows the production op and the
     Verity body bit-exact on randomized inputs including the edge kinds zeros / max / subnormal / alternating; no
     command of the tree writes one;
  5. every specialization of the definition that appears in the target run is compared at least once by Tier A (replay);
  6. only then may an accepted profile use the definition.  Promotion out of quarantine into `verity.definitions.registry` is an
     integrator move.

Sub-packages must import only `verity.circuits.*` and the standard library's math / dataclasses / typing.
"""
