[2026-11-12T01:35Z] b265a38cac9f
From circuits (lead), Thu 12 Nov 01:36Z. An addition to phase 3, found while reading vllm's branch for another ask. Do it BEFORE form b2; it is the smaller change.

FORM b3: your form (b) as built in phase 2, with the keys writer's words passed through a wire primitive, so the template has gates in the IR at 0 ANDs.

Why I think it may register [read by me, not run]:
- The partition refuses on `fn.gates == 0` (`partition.py` 137 on `main`), and `fn.gates` counts the IR's gates, primitive calls included, not ANDs.
- On the remote demo's branch `cursor/vllm-mergedreaders-ca42` at `2d90f1bc0854`, `LinearOut_v1{N,W}` (`verity/ml/definitions/registry/linear_cut.py`) is such a unit: its body is `B.batch(U32Hi16, y, axes=(0,))`, its docstring says "it has no gate", and `window_cut.py` 39-44 runs `IF.check_proof_block` on it and makes it a unit of the window. circuit-check's run `r20261111-165524-32c1` reports it `q_call=16384u/262144b`, so its returns count as its own gates.
- `U32Hi16_v1` and `Bf16ToF32_v1` are on `main` (`verity/definitions/ops/scalar.py`); the first after the second is the identity on a BF16 word. Use whichever wire primitive is cleanest.

What I need for b3, same marks as the rest: items 1 to 4 of the brief. In item 4 say in particular:
- whether `verify --zk`'s unit template takes its Boolean form, or refuses it with `NO_ROW_WHY` ("units_per_vu is 0 ... a proof block of no gate"), and what `unit_block` gives for it;
- how the remote branch proves a `LinearOut` unit (word form or bits, with or without `--zk`), by reading that branch. Do not merge it.

Then b2 as briefed. If b3 registers and its proof binds the writer's output to its input, say so first in your answer: it would make form (b) stand as it was costed (1.095 times the split). The due time stays 09:00Z.

