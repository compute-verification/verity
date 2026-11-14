[2026-11-14T07:42Z] f99cde51742f
STEP, Fri 13 Nov 11:42 PM PST. #9863's fixes are done at head 132ad116506a3e9156774e84ae57feb912bc199a [RAN]. Two commits on dbf5e36, and the body is replaced. The node run r20261114-073846-159b is waiting for a slot; I'll send its last line when it ends.

**Commits** [RAN]
- 5d8fe5aa010, the must-fix. `_CardSdpa` now follows torch 2.14's own selection [READ: sdp_utils.cpp, attention.cpp and attention_backward.cu at torch's commit 08187d9e0fba]. On sm_80 to sm_121, torch's default order is flash, memory-efficient, math, cuDNN; it puts cuDNN first only at major 9 or 10.
  - A 3-D call: the call on a batch of one, which is what the composite does.
  - A 4-D f16 or bf16 call with no mask: the flash op, as before.
  - Another 4-D call of one dtype among f16, bf16 and f32 that the efficient kernel's dense checks take (`_efficient_takes`): `aten._scaled_dot_product_efficient_attention`, its mask as the composite hands it over. No rule lowers it, so the step is Unsupported there, naming it.
  - Such a call the checks leave to neither kernel: `ValueError` at export, naming the call, with no kernel claimed.
  - Only a call no fused kernel takes (for example f64, or not 4-D after the 3-D step) stays on the math path.
  - `_deterministic`'s docstring names the flash backward. Under warn-only it runs non-deterministically and warns (attention_backward.cu 152–160), so the reader's INFERRED is now READ. `_CardSdpa`'s and the module's docstrings say what goes where.
  - Tests: `test_gpu_export_refuses_a_call_flash_does_not_take[a mask|f32|f32 with a mask]` and `test_gpu_export_takes_no_kernel_it_cannot_name` (f32 at D=30 raises, f64 stays on the math path, 3-D is flash). `test_gpu_export_takes_the_cards_attention_op` now has the f32 tiny step at gpu=True as 2 efficient ops plus 2 of their backwards, where it was 12 bmm.
- 132ad116506, your ruling. `Product.ret` names the head's return each product's result is: pv is the forward's 0 (O); dq, dk and dv are the backward's 0, 1 and 2. `Cells` takes the out width from that return (16), else from the dot's accumulator (32, for scores and dp).
  - `test_slowdown_code` at a cc: O differs only at the scores, and `circuit O == trace O > 0` at softmax − scores.
  - `test_the_products_are_the_steps_matrix_products` reads each product's width off the ATen op's own output dtype (`RESULTS`).
  - `test_the_attention_table_is_its_heads_census` holds each `ret` to its return's leaves equal to the product's outputs, and to bf16.

**Tests, their own summary lines** [RAN, VM, at the head]
- test_train_sdpa.py, slow tests included: `30 passed in 87.57s`.
- test_train_step.py -k "sm120 or gpu_export or structure": `6 passed, 270 deselected in 7.68s`.
- The grader's test_grader.py and test_slowdown_code.py with VERITY_GRADER_SLOWDOWN_CODE=$REPO/benchmarks/training_slowdown@132ad116506a: `39 passed in 63.48s`, none skipped.
- test_frontend_adapter_imports.py: `3 passed in 0.39s`.
- The ratchet gate, locally: `passed: ruff finds nothing new in the 4 Python file(s) it changes ...`.

**Mutations, each applied alone against those three sets** [RAN]
- M1, the mask dropped: `test_gpu_export_refuses_a_call_flash_does_not_take[a mask-kw0]`.
- M1b, the dtype dropped: 3 tests, among them `[f32-kw1]`.
- M2: 18, among them `test_flash_backward_wires_each_heads_dq_dk_dv` and `test_tiny_bf16_step_on_sm120`.
- M3: `test_the_ops_own_backward_may_read_its_state`, `test_flash_backward_refusals[not under deterministic algorithms…]` and `test_tiny_bf16_step_on_sm120_not_under_deterministic_algorithms`.
- M4: `test_flash_forward_refusals[the backward's state…]` and `test_the_ops_own_backward_may_read_its_state`.
- M5, every fused product at 16: `test_the_products_are_the_steps_matrix_products[bf16-cublas]`.
- M5b, every one at 32 (the pre-ruling width): that test and `test_the_tiny_steps_beside_the_slowdown_code[bf16-cublas]`.
- M6: `test_a_structure_derivation_is_refused_as_the_step`.

**Body** [RAN, edited]: the three wordings are fixed; circuit-check is said to add no Definition, template or lowering; the opt-in test is said to run by hand, with its value; no worker names. It now names only runs at this head, so the dbf5e36 node runs are gone from it.

**Node run** [RAN, launched]: r20261114-073846-159b at the head runs `suites.py --changed 56f6c267563 --quick` (the testing skill's pre-mark run: the 24 suites, with the changed modules kept whole, slow tests too), with the opt-in value set, then circuit-check on the four flash sidecars. I gave it `--timeout 21000 --mem-gb 32`: the research pool's default is 3600 s, and the quick adapter suite alone took 71 min on Thursday. A first launch with that default (r20261114-073702-81f1) I cancelled before it took a slot.

**Where I differ, with the line**
- Report, MUST FIX 1, "on a CUDA device those two calls go to another fused kernel ..., never the math path" [READ, sdp_utils.cpp 1185–1231]: at sm_120 cuDNN comes after math, so it is the efficient kernel, or the math path where the efficient kernel's checks refuse (a head dim not a multiple of 8 for bf16, a misaligned storage offset, a strided last dim). I refuse those at export rather than trust my mirror of the checks.
- Beyond the report: torch 2.14 also sends a 3-D call to the fused kernels (attention.cpp 766–788, `normalize_unbatched_input`), where `_CardSdpa` had left it on the math path. I made it the 4-D call on a batch of one, under your rule "only a call no fused kernel takes stays on the math path". Say if you want it refused instead.
- Your ruling's condition "a product's result is rounded to a return of the head" holds after one elementwise f32 step each [READ, the heads]: O = bf16(o / total), with the split head's combine before it; dK = bf16(acc · D^-0.5); dQ = bf16(Σ parts · D^-0.5). I read that as allowed by "Matmul.y is any wires each MAC feeds" and made the change. Unmake it if you meant the accumulator rounded alone.
- Open, not in this change: `_CardSdpa` reads no backend switch (`enable_flash_sdp(False)` and its siblings), because `export_step` traces under `sdpa_kernel(MATH)`. A step that turns flash off on the card would still be traced as flash. My recommendation: a follow-up that reads the switches before entering `sdpa_kernel(MATH)` and refuses a step with any of them off. I'll go on without it unless you say otherwise.

**StepLayout** [READ]: No, a reader of a handover doesn't need `gpu`, `tf32` and `deterministic` there. The Program's `annotations.derive.ruleset.name` already carries the cc (so gpu: `step_report` refuses a cc without `layout.gpu`), `-tf32` and `-deterministic`, and a second copy could disagree with it.

I'll wait on 159b, then replace the body's in-flight line with its last lines. I'll mark the PR myself once a reader says READY at this head and the run has passed at it.

