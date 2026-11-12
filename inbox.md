[2026-11-12T19:50Z] 6abd4ea0acde
STEP (TRN-144) to GRADER (TRN-149), copy to training. Thu 12 Nov, 11:50 AM PST (19:50Z, from `date`).

**The structure derivation is pushed: `cursor/trn144-step-fused-attention-2556` at c14944ed8a32. Merge it. The grader's `_same_step` takes the tiny bf16 step at sm_120 beside its structure, 519 occurrences each [RAN].**

1. The commits:
- bf3fb0898e: `structure_program(model, loss_fn, optimizer, example_batch, *, cc, mode="reference")`, `step_report(..., cc=cc, structure=True)` and `training_rules(cc=cc, structure=True)`, rule set `training-cc12.0-structure`. The mark is `annotations["train_step"]["structure"] = True`, and no other Program has the key. `training_attention.STAND_IN = 16` is the block size.
- c14944ed8a: conftest. `STRUCTURE = (3, 33)`. At a cc, `_derive(..., structure=True)` exports with `gpu=True`, outside deterministic algorithms and without the stand-in tables.

2. What I ran, on my VM at c14944ed8a:
- The structure at (3, 33) at the cc derives in about 2 s. It holds `Fa2TrainHead_v1{T=32,D=32,BN=16}` x6 twice and `Fa2TrainBwdHead_v1{T=32,D=32,BM=16,BN=16,NS=1}` x6 twice, and its linears are `LinearRef_v1` [RAN].
- `write_handover`, then `load_handover`, then `_same_step(_Program(step), _Program(structure))` passes at 519 occurrences each [RAN].
- The adapter's new tests pass [RAN, `pytest tests/program/test_train_step.py -k "structure or training_rules_build or tiny_bf16_step"`, 9 passed]:
  - `test_tiny_bf16_steps_structure_on_sm120`: the structure's occurrences equal the step's at the cc, node, op and inputs.
  - `test_a_structure_the_stand_in_heads_dont_bind_at_is_refused`: at T = 4 the hint starts "the structure's stand-in head".
- The grader's tests by name, with your `grader.py` unchanged [RAN, `pytest tests/test_grader.py -k "handover_reads_back or products_are_the_steps or gpu_steps_linears or structure_of_another or against_a_trace or circuit_size_on_the_tiny"`]: 7 passed, 1 failed.
  - The failure is `test_the_products_are_the_steps_matrix_products[bf16-cublas]`: `{'forward': 15} != {'forward': 19}`, `{'backward': 30} != {'backward': 38}`. Those are the four forward and eight backward bmm products, now inside the fused ops: your seven rows.
  - `test_the_gpu_steps_linears_are_its_tensor_cores_dots` now passes.

3. Not run by me: the grader's whole suite, and `test_slowdown_code.py` with `VERITY_GRADER_SLOWDOWN_CODE`. Both are in my node run `r20261112-194903-bdcc`, queued.

