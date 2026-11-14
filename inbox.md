[2026-11-14T06:34Z] 519e24cb1a07
TRN-164 READSTEP, Fri 13 Nov, 10:34 PM PST.

**#9863 is NOT READY with one must-fix: at a cc, an attention call with a mask, or one in f32, silently takes torch's math path beside the flash circuit in the same step. The factor of 2 is true, and v2 regrades no stored record.**

Report note: training-20261114T0631Z-report-trn164-first-read-9863. Handoff note: training-20261114T063356Z-handoff-READSTEP. Runs launched: none. The whole report follows.


# First read of #9863

dbf5e361776219b013a62c27d26baa2cb0b5ac08 NOT READY, 1 must-fix.

**The circuit is right where it fires, and the factor of 2 is a true statement about the two pricings. But a step at a cc
silently mixes the card's flash circuit with torch's math path for an attention call with a mask, or one in f32.**

## What I read whole, sampled, and did not open

- Read whole: the diff of all ten files (`git diff origin/main...dbf5e36`), the PR body, and STEP's handoff
  (`training-20261114T010107Z-handoff-STEP`).
- Sampled, in the code the change leans on: `rules/training_matmul.py`'s `Matmul` (its docstring, `applies`, the start of
  `emit`); `training_slowdown/capture/products.py` (`ATTENTION_PREMISE`, `priced_dtype`); `training_slowdown/opnames.py`;
  `training_slowdown/work.py`'s `HE_BITS`; `grader.py`'s `against`; `grader/settings.py` and `grader/pyproject.toml`;
  `tools/verity/check/check.py`'s `OPT_IN`; #9840's file list and body.
- Not opened: the Definitions `fa2_train` and `fa2_train_bwd` (on `main`, BACKWARD's), `derive_graph`'s internals, the
  slow tests' bodies (I didn't run the slow tier; the cancelled run's `verity-vllm` pass line covers it), and the
  `training_slowdown` README beyond the grader's section.

## MUST FIX

1. A masked or f32 attention call at a cc is silently torch's math path, beside the flash circuit in the same step.
   - Where: `adapters/vllm/verity_vllm/program/frontend/train_step.py:183-184` (`_CardSdpa.__torch_function__`).
   - What happens: a 4-D call with an `attn_mask`, or one whose q, k and v aren't f16 or bf16, is handed back to the trace, which
     decomposes it to the math path. At a cc without TF32, `Matmul` lowers the math path's f32 `bmm`s as `LinearRef_v1`, so the
     step derives.
   - RAN (`/tmp/mixed.py`, on the VM): one step at cc 12.0, under deterministic algorithms, at a stand-in shape.
     - Unmasked call only: `Fa2TrainHead_v1` and `Fa2TrainBwdHead_v1`.
     - Unmasked call, then a masked bf16 call: both heads, plus 12 `LinearRef_v1` f32 and 1 `SoftmaxRef_v1` f32. Nothing
       refused and nothing recorded.
     - Unmasked call, then an f32 call: the same mix.
   - On a CUDA device those two calls go to another fused kernel (memory-efficient or cuDNN attention), never the math
     path, since flash takes no mask and no f32 (INFERRED from torch's backend selection). So the body's "dispatches
     attention as the card does" is false for them, and so is `_CardSdpa`'s "as torch's composite dispatches it on a CUDA
     device".
   - Fix: at `gpu=True`, send a 4-D call that flash doesn't take to the op of the kernel the card takes
     (`aten._scaled_dot_product_efficient_attention`), which no rule lowers, so the step is `Unsupported` naming it. Or make
     `export_step` raise naming the call. Only calls that no fused kernel takes stay on the math path. Add a test: a masked
     call at `gpu=True` is refused. It also kills mutation M1 below.

## SHOULD FIX

1. No test sends a masked call through `export_step(gpu=True)`.
   - Mutation M1 (drop the `attn_mask` condition at `train_step.py:183`) leaves every test passing: `test_train_sdpa.py`,
     and `test_train_step.py -k "sm120 or gpu_export or structure"` (RAN).
   - Fix: the test named in MUST FIX 1.
2. The hand-settled factor-of-2 assertion runs in no check.
   - Where: `grader/tests/test_slowdown_code.py::test_the_tiny_steps_beside_the_slowdown_code`, which skips unless
     `VERITY_GRADER_SLOWDOWN_CODE` is set (`:117`). `check.py`'s `OPT_IN` doesn't name it, so neither `check` nor the
     nightly sets it (READ). This is `main`'s state, not new.
   - The circuit's side, 32 bits, is also pinned outside it: M5 fails `test_grader.py::test_the_products_are_the_steps_matrix_products[bf16-cublas]`.
     The trace's side, 16 bits, is pinned only by the opt-in test.
   - Fix: name it in `OPT_IN`, which needs a value of DIR@COMMIT, so it is a check change and infra's. Or the body says it
     runs only by hand.

## NOTE

1. Where `deterministic` is recorded (RAN, `/tmp/det.py`).
   - It reaches the Program only as the rule set's name, `annotations.derive.ruleset.name` = `training-cc12.0-deterministic`,
     and in each backward call's record (`{"deterministic": True}`, `training_attention.py`, `FlashSdpaBackward.emit`).
   - `StepLayout.to_json` (`train_step.py:135`), the `train_step` annotation, leaves out `gpu`, `tf32` and `deterministic`.
2. Warn-only (`train_step.py:160`).
   - `_deterministic()` takes warn-only as not deterministic. torch's documented rule is the opposite: RAN,
     `torch.are_deterministic_algorithms_enabled()` is True under `warn_only=True`.
   - torch's flash backward, as I recall its CUDA source, is the exception: it runs non-deterministically under warn-only, and
     warns. So the effect is right for the one kernel that reads it (INFERRED; there is no CUDA build here).
   - The docstring "Whether torch's kernels run their deterministic algorithms" should name the flash backward.
3. When the setting is read.
   - `_CardSdpa` reads it at the forward call, and the backward kernel reads it when it runs.
   - A step that changes it between its forward and its backward isn't caught; one that changes it between two forward calls
     is (`train_step.py:276`, tested).
4. #9840 moves this change's imports.
   - #9840 (top, open, not a draft) renames `verity/definitions/registry/fa2_train.py`, `fa2_train_bwd.py`, `cublas_bf16.py` and
     the registry's `__init__.py` into `adapters/vllm/verity_vllm/definitions/registry/` (RAN, `git diff --name-status` of its
     branch).
   - This change adds `fa2_train_bwd` imports to five files. Neither side shows a text conflict.
   - Whichever lands second is crossed with `tools/move/restack.py` on #9840's map (`Layout-move: tools/move/definitions_served_map.toml`).
5. Circuit-check.
   - The body gives the run's last line and the report's path in the run, not the report.
   - The 34 targets are `main`'s sidecars of `fa2_train`, `fa2_train_bwd` and their Boolean versions, at small shapes (READ,
     the run's `out/circuit-check.json`): 0 failures.
   - Neither the heads at the captured shapes the rules emit (`TRAIN_SDPA`, `TRAIN_SDPA_BWD`) nor the stand-in heads are
     targets.
   - The change adds no Definition, template or Boolean lowering, so AGENTS.md's rule arguably doesn't bind. The body
     should say that or paste the summary.
6. Body wording: "The import-list test ... is gone on `main`" reads as if the test were gone. It is the failure that's gone:
   the file passes, 3 passed.
7. Body: "Without a cc nothing changes" holds for the frontend.
   - The grader's output changes form at every step: `grader-products/v2`, each row's `product`, `attention`, `totals.fused`,
     `gates_by_op.attention` = 0, and the printed line.
   - Its figures don't change (NOTE 8).
8. What a v1-era step gets from this head's grader (RAN, `/tmp/v1v2.py`).
   - Two no-cc handovers (f32 and bf16) were each graded by `main`'s grader and by this head's.
   - The rows are identical but the new `product: None`. N, O, |C|, `gates`, `gates_by_phase`, the widths and `C` are the same.
     Only `fused` and `gates_by_op.attention` = 0 are new.
   - A cc handover of `main`'s, whose attention is math-path `bmm`s, takes the same code path, since it has no head cells
     (INFERRED).
9. Item 2 of the brief: the backward's inputs are tied, not free.
   - `FlashSdpaBackward` reads (dO, q, k, v, O, LSE): `ctx.coll` of the graph's own nodes, where O and the LSE are the
     forward's getitems 0 and 1, which `FlashSdpa` defined as its heads' outputs.
   - It reads the philox seed and offset (getitems 6 and 7) only as arguments, unread without dropout.
   - RAN: `test_flash_backward_wires_each_heads_dq_dk_dv` passes. Its Program's only inputs are q, k, v and g, and it
     evaluates to the backward head applied to the forward head's O and LSE.

## For the lead to read

1. **The factor of 2 (`test_slowdown_code.py:155-160`) is true, not bent.**
   - Circuit side: `grader.py:583` gives each fused product T·D outputs a head at P V, dV, dK and dQ, each at
     `max(dot_ret.width_set())` = 32. RAN: `HopperBF16WgmmaDot16`'s acc and return are 32 bits, its operands 16.
   - Trace side: `grader.py:829-831` gives `outputs × b_y`, with `b_y = HE_BITS[cell_gates][1]` = 16 for
     `HopperBF16WgmmaDot16_v2` (RAN, `work.HE_BITS`). Capture 4.26.0 prices a 16-bit call's products at the query's dtype
     (READ, `capture/products.py` `priced_dtype`).
   - The output counts are equal at those four products (T·D, and D·T for dK, as the math path's `bmm(qᵀ, dS)` has it), so the
     ratio is exactly 32/16. At the scores, the counts differ: the causal half against the whole rectangle.
   - What it decides: the grader's O counts a fused product at its f32 accumulator. The linears' GEMMs at a cc count at their
     route's bf16 output (`out_widths` [16, 32]), and P V's result leaves the head as bf16. Whether O takes the accumulator
     for fused products is your call; the test pins it either way.
2. `train_step.py:177-189` (`_CardSdpa`) decides which attention calls are the card's flash kernel: no mask, 4-D, f16 or
   bf16. It is MUST FIX 1.
3. `training_attention.py:194` refuses every backward without deterministic algorithms, so the `training_slowdown` GPU job's
   steps, which set none, are refused at their flash backward. The body says so.
4. `grader.py:123-131` (`ATTENTION`) decides the seven products' MACs and outputs (visible against visited pairs). Its tests
   hold it to the heads' census.
5. `grader.py:818` and `totals` decide that attention's gates are the heads' whole, never split by product: `None` at each
   `against` key.
6. `grader.py:65` (`READ` = `grader-products/v2`).
   - Only `grader.read` writes the name, into OUT.json's `format`. Only the grader's own tests read it (READ, `git grep` over
     `origin/main` and the head). The grader reads handovers (`grader-handover/v1`, unchanged), never its own output.
   - No note names a stored grader output (READ, grep of the notes). The store's index isn't reachable from my VM, so the
     store itself is unchecked.
   - On any step v1 could read, v2's figures are v1's (NOTE 8). A v1 grader refuses any op with "attention" in its name
     (`opnames.GEMM_LIKE`), so no v1 record holds a fused attention.

## The brief's items

1. When each rule fires.
   - Forward (`training_attention.py:84`): a flash op of bf16 q, k and v of one 4-D shape that `TRAIN_SDPA` lists, causal, no
     dropout, the default scale (as an f32), no debug mask, its state read only by the op's backward.
   - Backward (`:169`): dO, q, k, v and O bf16, the LSE f32, one shape that `TRAIN_SDPA_BWD` lists, dense, causal, no dropout,
     the default scale, `deterministic` set.
   - Each refuses every other call of its op with a reason; f16 reaches the flash op and is refused there.
   - A call `_CardSdpa` doesn't send to flash gets the math path, silently: MUST FIX 1.
   - `deterministic` turns the backward's refusal into a lowering and adds `-deterministic` to the rule set's name (NOTE 1).
   - Calls of one step that disagree on it are refused: `export_step` raises "changes torch.use_deterministic_algorithms"
     (`train_step.py:276`, tested).
2. The backward is tied to the forward: NOTE 9.
3. The grader: "For the lead to read" 6 and NOTE 8. No stored record grades differently.
4. The factor of 2 is true: "For the lead to read" 1.
5. Whose files.
   - `git log 56f6c267563..origin/main` on the three frontend files is empty. Of the ten files, only
     `benchmarks/training_slowdown/README.md` changed on `main` (`git diff 56f6c267563 origin/main`).
   - `git merge-tree --write-tree origin/main dbf5e36` (main at 8e018c9331f) is clean, rc 0, tree ffcc1b1f014. The merged
     README is 47,442 bytes, under its 49,152 cap.
   - Hunks in the vllm lane's files:
     - `verity_vllm/program/frontend/train_step.py`: 21 hunks, at new lines 5, 12, 18, 36, 46, 124, 133, 160, 215, 224, 231,
       272, 276, 281, 301, 315, 320, 326, 329, 373, 422.
     - `rules/training_attention.py`: 16 hunks, at 1, 9, 11, 16, 20, 31, 58, 62, 65, 73, 95, 105, 110, 117, 124, 214.
     - `rules/training.py`: 8 hunks, at 24, 28, 50, 581, 594, 598, 607, 615.
     - `tests/program/test_train_sdpa.py`: 12 hunks, at 1, 11, 15, 20, 22, 37, 49, 61, 73, 118, 258, 263.
     - `tests/program/test_train_step.py`: 12 hunks, at 15, 37, 60, 800, 809, 818, 892, 897, 901, 924, 928, 933.
   - I didn't find the vllm refactor's PR. No open PR among the 300 newest touches the three frontend files but #9863 and
     #9514; #9840 is NOTE 4.
6. STEP's evidence. Each run's `tree_sha` (`research status --json`) is dbf5e36, and each was fetched with `--all`. Each
   last line:
   - r20261113-213708-84e1: cancelled, rc 143, STAGE_TIMEOUT, so no summary line.
     - `out/suites.log` holds 21 `passed` lines for its 24 suites and no failed line. The missing three are exactly
       `verity-numerical`, circuit-check's suite and `verity-flock` (RAN, a set difference over its first line).
     - The grader suite's line is "39 passed in 80 s", with `VERITY_GRADER_SLOWDOWN_CODE set at dbf5e3617762`, so the opt-in
       test ran there.
   - r20261114-030744-b391: "1 suites: 1 passed (0 cached), 0 failed; 0.8 min of test time" (`verity-numerical`).
   - r20261114-030744-242b: "1 suites: 1 passed (0 cached), 0 failed; 37.2 min of test time" (circuit-check's suite).
   - r20261114-030744-49da: "1 suites: 1 passed (0 cached), 0 failed; 48.9 min of test time" (`verity-flock`).
   - r20261113-214203-39ed: "RATCHET rc=0", after "passed: ruff finds nothing new in the 4 Python file(s) it changes ...".
     - Root ruff: "All checks passed!".
     - vllm ruff: "Found 3 errors." (I001), "RUFF vllm rc=1".
     - RAN: ruff 0.17.2 with `ruff_vllm.toml`, from one cwd, gives the same I001 findings in the same five files on `main`'s
       copies and on the head's: nothing new.
   - r20261113-213726-4c30: "circuit-check: 34 target(s), 0 failure(s), 36 unchecked (warnings, which never fail) on 28
     Definition(s)", then "CIRCUIT-CHECK rc=0".
   - Not checked: the Thursday run r20261112-194903-bdcc is at an older tree. The 84e1 `verity-vllm` pass covers those
     tests at the head.
7. Circuit-check: NOTE 5.
8. Mutations, each run on the VM against the named tests, then undone; the worktree is clean.
   - M1, the forward fed a masked call (the mask condition dropped from `_CardSdpa`): no test fails. SHOULD FIX 1.
   - M2, the backward rule claims no op: 7 fail (`test_the_ops_own_backward_may_read_its_state`, `test_flash_backward_wires_each_heads_dq_dk_dv`,
     2 of `test_flash_backward_refusals`, `test_tiny_bf16_step_on_sm120_not_under_deterministic_algorithms`,
     `test_tiny_bf16_step_on_sm120`, `test_tiny_bf16_steps_structure_on_sm120`). With no rule, the backward is
     `Unsupported`: there is no silent fallback.
   - M3, no deterministic refusal: 3 fail (`test_the_ops_own_backward_may_read_its_state`,
     `test_flash_backward_refusals[not under deterministic algorithms]`, `test_tiny_bf16_step_on_sm120_not_under_deterministic_algorithms`).
   - M4, any reader of the forward's state allowed: 2 fail (`test_flash_forward_refusals[the backward's state]`,
     `test_the_ops_own_backward_may_read_its_state`).
   - M5, a fused product's output at 16 bits: 2 fail (`test_the_products_are_the_steps_matrix_products[bf16-cublas]`,
     `test_the_tiny_steps_beside_the_slowdown_code[bf16-cublas]`).
   - M6, the structure mark's refusal removed: `test_a_structure_derivation_is_refused_as_the_step` fails.
9. The body.
   - It is true of the head but for "dispatches attention as the card does" (MUST FIX 1) and the two wordings in NOTE 6 and 7.
   - It keeps what was run apart from what was read.
   - It says the Definitions' match to the card is BACKWARD's capture and completeness, and that `_CardSdpa` stands in for
     torch's dispatch.

## Claims of the body, by how I know them

- The port table and the `-x` trailers: RAN, `git log` bodies. All eleven name their old commits as the table does.
- The four old merges carry no hand edit: RAN, `git show --cc` over the ten files is 0 lines for each.
- What the rules fire on and refuse: READ, the code; RAN, the fast tier of `test_train_sdpa.py` (21 passed) and M2 to M4.
- The grader's seven products and their census: READ, the code; RAN, the grader's cc tests, "10 passed, 29 deselected".
- `grader-products/v2`, and that no VERSION moved: READ (`git grep`); RAN, `/tmp/v1v2.py`.
- The suites, ruff, the ratchet and circuit-check: READ, the six runs.
- The README's size: RAN, `wc -c` (46,424 at the head; 47,442 merged).
- The Definitions match the card: not checked, and said so in the body.

## Runs I launched

None. Every claim of the change had a pass at this tree.

## On the VM (RAN)

- `pytest tests/program/test_train_sdpa.py -m "not slow"`: 21 passed (the progress line only; the suite prints no summary).
- `pytest tests/program/test_train_step.py -m "not slow" -k "sm120 or gpu_export or structure"`: 6 passed.
- From `grader/`, with `VERITY_GRADER_SLOWDOWN_CODE=<worktree>/benchmarks/training_slowdown@dbf5e36`:
  `pytest tests/test_grader.py tests/test_slowdown_code.py -k "cublas or structure_derivation or cards_attention or heads_census or every_head"`
  gives "10 passed, 29 deselected in 15.85s".
- `/tmp/mixed.py`, `/tmp/det.py` and `/tmp/v1v2.py`: scratch scripts, not in any tree.

