[2026-11-14T07:43Z] 488661a3be02
From @training, Fri 13 Nov, 11:45 PM PST. STEP: your three points on #9863 at 132ad116506a, answered; nothing here stops your mark.

- **A 3-D call as the 4-D call on a batch of one: yes, keep it.** It is what torch does with it, and a refusal would turn away a step the card runs. The reader checks your reading of `normalize_unbatched_input` against the source.
- **A product's result rounded to the head's return after one elementwise f32 step: yes, that is what I meant.** `ret` names the wires each MAC feeds; the accumulator alone was not the condition. Keep the test that holds each `ret` to its return's leaves.
- **The backend switches: yes to your recommendation, as a change of its own after this one is marked**, not in #9863: read the switches before entering `sdpa_kernel(MATH)` and refuse a step with any of them off, naming the switch, with a test per switch. Until it lands, #9863's body says in one sentence that a step which turns a fused kernel off is traced as if it were on. Open that change yourself when #9863 is marked; `Refs TRN-164`.
- Your StepLayout answer stands: no second copy.
- READSTEP holds the delta and sends its verdict to you and to me.

[2026-11-14T07:45Z] ed0c4b63972e
READSTEP, Fri 13 Nov, 11:44 PM PST. Re-read of #9863, `dbf5e36..132ad116506`.

**The must-fix is fixed the way the coordinator preferred, the ruling is carried with each `Product` and pinned by tests, and the body is true at the head but for one sentence about dV.**

The verdict is for head 132ad116506a3e9156774e84ae57feb912bc199a, the branch's head when I sent this (RAN, `git fetch` and `gh pr view`).

## The delta

Two commits, 5d8fe5aa010 and 132ad116506, on dbf5e36 (RAN, `git log`).
- There is no merge commit and no merge of `main` (RAN, `git log --merges`).
- Five files changed, each part of the two fixes: `train_step.py`, `test_train_sdpa.py`, `grader.py`, `test_grader.py`, `test_slowdown_code.py` (RAN, `git diff --stat`).

## Check 1: `_CardSdpa`

It takes the coordinator's preferred shape (READ, `train_step.py:199-235`):
- A 4-D call that flash doesn't take, with a mask or in f32, becomes `aten._scaled_dot_product_efficient_attention` when `_efficient_takes` says that kernel takes it. No rule lowers that op, so the step is `Unsupported` there.
- When `_efficient_takes` says no, `export_step` raises `ValueError` naming the call (`:216-220`).
- Only a call no fused kernel takes stays on the math path (f64, or q, k and v of mixed rank or dtype).

`_efficient_takes` restates torch's selection; it doesn't call it. torch's own predicate needs a CUDA device, so it can't be read on fake tensors (INFERRED). I diffed the restatement against torch's source at the tag v2.14.0 (RAN, fetched).
- Backend order: `Context.h:483-487` is `flash_attention, efficient_attention, math, cudnn_attention`. `check_prefer_cudnn_attention` (`sdp_utils.cpp:94-118`) puts cuDNN first only when `major == 9 || major == 10`, so on sm_120 a call flash refuses reaches the memory-efficient kernel next. The docstring at `:183-186` says the same.
- `can_use_mem_efficient_attention` (`sdp_utils.cpp:1098-1180`) agrees with `_efficient_takes` line for line:
  - the head dims: `query_size_last == params.key.sym_size(-1) && query_size_last % alignment == 0 ...`
  - `minimum_gemm_alignment` (`:147-160`): 8 for f16 and bf16, 4 for f32 on sm_80 and up
  - `storage_offset` alignment, last-dim stride 1 (the mask's too), nonzero lengths, batch and heads equal or MQA-broadcast
  - dtypes `{kHalf, kFloat, kBFloat16}` from sm_80, and hardware `[sm50, sm121]`
- The composite's efficient branch (`attention.cpp:820, 846-855`) matches `_fused`: `convert_boolean_attn_mask`, `preprocess_mask` (`:624-638`, pad to an 8-aligned stride, then expand), and `should_compute_logsumexp_with_attn_mask` for the LSE flag.
- `normalize_unbatched_input` (`sdp_utils_cpp.h:49-61`) unsqueezes a 3-D call before selection, which the new 3-D branch mirrors.
- The installed wheel's `torch.version.git_version` is 08187d9e0fba, the commit the body names. Its three files are byte-identical to the tag's (RAN).

Each case has a test, and each passes at the head:
- A masked call, an f32 call, and an f32 masked call: `test_gpu_export_refuses_a_call_flash_does_not_take` (`test_train_sdpa.py:235`).
- f32 at head dim 30 raises, f64 stays on the math path, and a 3-D call is flash: `test_gpu_export_takes_no_kernel_it_cannot_name` (`:245`).

Neither commit message names a mutation, so I ran my own, each applied alone and undone (RAN):
- M1, the mask dropped from the flash test at `:216`: fails `test_gpu_export_refuses_a_call_flash_does_not_take[a mask-kw0]`. My first report's SHOULD FIX 1 is closed.
- M7, `_efficient_takes` always True: fails `test_gpu_export_takes_no_kernel_it_cannot_name`.
- M8, f32 handed back to the math path, as at dbf5e36: fails 4 tests (`takes_the_cards_attention_op`, both f32 cases, `takes_no_kernel_it_cannot_name`).

## Check 2: the ruling

The width is carried with each `Product`, not recomputed.
- `Product.ret` (`grader.py:108`) is the index of the head's return; `None` for the scores and dP.
- `Cells` reads the width off the head Definition's own return type: `out = dot_ret if pr.ret is None else ret.fields[pr.ret]`, `"out_width": max(out.width_set())` (`:587-589`) (READ).

The head's circuit holds all four at bf16 (READ):
- `fa2_train.py:182-184`: `_head_type` returns `Tuple(rows, Array(S.T, F32))` with `rows = Array(S.T, Array(S.D, BF16))`, so O is index 0. The row ends `tuple_of(B.batch(P.F2fpBf16, o, axes=(0,)), lse)`.
- `fa2_train_bwd.py:235-238`: `Tuple(rows, rows, rows)` with `_rows(..., t=BF16)`, so dQ, dK and dV are indices 0, 1 and 2, matching `ret=0/1/2`.
- dK and dV come from `Fa2TrainBwdDkDv` (`:209-215`): `tuple_of(F2fpBf16(scaled), F2fpBf16(acc[0]))`. dQ comes from `Fa2TrainBwdDq` (`:232`): `F2fpBf16(scaled)`.
- `test_the_attention_table_is_its_heads_census` (`test_grader.py:431-435`) holds each named return to `leaves == outputs` and widths `{16}` over all five `HEAD_SHAPES` (RAN, passes).

`test_slowdown_code.py:156-160` asserts `"O": scores` and `r["circuit"]["O"] == r["trace"]["O"] > 0` at P V, dV, dK and dQ. That is equality, not a tolerance (READ; RAN, passes with `VERITY_GRADER_SLOWDOWN_CODE` set, none skipped).

The count side and the trace side agree for the same reason, and the test reads it independently:
- The circuit side is the head's bf16 return.
- The test's expected width is the ATen op's own output dtype, `n.meta["val"][RESULTS[product]].dtype.itemsize * 8` (`test_grader.py:56, 158`).
- The trace prices the call at its query's bf16.
- All three are 16 because the op returns at the query's dtype (READ / INFERRED).

M5′, `ret` ignored so every output is at the 32-bit accumulator, fails `test_the_products_are_the_steps_matrix_products[bf16-cublas]` and `test_the_tiny_steps_beside_the_slowdown_code[bf16-cublas]` (RAN).

## Check 3: my first report's corrections

All are in, but for the dV sentence below (READ):
- The `_deterministic` docstring names the flash backward (`train_step.py:161-164`).
- The `_CardSdpa` docstring is now true.
- The body states that the opt-in test runs in no check, that the import-list test's failure is the thing gone, and that the grader's output changes form with no figure changed.

## On the VM at the head (RAN)

- `test_train_sdpa.py`: 30 passed, the progress line only.
- `test_grader.py` and `test_slowdown_code.py`, with `VERITY_GRADER_SLOWDOWN_CODE=<worktree>/benchmarks/training_slowdown@132ad116506a`: "39 passed in 53.67s", none skipped.
- ruff: the same two I001 findings in `train_step.py` and `test_train_sdpa.py` as at dbf5e36, nothing new; the root config reports "All checks passed!" on the grader files.

## Runs at the head

r20261114-073846-159b: state `waiting`, rc none, `tree_sha` 132ad116506a3e9156774e84ae57feb912bc199a (RAN, named keys). No run is done at this tree. The body says the run was in flight, which I don't count as a finding.

READY 132ad116506a3e9156774e84ae57feb912bc199a

## MUST FIX

None.

## SHOULD FIX

1. dV is rounded with no elementwise step, but two places say it is.
   - Where: `benchmarks/training_slowdown/grader/grader.py:124-125`, "each rounded to bf16 after an elementwise f32 step (the row's total, the scale D^-0.5)", and the body's matching bullet under "The grader".
   - Why it is false: `Fa2TrainBwdDkDv` gives `dV = bf16(dV)` (`fa2_train_bwd.py:213`, docstring; `:215`, `B.batch(P.F2fpBf16, acc[0], axes=(0,))`) (READ).
   - No count changes, since dV is still a bf16 return.
   - Fix: "P V, dQ and dK each rounded to bf16 after an elementwise f32 step (the row's total, the split combine, the scale D^-0.5), dV straight from its accumulator". Leave the commit message as it is (no rebase).

## NOTE

1. The 3-D branch is new behavior beyond the must-fix's letter: a 3-D call was math-path at dbf5e36 and is now flash. It is torch's own `normalize_unbatched_input` (`sdp_utils_cpp.h:49-61`) and is tested (`test_train_sdpa.py:245`) (READ, RAN).
2. Grouped-query calls are refused where the card would take them. `_efficient_takes` returns False for one, but torch 2.14's efficient kernel takes grouped-query calls (`check_batch_size_and_num_heads_dense<supports_gqa=true, true, supports_mqa=true>`, `sdp_utils.cpp:1155`). Such a call is refused at export rather than becoming the efficient op. Both are refusals, and the docstring says it (READ).
3. The body's list of flash checks `_CardSdpa` doesn't mirror leaves out one: sm_120's training limit, head dim in (192, 224] with grad (`check_requires_grad_and_head_dim_gt192_constraints_on_sm86_89_or_120`, `sdp_utils.cpp:530-549`). It is harmless, because the flash rule lowers only `TRAIN_SDPA`'s shapes, all at head dim 64 (`fa2_train.py:53`) (READ). One word in the body, if wanted.

Nothing addressed to @training reached my conversation.

