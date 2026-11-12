[2026-11-12T21:59Z] 5e995e291ee0
LOG, Thu 12 Nov 1:59 PM PST. **MF1 and SF1–SF3 are committed on #9039's branch, each in one commit, and main is merged once. Not pushed yet: the suites at that head and the 44 pairs again (for SF3's move) are on nodes. Two of my review's statements were wrong; corrected below.**

Branch `cursor/training-oplog-dtype-f980`, local head `ca6c4c2a54`:
- `34cfa7ec0b` SF2: `gates.row_dtype`, the one reader, at every place in my list. Tests: `test_gates.py::test_a_row_of_neither_kind_is_refused_and_never_read_as_the_other` and `test_grade.py::test_every_reader_of_an_op_log_refuses_a_row_of_neither_kind` (8 readers, eight-field row and empty dtype each).
- `84682f825b` SF1: `run.tf32` states `cuda.matmul.fp32_precision` where torch has it, and the legacy pair otherwise.
  - RAN on my VM, torch 2.14: that one reading reads the switch however it was set (new API, global API, `allow_tf32`, `set_float32_matmul_precision`).
  - Tests: the rewritten `test_run.py` TF32 test, and a fallback test for a torch without it.
- `c84bf35f2d` MF1: `TF32_FIX` names the switch off alone. The grade docstring and README say tf32 naming holds only outside M.
  - The old test is rewritten to grade traced jobs: a Shampoo job, whose 8 float32 GEMMs are outside M, grades named tf32 with ρ, and C drops by exactly f32→tf32 × those MACs.
  - New: `test_a_product_of_m_named_tf32_is_priced_apart_from_its_trace` (the `adapted` LoRA job: ρ and S refused).
- `2bca248b2c` SF3: `flops.in_place` ops are named `torch.result_type` over their tensor operands, in the counter and in the trace's `_computed_in`. Test: `test_flops.py::test_an_in_place_op_over_two_dtypes_is_named_in_the_one_it_computes_in` (mul_, addcmul_, lerp_, `_foreach_mul_`, and a make_fx trace of each).
- `ca6c4c2a54`: main `5f0c008aa040` merged, clean.
- Neither SF2 nor SF3 was bigger than a commit.

Corrections to my review:
1. **lerp_**: my probe cast its operand to bf16, so "bf16.lerp_(f32) named bf16" was never run. RAN now: torch refuses `lerp_` over two dtypes ("expected dtype BFloat16 for `end`"); only a 0-dim weight may differ, and it doesn't widen. So the test holds `lerp_` with a float32 0-dim weight named bfloat16, which is what torch computes. mul_ and addcmul_ stand as reported.
2. **SF5**: the author's handoff (`20261112T2034Z-handoff-…`, line 91) already says #9039 lands first. The stale "#9374 lands first" is in their progress note `20261112T1904Z-progress-trn126-priced-apart-refusal` (line 42). I put the dated line at that note's top (21:40Z) and left the handoff alone. My report note says so too.

Runs that ended:
- `r20261112-205313-d45b` (head 5c793f5e + main 5f0c008a, cpu-1): grader 31 passed, training_slowdown 613 passed, slowdown-code 6 passed. Last line: "merge rc 0 suites rc 0 grader rc 0".
- `r20261112-205354-0b93` (cpu-14): baseline 266 passed in 305 s. Each mutation is caught. Last line: "mutations done".
  - M1 (writer drops dtype): 22 failed, among them `test_flops::test_an_ops_row_names_the_dtype_it_computed_in`, `test_resume::test_two_segments_…` and both priced-apart tests.
  - M2 (unpriced priced as bf16): 12 failed, `test_grade::test_a_record_with_an_op_in_a_dtype_no_definition_prices_…` among them.
  - M3 (one_pricing off): 1 failed, `test_grade::test_a_record_that_names_a_dtype_in_some_rows_and_none_in_others_is_refused`.
  - M4 (priced_apart off): 1 failed, `test_grade::test_a_trace_that_prices_a_product_of_m_…`.
- `r20261112-205422-0613` (cpu-14): every mutation is caught. Last line: "mutations done".
  - M5 (TF32 check off): 4 failed, the TF32 test's 3 params and `test_run::test_a_record_states_torchs_tf32_switch…`.
  - M6 (op_key drops dtype): 15 failed.
  - M7 (dtype_of uses first operand): 2 failed in `test_flops`.
  - M8 (trace prices first operand): 7 failed, `test_run::test_a_jobs_trace_and_its_counter_agree…`.
- Suite time at the old head (0b93's durations): the `adapted` fixture's setup is 9.3 s; the PR's other new tests are each under 1 s.

Shampoo's p (RAN on my VM, the old pairs `r20261112-172815-142d` and `-172826-ea0d`, graded by this branch's code):
- No graded p exists for either entry: neither has a run of record, and neither config names an honest config. I give the p a grade would read at three counted steps each, against `s-t20-1.1x-ref-optim`.
- `s-t20-1.1x-shampoo`: 1.06799 before (assumed dtypes), 1.19277 after (named), ×1.11684.
- `s-t20-1.1x-shampoo-whole`: 1.21798 before, 1.34547 after, ×1.10467.
- The factor is the entry's |C| named/assumed over the reference's (1.12141/1.00410 and 1.10920/1.00410).
- If SF3 moves either, I'll resend from the new pairs.

In flight:
- `r20261112-215150-bce5`: suites `--changed 5f0c008a --fresh`, grader slowdown-code and ruff, at `ca6c4c2a54`, cpu-12, running.
- Pairs at `ca6c4c2a54`, the author's scripts unchanged, all waiting for a slot: `r20261112-215241-9af3` (xl), `-215312-0450` and `-215327-c868` (l16), `-215357-bd48` and `-215426-30ef` (m-t20), `-215438-c32a`, `-215452-7c5a`, `-215507-4885` and `-215534-fe4c` (s-t20).
- The xl one took 2.9 h last time. I'll push when bce5 passes, set the body when the pairs end, and send QUERY the SHA.

