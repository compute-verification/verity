[2026-11-12T21:00Z] 034ea27e0030
PRICE, Thu 12 Nov 1:00 PM PST. **The node runs back the review and the verdict stands at NOT READY with 3 must-fixes: the suites pass at the head merged with main, and each mutation (marker removed, marker widened by one op) fails the PR's tests.** The note (note:training-20261112T2007Z-report-price-9374-review) now has these lines in an addendum.

SUITES
- [RAN r20261112-193916-8cde, vy-nebius-cpu-1] Suites at the head merged with main (`b772fdb1e0`), with `VERITY_GRADER_SLOWDOWN_CODE` set and nothing cached:
  - `benchmarks/training_slowdown`: 587 passed.
  - `benchmarks/training_slowdown/grader`: 31 passed.
  - Five tests run over 5 s without the `slow` mark. None of them is new in this PR; the slowest is test_library at 8.3 s.
- [READ the author's r20261112-183341-bffa, rc 0, at the unmerged head 70d32d57ee] It agrees:
  - repository 349, grader 31, training_slowdown 587 and verity-graph 241, all passed;
  - `pins --stale`, ruff and the ratchet gate are clean.

MUTATION: MARKER REMOVED (`_marked_attention()` taken out of `_trace`)
- [RAN r20261112-203708-4565, vy-nebius-cpu-3] 13 tests fail in training_slowdown (94 pass):
  - `test_a_bf16_calls_ops_are_priced_at_16_bits_and_every_other_op_as_traced`: all six cases (plain, dropout, mask, module, no grad, wide).
  - `test_a_float32_calls_ops_stay_at_32_bits`: plain, dropout and mask.
  - `test_library`'s `test_a_row_says_what_share...`: priced-in reads float32 where bfloat16 is expected.
  - `test_run`: `test_the_job_traces_its_routes_step`, and `test_a_jobs_count_and_its_trace_agree_in_gates_at_attention` for whole and lora.
- The grader fails 2 of 6: `test_the_tiny_steps_beside_the_slowdown_code[bf16]` and `[bf16-cublas]`.

MUTATION: MARKER WIDENED (the forward marker held through the next matrix product after the call, one op outside it)
- [RAN r20261112-203735-3c8a, vy-nebius-cpu-12] 12 tests fail in training_slowdown (95 pass):
  - the same six bf16 cases and three float32 cases. Every one fails on its row set: an `mm` outside the call now carries `attention`.
  - `test_the_job_traces_its_routes_step`.
  - the count-against-trace test for whole and lora. The trace's gates at the call's nodes exceed the count's by exactly the out-projection.
- `test_library` passes, and so do the grader's 6.
  - [INFERRED] Marking a 16-bit product outside the call at 16 bits doesn't change a price. So the op-for-op row tests are what hold the marker's extent, and they do.

LIBRARY TABLE
- [RAN, failed] My two library runs, r20261112-194321-5b3c at main and r20261112-194349-fc88 at the merged head, died on `ModuleNotFoundError: numpy` because I invoked them wrongly (`uv run` from the member directory). I didn't rerun them.
- [READ the author's notes `lanes/training/trn138-job4-library-before-after-43.txt` and `-xl.txt`] I read the author's tables instead:
  - 44 entries, 36 with a figure.
  - |C| falls by a factor of 1.6932 (xl) to 2.528 (s lora), and the slowdown rises by the same factor on every row.
  - The binding term is the same on every row, and priced-in goes to bfloat16 100%.
  - This matches the body's "1.69 to 2.53". The 1.69 to 2.32 range is the 32 kept traces (note training-20261112T1923Z-report-grade-trn138-job4-full). SF4 stands: the body should say which set each range covers and cite the runs.

DONE AND LEFT
- Done: the review, both messages, and the node runs. The handoff note is note:training-20261112T2100Z-handoff-PRICE.
- Left, in order:
  1. Your call to mark #9374 or send it back.
  2. If you send it back, the author fixes MF1 to MF3, best by landing #9039 first.
- In flight: nothing of mine. The mutation branches are local and unpushed.

