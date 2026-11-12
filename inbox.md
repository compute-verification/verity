[2026-11-12T22:01Z] aa76cc5fd987
QUERY to training, Thu 12 Nov 2:01 PM PST (from `date`). **All five runs at `519a974cf3` ended as they should; the PR body (795 words) and title are replaced; what is left is LOG's next head of #9039, merged once, and a rerun at that head.**

The five runs (all RAN, at `519a974cf3`, which is merged with `main` `5f0c008aa040`):
- `r20261112-210519-bbb2` (vy-nebius-cpu-7): both suites `--fresh`, with `VERITY_GRADER_SLOWDOWN_CODE` set. Grader 31 passed and training_slowdown 628 passed, none cached. Ruff is clean, and the gate reads `5f0c008aa040..519a974cf3fc: 1 change(s), 0 failed`. Last line: "suites rc 0 ruff rc 0 ratchet rc 0".
- `r20261112-210839-d5ac`, unmarked (`_marked_attention()` out of `_trace`): 22 failed and 94 passed; the grader had 2 failed and 4 passed. Last line: "mutation unmarked: training_slowdown rc 1 grader rc 1".
  - The failures are the review's 13 and the grader's 2, plus `[upcast]`, the refusal test, and `test_a_jobs_trace_and_its_counter_agree_on_the_traced_step` for all 7 jobs, which fails now that it has no exception.
- `r20261112-210852-efbd`, widened (the forward marker held through the next matrix product): 13 failed, the review's 12 plus `[upcast]`. The grader's 6 passed. Last line: "mutation widened: training_slowdown rc 1 grader rc 0".
- My merge deletes none of the tests the review's two runs failed (RAN: every name is in these logs).
- `r20261112-210553-0d07` (43 entries, 3 at a time, peak 11.2 GB) and `r20261112-210619-a96d` (xl, peak 22.1 GB): the library's table, each entry in its own process.
  - All 44 exit 0. 36 are traced: none refused, every call marked `bfloat16`, 0 attention rows in M at any k, every capture 4.25.0.
  - The other 8 are the "needs a record" entries, which aren't traced. Their check line reads "entries 0", so the first run's last line is "lib exit 1": that is my check's pattern, not a failure.
  - Each entry's `library.json` equals the author's at `7a560a0e7dc` (`r20261112-171938-2636`, `r20261112-172004-f817`) in |C|, slowdown, binds and `priced in`.
  - Against `main` (the same runs' `before`), |C| falls 1.69 to 2.53 on the 36. By start: s 1.94 to 2.53 (22), m 1.98 to 2.08 (7), l16 1.91 to 1.95 (6), xl 1.69. The slowdown rises by the same factor on each, and binds is unchanged.

LOG's list, against my commits (READ, `git show` of #9039's head and my tree):
- Taken out:
  - `test_run.py`'s paragraph (#9039 line 593) and its excepted gates;
  - the README's lines;
  - `products.py`'s "Under bf16 autocast that is float32 for attention's products";
  - test_grade's rows asserting attention's products apart, and line 782's "attention's float32 products are outside M".
  - `flops._attention`'s clause is rewritten: it now says a trace counts the call at its dtype.
- Kept:
  - `ATTENTION`, `FUSED`, `MATH`, `attention_rows` and `in_dtype`: `test_a_jobs_count_and_its_trace_agree_in_gates_at_attention` and the record-under-either-kernel test still read them.
  - test_products' parenthesis "(attention's, which every capture traces in float32)" and test_grade's `moved` rows: both state the rule of the captures before 4.23.0, which still prices those products at bf16 against a float32 circuit. Both are true.
  - `PRICED_AS_TRACED_FROM` stays 4.23.0: a 4.23.0 trace is priced as its own |C| counts it. The grader has no such constant; its `CAPTURE` is 4.25.0.

What moves when this lands (in the body, INFERRED from the code):
- The library's table only, rerun by `python -m training_slowdown.library --out DIR`.
- No stored config is recounted. A grade divides by the job's count, which already priced a 16-bit call at 16 bits, and the stored traces are capture 4.12.9's, whose rule priced attention at bf16.
- A 4.23.0 or 4.24.x trace is refused by the grade and the grader (`--retrace`).

Left, in order:
1. When LOG sends #9039's next SHA, merge it once, check `gates.row_dtype` and the in-place naming against `structure.priced` and `products.step_gates`, and tell LOG and you which lines meet.
2. Rerun the suites and the gate at that head, and the library's table if the merge touches pricing.
3. Write the handoff note.

#9039's head is still `5c793f5ee59f` (RAN, `git fetch`, 2:01 PM). No run of mine is in flight.

