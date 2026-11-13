[2026-11-13T16:43Z] 503779d90972
QUERY, Fri 13 Nov, 8:43 AM PST.

**#9374's head is `d8e384ed2d4b`, merged with `main` `4ae17f7eafd7`. The suites, ruff and the ratchet gate pass there; the library and mutations pass at `3da42893fd9b`, which differs only by your docstring line. The README is 46,316 bytes, and #9374 is still a draft.**

Head `d8e384ed2d4b605fd07d5045d9d2332882decab4`, pushed with no force (RAN). The commits after `7f3b0b2cbd55` (RAN):
- `826020f84ac`, the merge, by hand, with no restack;
- `1810585569d`, the seventh product;
- `3da42893fd9`, the README;
- `d8e384ed2d4`, your line in `priced_dtype`'s docstring.

The body is updated (783 words; Lane, Refs and Goal kept) (RAN).

Step 1, whose text stands (READ: the conflicts; RAN: the suites):
- `__init__.py`: mine, 4.26.0. Main's side was #9608's 4.25.0.
- `tests/test_grade.py`: both. Attention rows are float32 in the trace and in the CPU job's log (#9608), and priced at the call's bf16. Priced-as-named is asserted on the rows outside attention, M's. The docstring now says the grade divides the products of M.
- `tests/test_run.py`: my docstring, with no exception. It names the GPU job (fused, bf16, equal to the circuit) and the CPU job (math, float32, above it).
- `README.md`: mine, plus #9608's CPU sentence.
- The meeting without a textual conflict is #9608 in `run.py`. A CPU job runs the math kernel, so its count names attention float32 while my trace prices it bf16. Only `test_grade`'s named-dtypes test met it (resolved as above). `grade.priced_apart` reads M only, so no grade refuses (READ: grade.py). `flops.py`'s `CPU_FUSED_ATTENTION` doesn't touch the capture, which already holds MATH (READ). #9686 is the QAT route's dials and entries and meets nothing of mine (READ: git diff).

Step 2: the retired-terms and doc-path tests name none of my lines (RAN locally, -k free, 27 passed; `repository` passes in both suites runs). My added lines name only paths inside the package (READ: git diff).

Step 3: `ATTENTION_PREMISE` reads "each of the six matrix products the count holds of a 16-bit attention call is…". `PARTS["premise"]` ends: "The card's backward computes the scores once more, a product this count leaves out, so each slowdown reads high by that product's share of the count". test_library asserts both (RAN).

Step 4: README 50,523 → 46,316 bytes (RAN: wc -c). What went: the module table's `routes/`, `gates.py` and `as_math.py` rows, cut to pointers; a continued job's state hashing and its `output` limit; the state's disk sizing; a window id's definition; the op log's dtype cases; fused optimizers' counts; the premise's edges. The two sections you named are untouched.

Step 6, the runs and their last lines (RAN):
- `r20261113-162630-fe74`, at `d8e384ed2d4b`: suites `--changed 4ae17f7eafd7 --fresh` with the grader variable set. Grader 31, repository 361, training_slowdown 696 and verity-graph 246 passed, none cached; ruff clean; ratchet `4ae17f7eafd7..d8e384ed2d4b: 1 change(s), 0 failed`. Last line: `suites rc 0 ruff rc 0 ratchet rc 0`.
- `r20261113-155646-98e2`, the same at `3da42893fd9b`, with the same counts. Last line: `suites rc 0 ruff rc 0 ratchet rc 0`.
- `r20261113-155712-0f75`, unmarked: 22 failed plus the grader's 2, the same names as `7f3b0b2cbd55`'s `-a4e1`. Last line: `mutation unmarked: training_slowdown rc 1 grader rc 1`.
- `r20261113-155746-2621`, widened: 13 failed, the same names as `-4175`. Last line: `mutation widened: training_slowdown rc 1 grader rc 0`.
- Both mutations pass 2 more tests than before: #9608's two new tests (READ: git diff). I reran the mutations because step 3 touched `capture/` (the premise string).
- `r20261113-155808-3e25`, the library on 59 entries: all exit 0, and 37 are traced. Last line: `lib exit 1`, which counts the 22 that need a record.
- `r20261113-155819-eb96`, xl. Last line: `lib exit 0`. `--on auto` placed it on vy-nebius-cpu-2.
- Every traced entry, xl included, has a `library.json` equal to `7f3b0b2cbd55`'s, even in `step.capture*`. The table's `parts.premise` differs by the step 3 clause. None refused, every call marked bfloat16, and no attention row in M.
- Moved rows, all untraced: QAT `top2`, `top3` and `top4` gain the dials `grad_code` and `low_state`, from `bc595972916` (#9686).
- New rows from #9686 (13 QAT entries), each needing a record: `s-t20-1.1x-qat-top3-3000`, `-top3-grad4`, `-top4-keep10`, `-top4-keep15`, `-top4-keep15-2380`, `-top4-keep15-grad4-lowstate`, `-top4-keep15-grad8-lowstate`, `-top4-keep30`, `-top4-keep30-grad8-lowstate`, `-top4-keep50`, `-top4-keep50-7080`, `-top4-keep50-grad4-lowstate` and `-top4-keep7.5-decay60`.

`main` is now `d119240998be`, with nothing under `benchmarks/training_slowdown` since `4ae17f7eafd7` (READ: git diff). Handoff: `20261113T1642Z-handoff-QUERY` (synced).

