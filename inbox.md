[2026-11-13T16:15Z] 383afadf549c
REPEATS, Fri 13 Nov, 8:14 AM PST. **The counter's `repeats` and tests 1 and 2 are pushed and pass; the reader side (M, the strings, the grade) and tests 3, 4 and 5 are pushed too; all four shipped routes record `repeats` on every step.**

Branch `cursor/count-repeated-positions-a461`, head `07eaeb5a9c9` (base `4ae17f7eafd`, not yet merged with main). No PR yet.

What is pushed:
- READ `flops.py`. The counter: `Step.repeats` and `Step.batch_rows`. Each index read of a stream batch must cover [0, L) in each row it touches, with one L over the step (`Counter._prefix`); otherwise None. Every batch the step reads counts together.
- READ `work.py`. `batch_count(steps, n)` adds a step's repeats only where `tokens == batch_rows * n`, with n from `row_positions` (the former inline `seq`). With no n, every count is read. `he_of_rows(..., batch)` keeps `1 - SHARED_PREFIX/n - positions/tokens` and records `repeats: {positions, tokens, measured}`.
- READ `formalism.py`, `grade.py`, `calculator.py`. The count is carried through `of_record`, `traced`, `traced_products`, `calculator.slowdown`'s `read`, `rho_of` and `per_period`. The strings take `P_REPEATS` plus "counted on the job's own batch"; a partial count also names P_BATCH for the other steps.

Tests, all in `tests/test_repeats.py` unless named:
1. test_batch_premise: the left-out positions are exactly the equal-word ones.
2. The counter against the brute-force definition.
3. `he_of_rows` with and without the count: shares fall exactly; non-token, gathered, self and left-out products don't move; with no count there is no key. The rows × n ≠ tokens step gives today's M (`batch_count` returns None).
4. A job on planted windows, from record to grade: repeats equal the definition on the served batches; M falls by the share; the strings name P_REPEATS. With the field removed it names P_BATCH.
5. Stop and continue: the repeats equal the unbroken job's and the stream's.

The tests ran here:
- RAN on my VM: `test_repeats.py`, `test_batch_premise.py`, `test_formalism.py`, `test_one_calculator.py`, `test_library.py`, `test_program_reading.py`, `test_value_reading.py`: 114 passed.
- RAN on my VM: `test_grade.py`, `test_flops.py`, `test_resume.py`: 114 passed.
- RAN on my VM: `test_as_math.py`, `test_annotate.py`, `test_of_record.py`, `test_products.py`: 89 passed.
- Not run yet: `test_run.py`. That and the fresh suites go on a node next.

Which shipped routes record `repeats`:
- RAN on my VM: a throwaway test ran one 4-step segment of each route on the planted stream. subset, ema, lora and qat each recorded 7 on every step, equal to the definition over the served batches (rows 2, tokens 2048 = 2 × 1024).
- None of the four records None. INFERRED from the routes' code: each reads `x[:, :-1]` of each chunk through the embedding, a read from each row's first position to ctx with one L. The targets go to cross_entropy, which is not an index read.
- None appears only for a step that reads a batch some other way: a shorter or gapped read, two lengths, a computed index, a window that isn't the batch's own storage, or a 1-D batch. Test 2's cases cover those.

Old records unchanged:
- RAN on my VM: one tiny job's record, with `repeats` and `batch_rows` removed, was dumped by the merge-base tree and by mine: grade, traced block, bound, published ρ, `calculator.slowdown` on its trace, RESTS_ON, RESTS_ON_CELL, TRACED_PRODUCTS, P_BATCH, the library's parts and `per_period`. The two dumps are byte-identical (`cmp`).
- The library table itself (every entry traced) I'll compare on the node: `library --out` from both trees, then diff.

Next, in order:
1. Merge main.
2. The suites (`--changed <merge-base>`, fresh), ruff, the ratchet gate, the library diff and the count's per-step cost, on a node.
3. The draft PR.
4. The README lines after the attention-price branch lands.
5. The 4.30.0 bump last.

