[2026-11-12T21:10Z] bb104ed4ac76
QUERY to training, Thu 12 Nov 1:09 PM PST (from `date`). **The merge of #9039's head and the three must-fixes are pushed on `cursor/training-attention-price-0cd4` at `519a974cf3`, merged with `main` at `5f0c008aa040`; the suites, the library's table at that head and both mutations are queued on nodes, and none has a verdict yet.**

Pushed (RAN, `git push`, no force; 70d32d57ee..519a974cf3):
- `1e277e11d7`: one merge of #9039's head `5c793f5ee59f`, which hasn't moved (RAN, `git fetch`). VERSION and CAPTURE are 4.25.0. The join is `_counted_dtype(attention, n, _computed_in(n))` inside #9039's `try/except Unpriced` in `structure.priced`, and the same in `products.step_gates`. `traced_dtype` keeps #9039's promotion body.
- `209a51db7b`: #9039's attention exception is out of the count-against-trace test (`tests/test_run.py`: `counted["gates"] == tr["circuit"]["gates"]`, with nothing excepted), out of the README, and out of test_grade's two `priced_apart` tests.
  - READ, `grade.py`: `priced_apart` has no attention clause of its own. It reads only rows in M (`work.in_m`), and attention's products are out of M. The exception lived in those tests and texts, and in `priced_apart_reason`'s case split by `traced_dtype`. I moved that split to `counted_dtype`, which is the same for every row that states no call. Read against the merge, the reason's two cases are right: an old capture whose rule prices a product apart from its circuit gets "retrace", and a current one gets "trace on the device of the job's kind".
- `fb61433743`, MF1, MF2 and SF1:
  - MF2: the marker reads `_computes_in(query)`, which is autocast's dtype where autocast is on for the query's device and the query is floating and not float64; otherwise the query's own dtype. A call whose output is in another dtype than its marker raises `CaptureRefused("trace", "scaled_dot_product_attention", ...)`.
  - New `Attending` case `upcast` (q and k `.float()` under autocast): it is marked and priced bf16. RAN (VM, `-k`): 14 passed.
  - Test of the refusal (`_computes_in` patched to float16).
  - SF1: a test parametrized over `weights` (`nn.MultiheadAttention` with `need_weights=True`, bmm rows bf16 as traced) and `bound` (a builtin bound before the capture inside a `handle_torch_function` wrapper; bmm rows float32). Neither states a call, and both are priced exactly as an unmarked capture prices them.
  - MF1: every sentence citing `flops.dtype_of` now says what the merged code does (READ, `flops.py`): the math path's rows name float32, and the scalings are named by `dtype_of` on `Counter._attention`'s stand-ins.
  - (d) and (e) are stated in `priced_dtype`'s edge list and in the README.
- SF2 and MF3: `ATTENTION_PREMISE` drops the provenance and names the card as `gates.py` does ("... on sm_120, the card whose BF16 pipeline prices a bf16 GEMM").
- `ba1c168d5f`, MF3: the premise is in `library.PARTS["premise"]` and in grade's `rests_on["attention"]`, and the block prints it.
  - Decided the plainest way: `rests_on["attention"]` is set where the search trace prices a row on the premise (`products.on_premise`: some row's `counted_dtype != traced_dtype`). A float32 call states `attention` but takes no premise, so a row merely stating `attention` doesn't set it.
- SF3: the sentence, not the refusal. READ: `tests/test_formalism.py` (the `at_411` case, ~line 983) reads a capture-4.11 trace through `calculator.slowdown` and expects `work.HE_RETRACE`, so a calculator refusal outside `READS` would change what a test reads. The sentence in `priced_dtype` says that a 4.23 or 4.24 trace is priced as traced, that the grade and the grader refuse it, and that the calculator, the library and `work.repriced` read it as it is.
- `65429f4c22`, N3: the grader test's assertion is split.

Runs in flight (each RAN, launched at `519a974cf3`):
- `r20261112-210519-bbb2`: both suites `--fresh`, with `VERITY_GRADER_SLOWDOWN_CODE`, then ruff and the ratchet gate.
- `r20261112-210553-0d07` (43 entries, 3 at a time, 20 GB) and `r20261112-210619-a96d` (xl, 28 GB): the library's table, each entry in its own process, the author's `lib.sh` way. Each entry also notes whether the capture refused it and how many attention rows M holds. I'll compare each entry's `library.json` with the author's "after" at `7a560a0e7dc` (`r20261112-171938-2636`, `r20261112-172004-f817`).
- `r20261112-210839-d5ac` (unmarked) and `r20261112-210852-efbd` (widened): mutations in clones. They run the review's set: `test_products`, `test_library` and `test_run` in full, and the grader's `test_slowdown_code` with the variable set. RAN on the VM: the widened mutation fails `[plain]`, `[wide]` and `[upcast]`.
  - INFERRED: my merge deletes none of the 13 and 12 tests the review's runs failed. My new tests add to each list `[upcast]`. The unmarked list also gains the refusal test and the count-against-trace test, now that the latter has no exception.

Left, in order: read the five runs; the body (SF4); the head named with each run's last line; my handoff note.

