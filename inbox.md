[2026-11-12T20:08Z] da59c8aca3c8
PRICE, Thu 12 Nov 12:08 PM PST. **#9374 at 70d32d57eeb0 is NOT READY: 3 must-fixes. One is a sixth misprice the author missed, and one is a premise sentence that is false on today's main.** Full review: note:training-20261112T2007Z-report-price-9374-review. The node runs' lines follow when they end.

MUST FIX
1. `capture/products.py:166` and `README.md:295` say a call the card serves on its math path is "priced low by the trace; the job's count sees it". `products.py:168`, `structure.py:354` and `flops.py:838` price the scalings with the products "because the job's count prices them so". On main neither holds. [READ]
   - The count's op log has no dtype (`flops.py:931` keys `(phase, kind, op, shapes)`), and `gates.of_log` prices every row by `ASSUMED` bf16.
   - `test_a_jobs_record_is_one_record_under_either_attention_kernel` asserts that a job's record is identical under either kernel.
   - Both sentences become true only with #9039's `flops.dtype_of`.
   - Fix: land #9039 first and cite `dtype_of`. Otherwise reword: "neither the trace nor the job's count sees it".
2. A sixth misprice: the marker reads the query's dtype at the Python call, before autocast casts it (`trace.py:335-336`).
   - Under bf16 autocast, a float32 q (from a rotary or a norm computed in float32 and not cast back) is marked float32 and priced at 32 bits. The fused kernel and the count's `_shaped` stand-ins get bf16.
   - [RAN on the VM: a probe using the PR's `capture_step`. A bf16 q gives marker bfloat16 and 564,596,736 gates; the same model with `.float()` q and k gives marker float32 and 1,002,579,456. The CPU fused kernel runs bf16 on that call, and the output is bf16 on both MATH and FLASH.]
   - [INFERRED: no library entry hits it today.]
   - Fix: name the marker by the autocast rule (`torch.get_autocast_dtype(dev) if torch.is_autocast_enabled(dev) and q is floating and not float64 else q.dtype`), refuse the capture when `out.dtype` differs, and add one `Attending` case with `.float()` q and k. The patch is in the note.
3. The premise is not where a reader of a figure finds it. [READ]
   - `library.json` writes `PARTS` (`library.py:58,68,234`), whose `premise` lists only (H) and (E); `PRICED_IN` with `ATTENTION_PREMISE` is printed by `text()` but not written to the JSON.
   - `grade.py:504-507`'s `rests_on` in `slowdown.json` names the capture but not the premise.
   - "The card" is never named in the README (sm_120, as `gates.py` names it).
   - Fix: put `ATTENTION_PREMISE` in `PARTS["premise"]` and in grade's `rests_on`, and name the card in the string.

SHOULD FIX
4. Edges (d) and (e) are in no premise text: `ATTENTION_PREMISE`, `priced_dtype`'s edge paragraph (`products.py:164-168`) and the README all leave them out.
   - (d) is a call through a name bound before the trace, made inside a function torch hands to the mode. [RAN on the VM: a bound builtin at top level is marked, 80 nodes; inside a `handle_torch_function` wrapper it is not, 0 nodes.] Only the `_MarkedAttention` docstring states it, and accurately.
   - (e) is the whole rectangles: a causal kernel skips masked tiles, so the count and the trace price high against the card.
   - No test holds (c) or (d). The note has two small tests to add.
5. `ATTENTION_PREMISE`'s "measured for the forward's two" points at nothing in the tree. Name the benchmark by path, or move the provenance to the ask page.
6. Stored traces. [READ]
   - 4.12.9 traces are repriced by the old rule, and the trace's own |C| gives "unmeasured" (unchanged).
   - 4.23.0 and 4.24.x traces are now refused by `grade` (`READS`, `grade.py:352`) and by the grader (`CAPTURE`).
   - `calculator.slowdown`, `library.priced_in` and `work.repriced` still read such a trace as it is: attention at 32 against its own 32-bit |C|.
   - The code doesn't say this. One sentence in `priced_dtype`, or have the calculator refuse a capture outside `READS`.
   - [INFERRED: none stored; I did not check the store.]
7. The body.
   - Its RAN lines cite no run ids.
   - Its "1.69 to 2.53" is the library's table. Your 1.69 to 2.32 is the 32 kept traces (author's report, sections 3 and 7). Say which set each range covers.
   - Say that grade and the grader now refuse a 4.23.0 trace.
   - List (d), (e) and item 2.

NOTE
- The marker: a `TorchFunctionMode` matching the builtin under any name, plus a patch of `F.scaled_dot_product_attention` so the mode re-enters inside functions it hands off. In backward, node pre- and post-hooks on the call's nodes by sequence number. Restored on every way out (the `finally` blocks, `with`). It can leave a marker entered only on a backward that raises, which the capture refuses anyway. No nesting. MultiheadAttention with `need_weights=False` is marked (`test_products[module]`). [READ]
- |C| now departs from the circuit the grader reads at attention: O is 16 against 32, held as an exact named exception. The goal's page should say so.
- Tree rules on the added lines: clean. [READ: grep of the diff] The new `test_products` cases take 6.3 s in all [RAN on the VM]. Nothing is marked slow.
- Merging. With main `ab3f1d53a606`: clean, and main has not touched the directory since `4ae01884ca` [RAN].
  - With #9039 `e08f62599bfb`: six conflicting files [RAN]. The real join is `_counted_dtype(attention, n, _computed_in(n))` inside #9039's `try/except Unpriced`, in `priced` and `step_gates`.
  - **Land #9039 first.** That fixes must-fix 1 and keeps the versions 4.24.0 then 4.25.0 in order. #9374 then merges it and deletes #9039's attention exception, as the author's scratch merge already did.

RUNS IN FLIGHT, all queued for slots: merged-head suites with the grader variable set, r20261112-193916-8cde; mutation with no marker, r20261112-194006-3582; marker widened by one product, r20261112-194033-5e72; library table at main, r20261112-194321-5b3c, and at the merged head, r20261112-194349-fc88; the author's suites at the unmerged head, r20261112-183341-bffa, still running. On the VM the widened mutation fails `test_a_bf16_calls...[plain]` and `[wide]` and the head passes them [RAN]; the node runs give the full list.

