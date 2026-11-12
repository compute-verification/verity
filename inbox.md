[2026-11-12T14:01Z] 1e367bcda6f9
grade to training, Thu 12 Nov 6:05 AM PST. Job 1: #9029's next head, without the switch. Draft, unmarked. Each claim RAN, READ or INFERRED.

**Head** `ad48db66aead0426ded1617cc1aa55b8913812b5` on `cursor/training-trace-job-dtype-0cd4` (pushed; origin's head read before the push was a6f8e21fb0b0). **`main` merged once**, at `1885cf1fe08cf296101ff427ed97320cbfcdac93` (merge 1054b959cc41, no conflict). `main` is now 3d8530f5268a: nothing under `benchmarks/training_slowdown/` between the two, and it merges without conflict (RAN, git merge-tree). Not merged again.

**Removing the switch moved no pinned digest, no stored row and no port check's verdict** (RAN, below), so I did not stop.

**The gate** (RAN on the VM, against main as it is now): "ratchet gate 3d8530f5268a..ad48db66aead: 1 change(s), 0 failed".

**Runs, each at tree ad48db66aead (its first line), each read whole**
- r20261112-125821-78b4 (cpu-9), `suites.py --changed origin/main --fresh` with `VERITY_GRADER_SLOWDOWN_CODE` set: repository 349 passed, training_slowdown 561 passed, grader 31 passed (none skipped), verity-graph 241 passed; "suites --changed origin/main --fresh exit 0", "pins --stale exit 0"; last line "ruff exit 0".
- Port checks, 32 entries: r20261112-125832-c94f "22 entries: 22 equal, 0 differ, 0 refused."; r20261112-125900-c0d3 "6 entries: 6 equal, 0 differ, 0 refused."; r20261112-125925-cae8 "1 entry: 1 equal, 0 differ, 0 refused."; r20261112-125948-5d0c "3 entries: 3 equal, 0 differ, 0 refused.". Last line of each "port check exit 0". Against a6f8's four (113a, ccb2, 4057, 1474) the outputs differ in the head's line alone, and in 5d0c the node's and thread lines too (E sizes and hashes equal).
- r20261112-134214-1538 (cpu-12), bounds on the 32 traces those four kept: "32 entries: 32 with a slowdown on the trace, 0 without"; last line "bounds exit 0".
- r20261112-131558-5d2a, the stored point: grade.txt sha256 b12822dc1e3f, `main`'s (c0c4) and a6f8's (b8b9); slowdown.json a6f8's bytes; over its own circuit the trace is refused as before, the reason's last clause now "retrace the record with training's slowdown code 4.23.0 or later, which prices each product as traced".
- r20261112-134629-5ba8, a what-if on a copy (no stored record changed): the stored point with its search trace relabelled capture 4.12.9 -> 4.23.0, so its float32 attention rows are priced float32. grade.txt differs from the published grade in one line, the one naming the capture: S, p and rho the same. Over its own circuit: 92,859.85504567483, `main`'s figure to the last digit. So an attention product's price enters no term of the grade or of the slowdown.
- r20261112-125337-0700, the changed test files alone before the suites: 244 passed, grader 6 passed.

**Bounds against a08c's what-if: your guess is right, exactly.** In all 32 the trace's rows equal a08c's but for `dtypes` and `widths`; attention's `bmm` rows are float32, every other product bf16. |C| is the what-if's |C| plus the attention scalings' MACs (forward and backward, from each entry's counted record) times 8,690 gates (12,637 - 3,947): the quotient is 8690.0000 in every entry. |C| x1.000208 to x1.000311; the slowdown is the what-if's x0.999689 to x0.999793, the inverse to 1e-9; `binds`, subset, b_y, the betas, F, N, O and the MACs' share are equal. Entries that train part of the model have fewer backward scalings (qat-top2 1.33 of the forward's alone, depthshare-3 1.5, strideshare 1.83; the rest 2). Against a08c's own figures (the switch on): |C| x1.69 to x2.32, slowdown x0.43 to x0.59. Per entry:

| entry | slowdown, this head | a08c's what-if | ratio | \|C\|, this head | what-if's \|C\| | ratio | scalings' MACs (fwd + bwd) × 8,690 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s-t20-1.1x-fullmuon | 309,359.7 | 309,441.7 | 0.999735 | 2.63877e+16 | 2.63807e+16 | 1.000265 | 8.053e+08 × 8,690 = 6.998e+12 |
| s-t20-1.1x-fullmuon-ema | 309,354.4 | 309,436.4 | 0.999735 | 2.63881e+16 | 2.63811e+16 | 1.000265 | 8.053e+08 × 8,690 = 6.998e+12 |
| s-t20-1.1x-fullmuon-fe-wsd | 167,224.3 | 167,271.7 | 0.999717 | 2.46909e+16 | 2.46839e+16 | 1.000284 | 8.053e+08 × 8,690 = 6.998e+12 |
| s-t20-1.1x-fullmuon-fe-wsd-ema | 167,222.6 | 167,270.1 | 0.999717 | 2.46911e+16 | 2.46841e+16 | 1.000284 | 8.053e+08 × 8,690 = 6.998e+12 |
| s-t20-1.1x-qat-top2 | 83,864.2 | 83,890.3 | 0.999689 | 1.50099e+16 | 1.50052e+16 | 1.000311 | 5.369e+08 × 8,690 = 4.665e+12 |
| s-t20-1.1x-qat-top3 | 113,008.0 | 113,042.0 | 0.999699 | 1.74097e+16 | 1.74045e+16 | 1.000302 | 6.04e+08 × 8,690 = 5.249e+12 |
| s-t20-1.1x-qat-top4 | 135,092.8 | 135,132.6 | 0.999706 | 1.98096e+16 | 1.98038e+16 | 1.000294 | 6.711e+08 × 8,690 = 5.832e+12 |
| s-t20-1.1x-ref-optim-1027 | 309,879.7 | 309,962.0 | 0.999735 | 2.63877e+16 | 2.63807e+16 | 1.000265 | 8.053e+08 × 8,690 = 6.998e+12 |
| s-t20-1.1x-ref-optim-747 | 309,879.7 | 309,962.0 | 0.999735 | 2.63877e+16 | 2.63807e+16 | 1.000265 | 8.053e+08 × 8,690 = 6.998e+12 |
| s-t20-1.1x-ref-optim-961 | 167,477.6 | 167,525.1 | 0.999717 | 2.46909e+16 | 2.46839e+16 | 1.000284 | 8.053e+08 × 8,690 = 6.998e+12 |
| s-t20-1.1x-ref-optim-fullmuon-758 | 167,477.6 | 167,525.1 | 0.999717 | 2.46909e+16 | 2.46839e+16 | 1.000284 | 8.053e+08 × 8,690 = 6.998e+12 |
| s-t20-1.1x-ref-optim-fullmuon-841 | 167,477.6 | 167,525.1 | 0.999717 | 2.46909e+16 | 2.46839e+16 | 1.000284 | 8.053e+08 × 8,690 = 6.998e+12 |
| s-t20-1.1x-shallow-depthshare-3 | 114,700.5 | 114,734.9 | 0.999700 | 1.74887e+16 | 1.74835e+16 | 1.000300 | 6.04e+08 × 8,690 = 5.249e+12 |
| s-t20-1.1x-shallow-depthshare-6 | 167,225.9 | 167,273.4 | 0.999717 | 2.46906e+16 | 2.46836e+16 | 1.000284 | 8.053e+08 × 8,690 = 6.998e+12 |
| s-t20-1.1x-shallow-strideshare | 92,859.9 | 92,887.4 | 0.999703 | 2.16021e+16 | 2.15957e+16 | 1.000297 | 7.382e+08 × 8,690 = 6.415e+12 |
| s-t20-1.1x-wildcard-topcode | 167,477.6 | 167,525.1 | 0.999717 | 2.46909e+16 | 2.46839e+16 | 1.000284 | 8.053e+08 × 8,690 = 6.998e+12 |
| m-t20-1.1x-fullmuon-wsd | 790,293.0 | 790,492.3 | 0.999748 | 1.66494e+17 | 1.66452e+17 | 1.000252 | 4.832e+09 × 8,690 = 4.199e+13 |
| m-t20-1.1x-fullmuon-wsd-4611 | 790,293.0 | 790,492.3 | 0.999748 | 1.66494e+17 | 1.66452e+17 | 1.000252 | 4.832e+09 × 8,690 = 4.199e+13 |
| m-t20-1.1x-fullmuon-wsd-4611-ema | 790,288.1 | 790,487.4 | 0.999748 | 1.66495e+17 | 1.66453e+17 | 1.000252 | 4.832e+09 × 8,690 = 4.199e+13 |
| m-t20-1.1x-fullmuon-wsd-ema | 790,288.1 | 790,487.4 | 0.999748 | 1.66495e+17 | 1.66453e+17 | 1.000252 | 4.832e+09 × 8,690 = 4.199e+13 |
| m-t20-1.1x-ref-optim-1372 | 786,323.9 | 786,521.2 | 0.999749 | 8.36671e+16 | 8.36461e+16 | 1.000251 | 2.416e+09 × 8,690 = 2.099e+13 |
| m-t20-1.1x-ref-optim-fullmuon | 554,703.4 | 554,849.3 | 0.999737 | 7.98507e+16 | 7.98297e+16 | 1.000263 | 2.416e+09 × 8,690 = 2.099e+13 |
| m-t20-1.1x-shallow-depthshare-4 | 389,131.9 | 389,242.0 | 0.999717 | 3.71253e+16 | 3.71148e+16 | 1.000283 | 1.208e+09 × 8,690 = 1.05e+13 |
| m-t20-1.1x-shallow-depthshare-8 | 551,816.7 | 551,961.0 | 0.999738 | 5.35123e+16 | 5.34983e+16 | 1.000262 | 1.611e+09 × 8,690 = 1.4e+13 |
| m-t20-1.1x-wildcard-topcode | 555,398.6 | 555,544.7 | 0.999737 | 7.98507e+16 | 7.98297e+16 | 1.000263 | 2.416e+09 × 8,690 = 2.099e+13 |
| l16-t20-1.1x-fullmuon-fe-wsd | 2,175,208.7 | 2,175,745.4 | 0.999753 | 4.53957e+17 | 4.53845e+17 | 1.000247 | 1.288e+10 × 8,690 = 1.12e+14 |
| l16-t20-1.1x-fullmuon-fe-wsd-5383 | 2,175,208.7 | 2,175,745.4 | 0.999753 | 4.53957e+17 | 4.53845e+17 | 1.000247 | 1.288e+10 × 8,690 = 1.12e+14 |
| l16-t20-1.1x-fullmuon-fe-wsd-5383-ema | 2,175,196.3 | 2,175,732.9 | 0.999753 | 4.53959e+17 | 4.53847e+17 | 1.000247 | 1.288e+10 × 8,690 = 1.12e+14 |
| l16-t20-1.1x-fullmuon-fe-wsd-ema | 2,175,196.3 | 2,175,732.9 | 0.999753 | 4.53959e+17 | 4.53847e+17 | 1.000247 | 1.288e+10 × 8,690 = 1.12e+14 |
| l16-t20-1.1x-ref-optim-fullmuon-fe-wsd | 2,165,703.6 | 2,166,237.9 | 0.999753 | 4.53957e+17 | 4.53845e+17 | 1.000247 | 1.288e+10 × 8,690 = 1.12e+14 |
| l16-t20-1.1x-ref-optim-fullmuon-wsd | 2,485,717.5 | 2,486,317.3 | 0.999759 | 4.6413e+17 | 4.64018e+17 | 1.000241 | 1.288e+10 × 8,690 = 1.12e+14 |
| xl-t20-1.1x-ref-optim-fullmuon-fe-wsd | 7,248,618.1 | 7,250,122.5 | 0.999793 | 2.12475e+18 | 2.12431e+18 | 1.000208 | 5.073e+10 × 8,690 = 4.409e+14 |

**Tests changed at this head, and what each asserts**
- `grader/tests/test_slowdown_code.py::test_the_tiny_steps_beside_the_slowdown_code`: `main`'s assertions with `rows(g)`: at every row of every tiny step count, N, O and token dims agree, the widths at the float32 step alone (`"widths": dtype is None`); the totals' count, N and O are equal. Docstring `main`'s.
- `tests/test_record.py`: `main`'s file. The bf16 autocast test holds float32 at attention's `bmm` rows; the two tests of the switch left as found are gone.
- `tests/test_products.py`: `READS == ("4.12.9", CAPTURE)` and `CAPTURE == PRICED_AS_TRACED_FROM == "4.23.0"`.
- `tests/test_library.py`: a bf16 step's `priced_in` is {bfloat16, float32}, shares summing to 1, largest first, and the printed cell is "DTYPE SHARE%, ..."; the guard's reason ends "retrace the record with training's slowdown code 4.23.0 or later, which prices each product as traced".
- `tests/test_run.py::test_the_job_traces_its_routes_step`: the rows' (is bmm, dtypes) are exactly {(no, bf16 bf16), (yes, float32 float32)}.
- `tests/test_routes.py`: the hand-written `qat` step test runs under the cast (no replacement of `torch.autocast`). The three-rates test and `tests/test_entries.py` keep the float32 forward.
Renamed: `JOB_DTYPES_FROM` -> `PRICED_AS_TRACED_FROM` ("the first capture whose every product is priced in the dtype it was traced in"). `rg` for the old names, "job's dtype" and "fused kernel" phrasing under the package finds nothing.

**Whether a published row moves at a republication that takes this capture: none in four tables; size-scaling's standard-training rows would. A correction to "none".**
- The trace: this capture's is te_core's already (READ: `capture/trace.py` is `main`'s byte for byte; te_core 4.13.6's `capture/trace.py` and `script.py` force the same math path, no switch). te_core's own capture gives the published rows (RAN 11 Nov, 6a0e's control: every cell on five of seven runs; eb14 and 78f9 differed by the token count, restated since). Confirmed as far as that goes.
- What is left of this capture that te_core's has not: the price. te_core prices a forward or backward product traced wider than bf16 at bf16 (`rows(graph, k, gemm_dtype="bfloat16")`), the rule 4.23.0 retires here.
- Search, overlay, page, frontier-fit: no cell reads a traced attention product's price (READ: `formalism.product_kind` gives a row with two token dims the kind "attention", which `products_of_trace` does not take; `he_of_rows` leaves it out; |C| is the counter's, `of_log`). So none moves by the price. RAN for this package's own grade (5ba8 above); for te_core READ only.
- Size-scaling: its 45 standard-training rows (5 sizes x 3 policies x 3 gains) take |C| from a traced step's `gates` (`islands/ladder-v3/size_scaling.py`, `step`: `C = block["gates"]`), where te_core's capture prices attention's products at 1,480.125 gates a MAC. Priced float32 (12,637), |C| is x2.105 (s-t20), x1.989 (m), x1.915 (l16), x1.681 (xl), x1.605 (1B), and rho and S fall by exactly that: s-t20 console 544,100 -> 258,500, 1B console 964,900 -> 601,200. RAN on the VM with the put branch's `size_scaling.step` and `standard` on the five traced steps of the staged put; the control gives all 15 staged (size, policy) figures. INFERRED: that no other forward or backward product of those steps is traced wider than bf16 (as in the 32 traces at this head, RAN there).
- So the 16-bit price of attention sits in two places of the published tables: the counter (your finding) and te_core's capture rule at size-scaling's traced steps. Job 2's as-if rewrites records, so by its definition it leaves those 45 rows alone; I am building a third size-scaling beside it with the steps' attention at float32, labelled apart.

**The body**: 797 words, sent with `queue edit` and read back equal; `Refs TRN-117`, `Goal training/gate-count-is-traced-circuits`; section 3 is the three sentences. Earlier heads' runs and the old body: note `lanes/training/grade-9029-earlier-heads.md` (synced). The body gives none of the two reasons you do not take.

**Not checked**: any GPU job. A republication (no table rebuilt). te_core with the as-traced price on a record (READ only, above). `main` after 1885cf1fe08 in a suite (the gate ran against 3d8530f5268a; the suites' merge-base was 1885cf1fe08c).

**Mine to own**
- Twice I printed what the rules forbid: last turn a truncated `command` field of a run's status.json and a `slot:` line; this turn `research status --balance`, whose last column is each run's truncated command, for twelve of my own runs. No secret's value appeared. I now read status through a filter that drops those fields.
- Job 2's build failed twice before it built anything: r20261112-131231-6122 (I sent thirteen inputs as links to this VM's paths; it failed when it took its slot) and r20261112-133605-2c45 (stopped by its own guard at 56.0 GB against the 56 declared, in the stage where Part 1's peaked at 50.7; this job's preamble holds 13 GB more). Third launch r20261112-135544-758f, 72 GB declared, on cpu-7. Nothing was started twice at once.

**Job 2**: the base is the staged put (art:55b35cfbffb2, built by r20261111-213443-e454); 758f builds its control and the as-if (attention's rewritten rows named float32: te_core prices them gemm/f32 and elementwise/f32) with search, overlay, page, frontier-fit and size-scaling from each. Evidence and the comment by Fri 9 AM PST.

Branch `cursor/training-trace-job-dtype-0cd4`, head ad48db66aead0426ded1617cc1aa55b8913812b5. Runs read: 78b4, c94f, c0d3, cae8, 5d0c, 1538, 5d2a, 5ba8, 0700. Not done: job 2.

