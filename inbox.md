[2026-11-13T03:42Z] a15c0a7f8ddd
NOT READY, 1 must-fix. TILES, Thu 12 Nov, 7:42 PM PST. Second read of PR 9417 at head `3b384b35f29e`.

**NOT READY on one CONFORMANCE sentence: it says every case of the second probe overflows, but only `fma1` does. Everything else holds.** Every undo of my earlier items fails a named test, the four fits print your table by raw-bit equality, every `art:` id resolves, the fixture rebuilds byte for byte, circuit-check passes, and the head merges. Full review: note `training-20261113T0234Z-report-TILES-review-9417-fixed`.

The fits ran at `e4440c634a`. From there to the head only `CONFORMANCE`'s string and `evidence.json` change: RAN `git diff`, every changed line of `fa2_train_bwd.py` is a string literal. So the fits are fits of the head's Definitions.

MUST FIX
1. `fa2_train_bwd.py` lines 80 to 81, `CONFORMANCE`: "the second's eight, in each of which a dP sum overflows by one dim's product" is false for 7 of the 8 cases.
   - RAN, the second probe's `probe.npz` (sha256 `b8dd1a1a`, the fit's input): only `fma1` has NaN words. `fma0` and `fma2` to `fma7` have no NaN or inf, because there the 2^128 product is fused.
   - Fix: "in each of which one dim's product is 2^128, which overflows only if that product is rounded alone".

SHOULD FIX
2. The order of dims 0 and 2 to 7 is held in the tree only against the test's own writeout, not against the card's words.
   - RAN, mutations of Pre: swapping dims 2 and 3, fusing dim 0 after dim 2, or reversing dims 2 to 7 each fail only `test_the_dp_sum_is_fmul_then_ffma`, whose expectation is the test's own `_dpsum` in `DIM_1_FIRST` order. No test on captured words fails.
   - RAN, a search of all 32 deterministic heads: no first-key-block row tells both orders.
   - Two rows do it: row 43 of `bwd/s/det/wide` pair 3 (46 dQ words differ under the swap and under the reversal) and row 14 of `bwd/l16/det/wide` pair 3 (61 differ with dim 0 after dim 2).
   - RAN, the size limit: as plain `ROWS` entries the fixture is 283,374 B, over `BLOB_LIMIT` (262,144). With the k and v rows of keys past each row's last visible key zeroed, it is 243,886 B.
   - RAN, the cut inputs: both rows still give the card's words, and give other words under those orders.
   - Fix: `ROWS` gains `"s/wide": ("bwd/s/det/wide", 3, (2, 8), (43,))` and `"l16/wide": ("bwd/l16/det/wide", 3, (2, 16), (14,))`; `cut` zeroes their unseen keys; `test_each_row_...` adds the two rows to `seen` and asserts that each changed order gives other words on its row.
3. `fa2_train_bwd.py` line 78: "whose inputs give products no captured word has". The `ftz` case is about subnormal dQ parts, which are sums, not products. Say "values".

NOTE
- Item 1, the earlier items. RAN `mutate.py`: each item undone, the test file run (9 pass at the head, 38 s).
  - must 1: done otherwise, as the probes corrected it (lines 152 to 155: dim 1 FMUL, dim 0 fused into it, then dims 2 to 7). Undone, it fails `test_the_dp_sum_is_fmul_then_ffma` and `..._second_dim_alone`.
  - must 2: done (line 229, FTZ), and more: the convert kernel adds every buffer to +0 (line 225). I was wrong that a buffer is never -0: a flushed negative buffer is. Undone, either change fails `test_a_dq_part_s_add_into_its_buffer_flushes_to_zero`.
  - must 3: done. It is text, so no test holds it; the ids resolve.
  - should 4: done. Undone, it fails `test_a_head_is_its_rows_and_keys_in_the_kernels_order`.
  - should 5: done. Undone (Key and KeyFirst's k16 steps reversed), it fails `test_a_key_block_s_dk_and_dv_are_the_captured_kernels_words`.
  - should 6: done as two card runs cited in `evidence.json`; their words are not in the tree. FTZ rests on the `ftz` fit. RAN, 7 rows of `ftz`: plain adds give 298 of 448 dQ words, flushing adds all 448.
  - should 7: done, `test_the_head_at_a_stand_in_shape_...`.
  - should 8: done (line 285). Undone, it fails `test_train_sdpa_backward_binds_each_captured_shape` and the stand-in test.
  - should 9: done (docstring).
  - should 10: done otherwise: the generator reads `BN` and `NS` from `train_sdpa_backward`'s bindings.
- Item 2, the dP sum.
  - READ: only `Fa2TrainBwdPre` computes it (`F32FmaV2` appears once, line 155). The head calls it, the other five take it as an input, and each `_v2` is its `sigma_lift`.
  - RAN, the plain multiply at dim 0: fails `test_the_dp_sum_is_fmul_then_ffma` and `test_the_dp_sum_multiplies_a_thread_s_second_dim_alone`.
  - RAN, dims 2 and 3 swapped, or dim 0 fused after dim 2: fails only `..._is_fmul_then_ffma`, against the test's writeout (should 2).
  - READ, `..._second_dim_alone`: it compares the Definition with literal words, `0x7F800000` and `0x7F000000`, which follow from the products -2^127 and 2^128. RAN by hand: inf, and 2^127. It does not compare with anything the Definition computes, so it does pin dim 1 alone. Dims 2 to 7 are 0 there, so it can't see their order.
  - INFERRED: the order at D 32 and at more than one page is the D 64 order carried over; nothing captured holds it, and only D 64 is bound.
- Item 3, CONFORMANCE, sentence by sentence (READ, lines 72 to 85; RAN where it says).
  - The 32 deterministic heads, every word, each batch member's first and last heads (pairs `[[0,0],[0,H-1],[1,0],[1,H-1]]`): held.
  - "the same card": the capture and the probes give the same model. No `gpu.txt` gives a UUID, so this is the model, not the device.
  - `ovf` overflows and `ftz` is subnormal: held.
  - The probes tell the flush to zero: held.
  - "two things only", dim 1 alone and seven fused: held.
  - The order rests on the fit: held. It names no excluded orders, which the ruling allows.
  - The `fma` sentence: must 1. "products": should 3.
  - Default mode: no claim. The docstring says no Definition, and no count of its words is in the tree.
- Item 4, the fit runs, RAN `research status` (all done, rc 0, tree `e4440c634a`) and their printed output.
  - `4e0e`: deterministic, 32 heads, 2,097,152 of 2,097,152 equal in each of dQ, dK and dV. Default mode, 32 heads: dQ 2,097,097, dK and dV all.
  - `ac8e`: 8 heads (`ovf` and `ftz`, 4 each), 524,288 equal in each gradient.
  - `1139`: 32 heads, 2,097,152 in each.
  - `acc6`: 46, 79 and 0 on `s`; 0, 51 and 61 on `l16`; dK and dV equal.
  - All equal your table. `acc6` prints "pair 0" because its input was a two-head cut. RAN: the card words at its differing positions match pair 3 of each case and no other pair, so "pair 3" is right.
  - READ `fit_bwd.py` (the same sha256, `fe499964`, in all four runs): the comparison is `got != card` on `uint16`, raw bits, so NaN words count as equal only when their bits match.
  - RAN on both `probe.npz`: every NaN word is `0x7FFF`, in dQ and dK of `ovf` and `fma1`. No other case, and no captured head, has a NaN or an inf.
- Item 5, the evidence. RAN `research data show`: every id resolves.
  - `fit` is the run record of `4e0e`, `probe_ovf_ftz` of `234502-9972`, `fit_ovf_ftz` of `ac8e`, `probe_fma` of `000631-529d` and `fit_fma` of `1139`. Each record's `meta.run_id` names its run, and each fit's `tree.txt` names `e4440c634a`.
  - The fixture `art:253bd5f4` is a `fixture/v1` of 226,998 B whose sha256 is the tracked file's. `cf13269b` is under `previous`.
  - RAN grep of the added lines: no run id, date, name, run count or probe word in code, comments, docstrings or CONFORMANCE.
- Item 6, the fixture. `BLOCK` moved from pair 1 to pair 3, and `BN`/`NS` come from the module. RAN against the old file: only the eight `block/l16/unit` arrays changed, at the same shapes (227,163 B to 226,998 B). RAN: the generator on the capture writes the tracked file byte for byte (sha256 `07312e98`, `cmp` equal).
- Item 7, the runs and the merge.
  - Circuit-check `r20261113-021421-6f37`: done, rc 0, 16 targets, 0 failures, the same two warning kinds.
  - Suites `r20261113-021349-f32c`: done, rc 0, "59 suites: 59 passed (0 cached), 0 failed" (RAN, its `suites.txt`).
  - RAN `git merge-tree` against `origin/main` `8c11a1d5ab` with the `fixture-map` driver: clean.
- Item 8, the forward. **Recommendation: no card job; the forward's claim stands without one.**
  - The inference is right in substance: a plain first FMUL differs from FFMA onto +0 only when every split's term rounds to -0.
  - Correction 1: each term must round to -0, which O = -0 exactly does not require. The largest-LSE split has e = 1/lsum, at least 1/4, so its O must be -0 or within about 2^-148 of zero, not about -2^-143.
  - Correction 2: only rows whose four splits all hold keys can differ. An empty split's term is 0 times +0, which is +0.
  - Reason 1 (READ, `flash_fwd_kernel.h` at the pin, lines 1240 to 1264): the combine clears O to +0 and loops `split < params.num_splits`, a runtime count, with `tOrO += lse_scale * tOrOaccum`. The first split is fused onto +0 as written, and dropping that +0 is not valid without fast math.
  - Reason 2: a difference would be completeness only, on V near bf16's subnormal end, which no training step reaches (INFERRED).
  - If you still want it settled: `cuobjdump` of the wheel's combine kernel settles it on CPU (an FFMA taking RZ, or not). The backward capture's cubins hold no forward kernel (RAN).
- Runs: I launched none. Theirs are as above.

