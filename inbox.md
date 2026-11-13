[2026-11-13T00:40Z] 58f816eedcae
BACKWARD2, Thu 12 Nov 4:40 PM PST

**The fits at the fixed head e4440c634a are running on CPU; no single captured head tells all three orders, so each order is fitted on two heads.**

- RAN r20261113-002701-acc6 (cpu-14): condition 2. The kernel's order (control) and the three changed orders (dims 2 and 3 swapped; dims 2 to 7 reversed; dim 0 fused after dim 2), each on two whole captured heads: `bwd/s/det/wide` pair 3 and `bwd/l16/det/wide` pair 3.
- INFERRED (numpy twin over 32 heads, not a Definition fit): the swap changes 46 dQ words, all on `s/det/wide` pair 3. Dim 0 after dim 2 changes 61 dQ words, all on `l16/det/wide` pair 3. The reversal changes 807 dQ and 28 dK words over 12 heads. So two of the three orders rest on one head each.
- RAN (plumbing, VM, the first 128 rows of `s/det/wide` pair 3): with dims 2 and 3 swapped, 46 dQ words differ from the card, matching the twin.
- RAN r20261113-002743-4e0e (cpu-3): the 64 captured heads. RAN r20261113-002814-ac8e: the first probe's 8 heads (ovf, ftz). RAN r20261113-002826-1139: the second probe's 32 heads (fma0 to fma7). Every word, raw bits.
- READ (the probes' npz): every card NaN word is 0x7FFF: dQ and dK of ovf and fma1, all 262,144 a gradient. No other case has a NaN or an inf. The fit compares raw uint16 bits, so another NaN pattern would count as a differing word.
- Forward question (READ `fa2_train.py`): the forward makes a choice of the same kind in one place only, and it matters less. `Fa2TrainCombine_v1` sums each split's product onto +0 by FFMA, with no plain multiply. Its products are f32 times f32, which are not exact, so the forward's captured words already test which product is rounded alone. INFERRED: the only open difference is FFMA onto +0 versus FMUL for split 0, which differ only when the whole sum is -0 (bf16 0x0000 against 0x8000). A probe that tells it: a split-KV row in which every split's O is -0 in one column. An empty split adds a +0 product, so all four splits must be non-empty. I change nothing for it.
- One question for condition 4: CONFORMANCE has to cite something for "the probes tell two things". My plan is to put the two probes' run records, and their fits' records, in `verity/definitions/evidence.json` beside the capture's, and have CONFORMANCE read them by key. That is the same pattern the capture uses: no id in code, and no probe words in the tree. If you want them not at all, CONFORMANCE drops the probe clause and I remove the keys.

Next: the table and the head once the fits end. Then suites and circuit-check at that head.

