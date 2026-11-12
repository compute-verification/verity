[2026-11-12T10:55Z] 45b0da157b68
PRICER to training (TRN-135), round 4 report, Thu 12 Nov 10:55Z. Draft PR #9170, unmarked.

**Head:** 6242aba2e0 (pushed; origin is the same).

**Runs (RAN):**
- r20261112-100830-a119, tree 08e15ce27b, both suites `--fresh`. Last line: "2 suites: 2 passed (0 cached), 0 failed; 9.2 min of test time" (565 and 38 passed).
- r20261112-103602-b462, tree 6242aba2e0: the tiny lf readings at each k ("never_above: held"), then both suites `--fresh`. Last line: "2 suites: 2 passed (0 cached), 0 failed; 9.2 min of test time" (571 and 38 passed).
- Ratchet gate, `ratchet_gate.py origin/main HEAD`: "ratchet gate afa939410f6f..6242aba2e0ed: 1 change(s), 0 failed".
- Two earlier runs were cancelled before they started, because the head moved. One more failed on my shell quoting and was rerun.

**The rule** (default reading, nothing stated, in `work.py`'s docstring), in one paragraph. A softmax's rows, each over a row of a product's outputs through one-to-one ops plus a stated constant, differ pairwise when each reads another member of the product and the constant's rows differ beyond a shift (in which words are −inf, or in the finite words less the first). The softmax is shift-invariant, so the add alone makes nothing specific. Its columns differ when the constant's columns differ and each has a finite word: two keys of one token give columns whose ratio is exp of the constant's difference. That case covers dV = Pᵀ·dA. A product is specific when its x's slices are such rows or columns (or specific and read one to one) and w's rows differ (specific, θ′, or a product's distinct output columns). A primitive keeps a value specific only when it is one to one in a specific argument (`INJECTIVE`, with a nonzero constant for a multiply) and no argument is an unstated input, such as a rope table, which may hold a 0. Everything it can't decide is gathered. For the value-batch limit, a batch whose members read rows of current inputs against an argument they share gives those rows a gather's status (`shared_rows`), since two value rows may be equal.

**Your claims:**
- The mask's add alone does not make a score specific: HOLDS. In case (b), the L1 product right after the add repeats 16 pairs, and the rule leaves it out.
- The reduce after it does, on one axis, with P(q, k) = P(q, k'): HOLDS for words. As vectors, P's rows differ, and so do its columns, where the causal mask's columns differ; the rule uses both.
- Your lean rule: taken, restricted to a softmax as the reduce, and with "rows differ" meaning beyond a shift.
- "Any op with a specific argument gives a specific value" is too strong: HOLDS. With a ReLU MLP, that rule charges the down-projection at 144 distinct pairs of 256 MACs, below keep's 192. The test catches it: I mutated the rule and it failed.
- Simpler safe rules taken: P's columns, and the weak side on unstated inputs and value batches.

**The value test** (`tests/test_value_reading.py`, 12 tests). Each case is evaluated word by word at random weights, at iteration 3, on the batch [[1,2,1,3,5],[4,1,4,6,7]]. Charged MACs, then distinct pairs:
- sdpa: 1,920 and 1,904. The only repeats are dQ·Wq L2 at position 0, which keep covers.
- After the mask: 2,048 and 2,048. The L1 product after the add is left out, with 16 repeats.
- rope: 1,664 and 1,664.
- ReLU: 2,432 and 2,416. The down-projection is left out, at 144 distinct of 256.
- lf miniature: 4,736 and 4,736.
- (c), the first layer's P·V: left out in every case, with repeats.
- A value batch with two equal rows: nothing is charged, and the down-projection repeats pairs.
- Your rulings 2 and 3 are in: the premise is stated once in `work.py`, in `SHARED_PREFIX`'s terms. Six tests assert a new repeat at each break: a leading run, one token and two rows sharing two tokens on the lf miniature under both readings, and prefixes in proportion on sdpa. With rope, prefixes in proportion give no new repeat, which is asserted too.

**Rope Layout, rules (a) to (c), on the held batch:** safe. Its only repeats are dQ·Wq at position 0 (16 each), which keep covers.

**Figures, the tiny lf step** (RAN, r20261112-103602-b462). Charged MACs an iteration, at every k:
- trace's M: 9,216,000;
- nothing stated (the rule): 6,160,384;
- rope named: 6,684,672, the earlier INFERRED figure, now RAN;
- layout alone: 0.

Blocks at the calculator's F, by k, in the order calculator / trace's M / the rule / rope:
- 65,536: 0.79974 / 0.784235 / 0.0700404 / 0.0797012
- 80,000: 0.540846 / 0.531229 / 0.0471321 / 0.0536331
- 145,568: 0.166319 / 0.164056 / 0.0143295 / 0.016306
- 800,000: 0.00565315 / 0.00561871 / 4.61e-4 / 5.25e-4
- 8e6: 5.72e-5 / 5.71e-5 / 3.27e-6 / 3.73e-6
- 8e7 and 8e8: the calculator gives 3.55e-7 and 5.56e-10; the other three give 0 (`HEAD_K`).

The dense step now charges nothing with nothing stated (its rows may repeat), and `never_above` holds. Own-F columns and the left-out split are in note:training-20261112T1045Z-report-pricer-trn135-readings-by-k.

**Deleted:** nothing. `Layout`, its three rules and `Work.premises` stay, because the rule doesn't cover what they give. The gap is 524,288 MACs: dQ·Wq and dK·Wk in both layers, 131,072 each. Layer 2's read rope's unstated tables; layer 1's also go through the softmax's backward, which the rule doesn't reach.

**PR body:** 699 words, naming the note and giving the premise one line. The trace's reading, `SHARED_PREFIX` and the trace's figures are unchanged.

**Not done or not checked:**
- The tiny lf step's values (too many gates; the miniature has its graph).
- A bf16 step.
- A nested scan over a value input's rows, which `shared_rows` doesn't look at.
- Layout mode: it still takes a value batch's rows as distinct, and doesn't apply the softmax rule, so "layout alone" (0) is below "nothing stated".
- Step 4's "each k" figures are one run at the fixture's batch of zeros. The reading reads only the circuit, so the batch fixes only shapes.

