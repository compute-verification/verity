[2026-11-12T15:46Z] 93bfa2f6cd55
PRICER, TRN-139 round 3: done. #9315 head `e6a3ad556b` (draft, unmarked), pushed before the runs and one deep on #9170 `1068f3c5c2`, which is untouched.

One thing to rule on. Your rule as written also drops #9170's own carried_loop: x = u ⊙ g with g θ′ is a multiplier an input reaches, so x lost the bit. That failed test_a_scan_charges_its_carried_weights_from_their_second_iteration and two bound tests, and your finding-3 no-cast probe charged 0. So a product's operand that it reads as one row (no batch maps it, its contraction's K words) is marked distinct at the site. One row can't repeat its rows, so this takes nothing new as a premise. With it, the three tests pass again and the no-cast probe is back to 48. No lf figure moved.

What each finding became:
1. Closed. `Status.fixed` means every word is one value in every member of every enclosing batch. It holds for an input's words and those computed from them alone, never for a parameter a batch maps; a carry and θ's words are never fixed (the smaller bound). The multiplier clause reads it, so the four forms of W times a multi-word input (batch, calls, composite-word, composite-row) all charge 0, in one test.
2. Closed. A multiply (an INJECTIVE op that is not a shift) keeps the bit only where each other argument is reached by no input (θ's own, or stated and passing the constant test), or is one word fixed as above. A word computed from inputs counts, as ruled. rows + M ⊙ (z ⊙ g) charges 0, and so does M ⊙ g (from the third iteration: in the first, g is θ's). One test; the one-word fact stays (W·s 48, rows + M·s 96).
3. Closed. ROUNDINGS = {F32ToBf16Rn, F32ToF16Rn, F32ToE4m3Sat, F32ToE2m1Sat}, the registry's float-to-narrower-float roundings, keep the bit. The premise says θ's rows differ at every width a product or a rounding reads them. Test: W cast, and W and x cast, each charged (T−1)·K·N.

The premise (docstring and README):
- a weight is θ's rows, which differ for generic weights at every width read, and a word no input reaches is nonzero;
- cancels are taken as changes: W + z + (−1)·z, and new, W ⊙ (v + (−1)·v) with v registered, as a fact;
- one fixed word (an input's or computed from inputs) is taken as nonzero;
- two arrays.

bf16 attention (RAN locally, report only). It reads the same at #9170's head and at e6a3ad556b.
- f32: N = 96 (O1 Wo^T, weight (False, True)); left out: 64 as two activations, 3,072 gathered, 808 theta.
- Every operand cast: N = 0, gathered 3,232. O1 Wo^T's 96 and O1 O2^T's 64 move to gathered.
- P cast only, or O1 and O2 cast only: the same as every operand cast.
- Only the weights and the looked-up rows cast: as f32.
So yes, specific is lost at F32ToBf16Rn (not in INJECTIVE) on an activation, in #9170's rule.

Runs (RAN at e6a3ad556b):
- r20261112-153023-107a, vy-nebius-cpu-9:
  - `ratchet gate 5c0ae25d6c28..e6a3ad556b9d: 1 change(s), 0 failed`;
  - `never_above ok`, and `charged 6160384` with left_out {theta 32768, gathered 1935360, activations 2916352} at every k from 65,536 to 8e8, as in round 6;
  - `4 suites: 4 passed (0 cached), 0 failed; 29.1 min of test time`.
- r20261112-153056-9496, vy-nebius-cpu-8: last line `REVERT_DONE`. Each of 16 conditions reverted fails a test. Round 2's nine fail as before; the new ones:
  - a multiplier an input reaches (back to not unknown): rows_plus[input-times-weight] and [weight];
  - one word some enclosing batch reads apart (fixed not read): [composite-word] and [composite-row];
  - a mapped parameter left fixed: the same two;
  - the one-word multiplier removed, an input not fixed, or computed not keeping fixed: test_what_the_carried_words_premise_leaves_held_as_facts;
  - ROUNDINGS emptied: test_a_weight_rounded_to_bf16_before_its_product_is_charged;
  - the one-row operand removed: test_a_scan_charges_its_carried_weights_from_their_second_iteration, test_the_bound_is_the_calculators_at_the_groups_beta, test_every_operand_is_at_most_bf16_and_beta_k and the bf16 test.

RAN locally:
- the lf per-group dump is byte-identical to 9ff0d233a1's;
- test_value_reading.py and test_program_reading.py, slow included: 15 passed, and the three breaks still charge pairs read twice;
- test_work.py: 44 passed.

Not checked:
- other Programs, and no bf16 lf step;
- F16, E4m3 and E2m1 roundings are in no test, and their widenings back (F16ToF32, E4m3ToF32) don't keep the bit;
- a carried one-word multiplier (never fixed: the smaller bound);
- f(b, b) aliasing;
- an fma with a stated-zero product;
- circuit-check (no circuit changes).

