[2026-11-12T14:57Z] 43cb84cd09e8
PRICER, TRN-139 round 2: done. #9315 head `9ff0d233a1` (draft, unmarked), one deep on #9170 `1068f3c5c2`; #9170 untouched.

Ruling wanted, one item. The one-word multiplier is a new, explicit part of the premise, and lf needs it. AdamW's W' = p*decay + u: with the strict rule (a product by an unstated input loses distinct), lf charges 0 and leaves all 11,012,096 MACs out as gathered (RAN locally). So the rule keeps distinct where the other argument is one word every member reads (axes None, or a single word), taken as nonzero. W*s, s a scalar input, is charged; a held test fact. If you'd rather lf charge 0 than take that premise, say so.

lf: nothing moved. At every k, charged 6,160,384 and left_out theta 32,768, gathered 1,935,360, activations 2,916,352, as in round 6 (RAN).

The five:
- closed: 1 zero-init carry, 2 shared unregistered input, 3 bias by calls, 4 replicated weight (view and replicated kinds), 5 one-word register.
- also closed: batches, composite through a shared argument, assembled from two nodes, and a weight's row read twice.

Stated in the premise, each held by a fact in test_what_the_carried_words_premise_leaves_held_as_facts or the stated-zero test:
- the cancel case (W + z + -1*z);
- the one-word multiplier, taken as nonzero;
- two nodes computing two arrays from one array's words.

The premise paragraph says "a weight" means theta's rows, which differ for generic weights.

Runs (RAN at 9ff0d233a1):
- r20261112-144149-f1ff, vy-nebius-cpu-2:
  - `ratchet gate 7d296de0f3e9..9ff0d233a176: 1 change(s), 0 failed`;
  - `never_above ok`, figures as above;
  - `4 suites: 4 passed (0 cached), 0 failed; 26.1 min of test time`.
- r20261112-144215-fd4d, vy-nebius-cpu-7: each of 9 conditions reverted fails its test:
  - may_repeat reads distinct: 12 tests;
  - stated zero: test_an_update_by_a_stated_zero_leaves_a_weight_thetas;
  - first keeps the init: test_a_carry_from_a_stated_zero_plus_rows_is_left_out;
  - input not distinct: [shared];
  - composite through a shared argument: [composite];
  - argument from more than one node: [assembled];
  - argument reads a word twice: test_a_weights_row_read_twice_as_a_products_rows_is_left_out;
  - returned array assembled: [calls], [batches] and [view];
  - one-word multiplier: test_what_the_carried_words_premise_leaves_held_as_facts.
  The register test is held by two rules, so no single revert fails it. Last line `REVERT_DONE`.
- Locally, test_value_reading.py and test_program_reading.py, slow included: 15 passed. The three breaks still charge pairs read twice.

Not checked:
- Programs other than lf's tiny step and the by-hand loops;
- f(b, b) aliasing inside a composite;
- an fma whose product is a stated 0;
- circuit-check (no circuit changes).

Two slips:
- 9ff0d233a1 wasn't on the remote when the runs launched. The runs used --cwd source, so they tested that head; I pushed it just now.
- I printed one run's status.json with its command field (no secrets in it). From now on that field is filtered out.

