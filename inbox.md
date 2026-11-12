[2026-11-12T19:01Z] aff26f6956a2
training to grade, Thu 12 Nov 11:01 AM PST. One more question for your 1 PM PST report. It is a decision, so I want your reading, not a change.

**At a `cc` the step now keeps the card's fused attention, forward and backward (TRN-144, STEP's line `cursor/trn144-step-fused-attention-2556`), and the grader refuses the hand-over.** STEP's findings, RAN on its VM at its head with the conftest change (`229dfd4d96`):

- `test_grader.py::test_the_products_are_the_steps_matrix_products[bf16-cublas]` and `::test_the_gpu_steps_linears_are_its_tensor_cores_dots` fail with `grader.Refused: the step has 519 occurrences and the structure Program 559` (`grader._same_step`).
- The hand-over's structure Program "may be derived without the step's `cc`", and the conftest derives it so (batch (3, 5)). The step's graph holds two fused ops a layer where the structure's holds the math path.
- Past that check the grader reads only `mm` and `bmm` occurrences as products, so a fused attention would drop out of N, O and the product rows; the tests assert attention as 32-bit `bmm` rows.
- A lane's structure at another batch shape has no captured attention shape (four, all T = 1,024), so at a `cc` it can be derived neither with the `cc` (the rule refuses the shape) nor without it (`_same_step` refuses) [INFERRED by STEP].

**Say in your report:** (1) what the structure is for and whether it may hold the math path where the step holds the fused op, with the correspondence the grader would check (one fused forward and one fused backward against which math occurrences); (2) what N, O and the product rows should read for a fused attention: its seven products at 16 bits as #9374 prices them, or something else; (3) the smallest change to `grader.py` and its tests that does it, as a list of functions, not code; (4) what it does to a published row. If two readings are both defensible, give both and your default. I decide on it and a new worker builds it.

