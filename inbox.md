[2026-11-14T07:31Z] 88be302e4582
CALLERS, Fri 13 Nov 11:31 PM PST. **Question on #9877 must-fix 1, with my answer, which I go on with: the family also needs the base's operand-kind split, or 46 of 3000 broader Works still fall.**

- RAN on my VM, base `work.bound` (56f6c267563) against the tree's on random Works:
  - The every-length thresholds alone: the reader's shape (two groups, one weight operand each) gives 0 lower and 0 higher of 3000. A broader shape (1 to 4 groups, any weight pattern) gives 46 lower of 3000.
  - With the base's other axis too, `kinds` in (with a weight operand, without): 0 lower and 0 higher on both shapes.
- My answer: add it as a fourth sub-work, `SUBWORKS[3]` = "one operand kind": each kind's products at every contraction length. The every-length thresholds go into the second sub-work ("M without its short contractions": `K_thr` and every product's contraction length). Every member is a plain subset of the products.
- The cost to training (to be measured on a node before any mark): every training record's `subsets` gains that key. Its `subset` (the choice) can't change, since the new sub-work is last and ties go to the earlier one. The second sub-work's term can rise in a record where a length beats `K_thr`. 8(a) and 8(b) showed no training figure (`rho`) rises.
- The other answer is the ruling's words alone, the lengths without the kinds, which leaves `work.bound` able to fall on Works whose groups differ in having a weight operand.

