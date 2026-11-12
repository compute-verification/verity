Lane proofs
Goal proofs/sampled-audits-sound

**What this is.** #9416 (merged) added six sites to `benchmarks/one_stage/tests/test_a5_case_v2.py` that the
retired-terms lint counts (`tests/lint/test_repo_retired_terms.py`). That lint is off the merge gate, so nothing
failed and the count rose. This fixes the six forward and changes nothing else.

**The six.**

- Five spell the challenger's family as a string literal: the fixture's segment label, the two expected `challenger`
  parameters, and the two calls of `a5.case_coins`. a5 itself passes its own constant there (`a5_coins.FAMILY`,
  `a5.py`'s two calls of `case_coins`), so the tests now read the constant: a test that follows the constant checks
  the same thing and holds no spelling of its own.
- One docstring names SPOT by its older name; it says SPOT.

**The list.** The file's entry in `tests/lint/allowlists/repo_retired_terms.json` falls from 11 to 6, the sites the
file held before #9416 and holds now. Only this file's entry is touched.

**Checks, on the VM (one file each).**

- `pytest benchmarks/one_stage/tests/test_a5_case_v2.py`: 20 passed.
- The lint's own count for the file: 6 found, 6 allowed (12 found on `main`); `test_no_new_retired_terms` no longer
  names the file.
- `ruff check` on the file: clean.

**Not in this change.** The six sites the file held before #9416 (the evidence's `draw` key as a5 writes it, three
times; a test's name; a string; a docstring) and the tree's other files.
