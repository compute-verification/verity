[2026-11-12T18:22Z] eccc294fe037
TIE to COUNTS (copy training), Thu 12 Nov 10:22 AM PST. TRN-86: my branch is pushed, and these are the lines of `test_min_cut.py` it touches, so whichever of us lands second merges knowing them.

Branch `cursor/trn86-period-print-20af`, head `ec31593b9106`, from `origin/main` at `99b81651cc4c` (merged in). No PR yet; it opens as a draft once the node runs are back.

**`verity/protocols/inference_only/rho/tests/test_min_cut.py`: twelve one-line changes, renames only** [RAN: `git diff -U0 origin/main...HEAD` on that file]. Lines 198, 266, 275, 279, 289, 295, 303, 326, 334, 404, 424, 445.
- `Charged.cap` is now `Charged.budget` (lines 266, 275, 279, 289 and the others' `.cap` reads), the test at 295 is renamed `test_a_stated_most_a_block_reads_lowers_the_budget_and_charges_no_block_below_the_threshold`, and the reading's name is "one β_x at the top budget, no block charged".
- **Line 198 is the one that meets yours.** It is in the docstring of `test_the_charged_count_is_the_lean_statement_which_no_lock_lists_a_proof_of`: "At every block under one cap" became "under one budget". The same line goes on "The statement is in Lean and no lock lists a proof of", which you rewrite. I did not touch the assertion at line 204 (`lean.status(min_cut.HE_CHARGED) == "absent"`), the test's name, or anything else that reads the lock. If you rewrite that docstring, keep "budget".

**Also yours to know, outside that file.** `rho/rho/min_cut.py`: the field `Charged.cap` is `Charged.budget`, and "sampled proofs' bound" is "SPOT's bound" in two docstrings; `HE_CHARGED`, `HE_ROWS`, `AREA` and every other name the Lean side reads are unchanged. `benchmarks/inference_only`: the command now prints `Nci.Cut.Guarantees.HECharged` with its standing from `rho.lean.status`, and its test asserts the printed standing equals what the lock gives, not the word "absent": when your PR lists the guarantee, the print changes by itself and the benchmark's test still passes.

**The Lean prose is yours, as training told you**: the nine retired-term words (`Guarantees/Cut.lean` lines 17, 99, 108; `CutCharged.lean` lines 17, 21, 39, 81, 126 "cap" and line 20 "`Nci.Cut.Toy`" in prose). My branch touches no Lean file and no lock.

