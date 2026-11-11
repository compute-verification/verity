[2026-11-11T14:23Z] 3c8a8ecde72a
COUNTS to training (copy to FORM), TRN-115, Wed 11 Nov 6:25 AM PST. Answer to your 6:16 AM message.

STATE: `git status` clean; local = origin = `cursor/training-record-counts2-a8e2` at `f8634d2849971b21df455fe20ba3b6b1f683d4c4`. Nothing was unpushed.

W GROUPS 1 TO 4: ALL EIGHT STATEMENTS ARE PROVED AS FORM STATED THEM. Compared again just now by script against `LeafWidthsAsk.lean` at `0690666f63b0` (each theorem's text from `theorem` to `:=`, whitespace normalised): 8 of 8 equal. No hypothesis added, none found false or short. No `sorry`, `axiom` or `native_decide` in any file.

1. `LeafWidthsType.lean`: `wAt_mem_rangeWidths`, `homo_wAt`, `cover_spec`. Commit `fe99d83c7f`. Run `r20261111-133548-6e4e`: lean exit 0, 22 theorems, only propext, Classical.choice, Quot.sound; siblings no clash.
2. `LeafWidthsRange.lean`: `eqRange_sound`. Commit `a27cf24d08`. Run `r20261111-134639-68fb`: lean exit 0 in 62 s, 14 theorems, only the three axioms. ITS SIBLINGS STEP FAILED, in the slot's own stub (`unknown module prefix 'Commitments'`, before `main` was merged), so that run exited 1. Rerun at the head with `--siblings`: `r20261111-141917-31ec`, running now; I send its verdict when it ends.
3. `LeafWidthsPart.lean`: `residues_mem`, `partOk_sound`. Commit `9081c01eb8`. Run `r20261111-135819-633b`: lean exit 0 in 197 s, 12 theorems, only the three axioms; siblings no clash. It built Type and Range as its imports, after the merge.
4. `LeafWidthsBody.lean`: `seqBad_sound`, `bodyBad_sound`. Commit `f8634d2849`. Run `r20261111-140321-0107`: lean exit 0 in 31 s, 4 theorems, only the three axioms; siblings no clash.

Also on the branch: `LeafWidths.lean`, FORM's file unchanged (commit `a0221bd547`; `git diff` against `cursor/record-leafwide-statements-3b1c` is empty). `LeafWidthsAsk.lean` is not on my branch. `main` (`72e4d46d9209`) is merged at `c83f91cd96`.

WHAT I WAS IN THE MIDDLE OF at 14:15Z: no proof. After the 14:05Z push I owed this message for groups 3 and 4 (I cannot tell from here whether one went out before the error; this one replaces it), the Linear comment for 3 and 4, a note, and the rerun of Range's siblings step. The VM was reset, so the rerun is the one above.

NOT CHECKED: the package root, the lock, `check.py`; my files on FORM's head; a siblings step reads only the modules built in its slot (803 for Part and Body, 679 for Type), so FORM's whole-tree merge is the first real clash check; Range's siblings step has not passed yet.

ESTIMATE for groups 1 to 4: done, nothing left. NEXT: your reading on a unit that is one primitive operation (both questions), sent today well before 6 PM PST.

