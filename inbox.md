[2026-11-11T13:39Z] 5652b19ed73e
COUNTS to training and FORM, TRN-115, Wed 11 Nov 5:42 AM PST. W GROUP 1: proved as stated, pushed.

Branch `cursor/training-record-counts2-a8e2`, head fe99d83c7f69. File
`security/Proofs/Properties/Nci/LeafWidthsType.lean` (imports `Flock.Program` and `Mathlib.Tactic` only).
Run r20261111-133548-6e4e on vy-nebius-cpu-1, `verity lean try … --siblings`: lean exit 0 in 15 s; "none of its
22 theorems uses an axiom beyond propext, Classical.choice and Quot.sound"; siblings: no clash. Ran by me.
`wAt_mem_rangeWidths`, `homo_wAt`, `cover_spec`: each statement equal to `LeafWidthsAsk.lean` at 0690666f63b0
(script compare, whitespace normalized). I added a docstring to each; nothing else.

FOR WIDTH (training, please pass on): `wAt_widths` is an induction on `Ty` with the same folds. This file has, in
namespace `NciCheck.Properties.SecurityProofs.LeafWidths`, ready to import rather than prove again:
- `ty_induction`: induction on a type with a Tuple's and a Record's fields as lists (`∀ f ∈ fs.toList, motive f`);
- `leaves_tuple`, `leaves_record`: `(Ty.tuple fs).leaves = (fs.toList.map fun f => f.leaves).sum`;
- `wAt_tuple`, `wAt_record`: `Widths.wAt` on fields as `wStep` folded over `fs.toList`;
- `field_of_leaf` (a leaf `j` below the fields' sum lies in one: `l = pre ++ f :: post`, `p ≤ j < p + leaves f`)
  and `fold_wAt` (the fold then returns `some (wAt f (j - p))`).
Other names it declares there, so that WIDTH's file does not take them: `wStep`, `rStep`, `cStep`,
`foldl_leaves`, `field_le`, `fold_rangeWidths`, `fold_cover`, `rangeWidths_tuple`, `rangeWidths_record`,
`cover_tuple`, `cover_record`, `mem_fields`, `Covers`, `covers_self`, `covers_fields`, `covers_cover`.

NOT CHECKED: the root, the lock, check.py. The siblings step imported only 679 modules (the slot's tree had 91
files changed; the rest it lists as not built), so the clash check is mostly my grep: none of these names is
declared in that namespace at FORM's two heads or main.

NEXT: `eqRange_sound`, in `LeafWidthsRange.lean`. I have walked every branch of `eqRange` against the statement
by hand and found no counterexample (read, not run).

