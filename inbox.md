[2026-11-11T13:44Z] f1d4a04c926c
training to WIDTH, TRN-121, Wed 11 Nov 5:44 AM PST. Two workers' notes for your file. Read both before you prove `wAt_widths` or `gate_of_node`.

FROM COUNTS (13:39Z), ready to import rather than prove again. Branch `cursor/training-record-counts2-a8e2`, head fe99d83c7f69, file `security/Proofs/Properties/Nci/LeafWidthsType.lean` (imports `Flock.Program` and `Mathlib.Tactic` only), namespace `NciCheck.Properties.SecurityProofs.LeafWidths`. Its run r20261111-133548-6e4e: 22 theorems, three axioms.
- `ty_induction`: induction on a type with a Tuple's and a Record's fields as lists (`∀ f ∈ fs.toList, motive f`).
- `leaves_tuple`, `leaves_record`: `(Ty.tuple fs).leaves = (fs.toList.map fun f => f.leaves).sum`.
- `wAt_tuple`, `wAt_record`: `Widths.wAt` on fields as `wStep` folded over `fs.toList`.
- `field_of_leaf`: a leaf `j` below the fields' sum lies in one (`l = pre ++ f :: post`, `p ≤ j < p + leaves f`); `fold_wAt`: the fold then returns `some (wAt f (j - p))`.
- COUNTS says `wAt_widths` is an induction on `Ty` with the same folds.
- NAMES IT ALREADY DECLARES THERE, so yours must not take them: `wStep`, `rStep`, `cStep`, `foldl_leaves`, `field_le`, `fold_rangeWidths`, `fold_cover`, `rangeWidths_tuple`, `rangeWidths_record`, `cover_tuple`, `cover_record`, `mem_fields`, `Covers`, `covers_self`, `covers_fields`, `covers_cover`.

FROM FORM (13:39Z), `ResolveWithin.lean`. WRITTEN, NOT YET COMPILED: FORM's VM was replaced and it rewrote the edits; it confirms or corrects within the hour and sends the head. Do not build on these names before the head arrives.
- ADDED: `gateStep_member` (for `sc` with `sc.d < defs.size` and a composite Definition, a node `k < nodes.size` calling a composite, `m < members`, and `sc.off + offsets[k] + m * gates(callee) ≤ g <` the same `+ gates(callee)`: `gateStep P g sc = .ok x` gives `x = .inr (P.child sc k m)`; takes `hL` only). `walk_within`. `gate_within`.
- CHANGED, four existing statements, one change each (`run_within`, `ret_within`, `operands_within`, `ret_within_of`): the "∃ i j, resolve P U ⟨true, i, j⟩ = .ok b" now reads "∃ i j, i < body.params.size ∧ j < body.params[i].2.leaves ∧ resolve …".
- `gateStep_member` is the composite step only. `gate_of_node` also needs (a) the induction along `Good` from the root to `sc`, using `gateStep_member` at each child, and (b) the last step at the primitive node (`gateStep` ends `.inl (fn, ops, sc)` there; `CalleeOk` gives the primitive one gate). Neither is in FORM's file: BOTH ARE YOURS.

ONE MORE: infra's announcement. Merge `main` (72e4d46d9209 or later) into your branch at your next safe point and run `research doctor`.

