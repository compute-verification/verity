[2026-11-11T17:23Z] dabeb19e75c9
WIRED to COUNTS, training and FORM, TRN-116, Wed 11 Nov 5:23 PM UTC. P3 COMPILES: `fresh_rows : FreshRows`, pushed.

Branch `cursor/training-record-fresh-05f2`, head 28bcb235fc (pushed): COUNTS' 7fd3617700 plus one commit, one new file `security/Proofs/Properties/Nci/RecordFresh.lean` (348 lines), no existing file edited. It merges into COUNTS' 62e842b48d with no conflict (RAN: `git merge-tree --write-tree`, rc 0).

RAN, run r20261111-171235-ecff on vy-nebius-cpu-1, `lean_changed.py --try security/Proofs/Properties/Nci/RecordFresh.lean --siblings` at 28bcb235fc:
- "lean exit 0 in 41 s"; imports built ("Build completed successfully (10847 jobs)"); no error; no warning names the file.
- "axiom probe: none of its 14 theorems uses an axiom beyond propext, Classical.choice and Quot.sound".
- siblings: "no built module of its package that doesn't import Proofs.Properties.Nci.RecordFresh declares a name it does (58 imported in 9 s, peak 4072 MB); 3844 unchecked". THAT CHECK IS THIN on this slot: 58 modules, where my run on FORM's file had 1754. So I also RAN `git grep` of my eleven public names over every `.lean` file on 28bcb235fc, COUNTS' 62e842b48d, FORM's 73dbb102d3 and b5a083b55e: no other file declares or uses one. The four private ones cannot clash.
- Statement: `theorem fresh_rows : FreshRows` in `NciCheck.Record.SecurityProofs`; RAN by script: the line is exactly that, and `RecordFreshRows.lean` is the same blob (ff4fd9b40a) at COUNTS' e4589f4e8b and at my head. No `sorry`, `axiom` or `native_decide` in the file.
- Retired-terms sites in the file: 0 (RAN: `test_retired_terms.offenders` returns `([], set())`).

COUNTS, your one line: `import Proofs.Properties.Nci.RecordFresh` and `form_counts : Counts form := counts_of_fresh fresh_rows` (`fresh_rows` is `NciCheck.Record.SecurityProofs.fresh_rows`). NOT CHECKED by me: that line compiled; it is yours.

The file imports `RecordBlockGates` (WIDTH's `tmpl_grids`, `tmpl_slot_units`), `RecordFreshRows`, `RecordKinds`, `RecordSlots`, `RecordTally`. Public, all in `NciCheck.Record.SecurityProofs`: defs `slotRows`, `slotList`; theorems `slotRows_mem`, `slotRows_card`, `slotRows_apart`, `slotList_mem`, `slotList_bits`, `slot_in_tmpl`, `rowWords_row`, `decl_fits`, `fresh_rows`. Private: `sum_toFinset_le`, `sum_flatMap_le`, `sum_two`, `covers_le`, `find?_of_lookup`.

Names compiled on a node, mine: `laysOut_exists`, R1 to R5, I1 to I3, M1, M2, C1, C2, and now P3 `fresh_rows` with its eight lemmas. Names left, mine: none. Job 1 (FORM's `form_laidOut`, run r20261111-165152-44e5) and job 2 are both done.

What I would take next, in this order: (1) a node try of the umbrella once COUNTS' `form_counts` and FORM's `form_laidOut` are on one head, on a slot with a full build so the siblings check covers the package; (2) any lemma FORM's or COUNTS' assembly finds missing.

