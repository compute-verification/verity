[2026-11-11T16:59Z] 8830e97d510b
WIRED to COUNTS and training (copy FORM), TRN-116, Wed 11 Nov 9:00 AM PST. P3 `fresh_rows : FreshRows`: MY READING. TRUE AS STATED, I TAKE IT; no hypothesis missing.

READ at `cursor/training-record-counts2-a8e2` e4589f4e8b: `RecordFreshRows.lean`, `RecordKinds.lean`, `RecordTally.lean`, `RecordSlots.lean`, `RecordBlockSize.lean`, `RecordGrid.lean`, `RecordWire.lean`, `RecordBlocks.lean` (the `gridsOf_*` and `tiles_*` statements), `tmpls_named`, and the check's `summarize`, `tally`, `kindOf`, `inputsOf`, `rowWords`, `Wire.indexOf`, `disjoint`, `pairwiseDisjoint`. `RecordSameSlot.lean`: not read, P3 does not call it. Nothing below is compiled: INFERRED from that reading.

WHY IT IS TRUE, the four places it could have failed and does not:
1. A row's bits against the summary's inputs. A unit's port of `words` words cut `k` ways has `k` rows of `16 * words / k` bits: `k * (16 * words / k) ≤ 16 * words`, for every `k`, 0 included. So the rows of slot `s` are at most `inputs` bits, a template having `t.units / c.S` units in the slot (`slot_units`).
2. A declaration's bits against its rows' `rowBits`. A later declaration: `inputsOf_decls`' last clause is `rowBits`' own two `find?`s, an equality. A wire's: `16 * rowWords`, and `rowWords` reads `t.rows.lookup port`, the first row of that name as `find?` does, so `16 * (words / k) ≤ 16 * words / k`. `text.cutsOf = c.W.cutsOf` since `cutsOf` reads `cuts` alone.
3. A declaration's rows inside the slot's rows. A unit with a slot is in a grid of its template, which lies in the template's units (`gridsOf_ok`, `tiles_ordered`); the cut's rows are below `cutsOf` (a wire: `shape_spec`; a later one: `inputsOf_decls`). The wire's bound on its units (`boundsOk`) is not needed, so nothing of `piece` or `templates_of_read`.
4. A fresh row is under no declaration of kind inside or theta: `fill_of_cover` gives it that declaration's fill, so its kind. And no row is under two declarations (`pairwiseDisjoint`).
The subtraction in `fresh` truncates; it does not matter, the bound is proved as a sum (`Σ_F rowBits + tally[inside] + tally[theta] ≤ inputs`) before `fresh_ge`.

WHAT IT CALLS: `read_front`, `finish_text`, `front_text`, `tmpls_named`, my `inputsOf_decls` and `wired_of_read`, `wire_shape`, `shape_spec`, `fill_of_cover`, `gridsOf_ok`, `gridsOf_period`, `gridsOf_lookup`, `tiles_ordered`, `tiles_sum`, `mem_of_lookup`, `eq_of_nodup_map`, `slot_units`, `inSlot_count`, `tally_eq`, `fresh_ge`, `rowBits_eq`, `find?_one`. Not `cover_apart`: two covers are apart straight from `Spot.Wires.disjoint`.

THE ROUTE: yours, with two changes. (a) The universe is one list, templates, then their units of the slot, then rows, then the cut's indices, summed by one lemma for a nested `flatMap` (`((l.flatMap f).map b).sum ≤ (l.map g).sum` from a bound on each part), so no `Finset` product for it. (b) A declaration's fit is one private lemma for both sorts: for `d ∈ c.x.decls` with `d.bits ≠ 0`, the two `find?`s give a template and a row with `d.bits ≤ 16 * row.2.1 / cutsOf` and `cut0 + cuts ≤ cutsOf`; then `rowBits` unfolds on it with no distinctness.

HOW BIG: one new file, `security/Proofs/Properties/Nci/RecordFresh.lean`, about 400 lines, on a new branch of mine `cursor/training-record-fresh-05f2` cut from your e4589f4e8b. Five parts: the abstract step (Mathlib only), the cover's count, the declaration's fit, the universe's sum, the assembly. The universe's sum and the assembly are where the node tries will go. I write it against a local mock by your recipe (thank you: my VM has the Mathlib oleans) and send each part's node try as it passes; I do not give an hour.

The statement compared with the ask by script when the file exists: `theorem fresh_rows : FreshRows`, the `def` yours and untouched.

Names compiled, mine: R1 to R5, I1 to I3, M1, M2, C1, C2. Names left, mine: `fresh_rows`.

