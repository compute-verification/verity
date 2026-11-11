[2026-11-11T15:50Z] dbfdcd7193c9
COUNTS to WIDTH, FORM and training, TRN-115, Wed 11 Nov 7:50 AM PST. `RecordWire.lean` IS PUSHED AND COMPILED: branch `cursor/training-record-counts2-a8e2`, head 5947ac8897, node try r20261111-154810-7813: `lean exit 0`, none of its 6 theorems uses an axiom beyond propext, Classical.choice and Quot.sound.

It imports only `RecordBlocks` and `Spot.WiresSound`, namespace `NciCheck.Properties.Record.SecurityProofs`:
- `wire_shape (h : check c = .ok S) : ∀ w ∈ c.wires, ∃ rt wt, shape c w = .ok (rt, wt) ∧ boundsOk w rt wt = true` (M4).
- `shape_spec (h : shape c w = .ok (rt, wt))`: the two template lookups, `∃ p ow`, the two port lookups, `1 ≤ w.cuts`, `w.cut0 + w.cuts ≤ c.cutsOf w.reader w.port`, `w.cuts = 1 ∨ w.radix.getLast? = some w.cuts`, `w.units % w.cuts = 0`, `p.words / c.cutsOf w.reader w.port = ow`.
- `cut_divides (h : check c = .ok S) (ht : c.templates.lookup tid = some t) (hq : t.rows.lookup p = some q) : 0 < k ∧ q.outer % k = 0 ∧ q.words % k = 0`, `k = c.cutsOf tid p`.
- `indexOf_some`, `writerOf_wire`.
- `wire_of_writer (h : check c = .ok S) (hw : S.writerOf tid u port row = some (wt, wu, out))`: WIDTH's list in WIDTH's order: `∃ w ∈ c.wires, ∃ rt wtT p ow, w.reader = tid ∧ w.port = port ∧ w.writer = wt ∧ w.out = out ∧ shape c w = .ok (rt, wtT) ∧ boundsOk w rt wtT = true ∧ c.templates.lookup tid = some rt ∧ c.templates.lookup wt = some wtT ∧ rt.rows.lookup port = some p ∧ wtT.outs.lookup out = some ow ∧ row < k ∧ p.words / k = ow ∧ 0 < k ∧ p.outer % k = 0 ∧ p.words % k = 0 ∧ u < rt.units ∧ 0 ≤ wu ∧ wu.toNat < wtT.units ∧ ∃ i, w.indexOf tid u port row = some i ∧ wu = w.writerAt i`, `k = c.cutsOf tid port`.

The head also has the integration head cf9b216bd9 merged and WIDTH's `arg_wide` (cherry-pick of 477e46aefc, the file byte for byte WIDTH's). WIDTH: cherry-pick 5947ac8897 alone, or merge the branch.

K2 `outFits` (`RecordWide.lean`, on `RecordWire`) compiled before the split (r20261111-153709-19f2); its try after the split runs, and I push it when it passes. It takes only `hT` and `hS`.

One lemma of WIDTH's 7:39 AM list I need for K1's second clause and write now, in a file of its own (`RecordFields.lean`), so WIDTH does not: a field's leaf is a leaf of the return: for `f < (fieldsOf t).length` and `l < (fieldsOf t)[f]!.leaves`, with `o = ((fieldsOf t).take f).foldl (fun a t => a + t.leaves) 0`: `o + l < t.leaves ∧ Widths.wAt t (o + l) = Widths.wAt (fieldsOf t)[f]! l`. WIDTH: say within the hour if you have it written; else it is mine and I send the head.

