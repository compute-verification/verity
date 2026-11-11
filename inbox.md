[2026-11-11T13:26Z] 368a6f7bcdc8
COUNTS to training and FORM, TRN-115, Wed 11 Nov 5:33 AM PST. ONE PROOF: done and pushed.

Branch `cursor/training-record-counts2-a8e2`, head 5c971e36aa7cc289f65aaf92d578321e4df52bcd.
Run r20261111-132325-a5ee on vy-nebius-cpu-4, `verity lean try RecordLanding.lean --siblings`: lean exit 0 in 29 s;
"none of its 23 theorems uses an axiom beyond propext, Classical.choice and Quot.sound"; siblings: no clash (2945
modules imported). Ran by me.

WHAT CHANGED. `RecordLanding.lean` now holds `LeafHolds`, `leafHolds_of_leafWide`, `width_of_gate`,
`outLeaf_land_le` (takes `LeafHolds`), `landing_le_of_holds`, and
`landing_le := by exact landing_le_of_holds … (leafHolds_of_leafWide hW) …`.
`landing_le`'s docstring and statement: string compare against d0333c11f8 from `/-- **The landing bound**` to
` := by`: EQUAL (765 characters). `RecordLeafHolds.lean` is deleted, so the repeated proofs and the unnamed
`example` are gone: every theorem compiled is in the probe. `rowLeaf_width` and `outLeaf_width` are gone too
(nothing on FORM's heads or main read them; grep).

FOR FORM'S MERGE: your head has RecordLanding.lean from d0333c11f8 and no RecordLeafHolds.lean, so this is one
changed file. No name of RecordLanding.lean that your tree reads has moved: `landing_le` only.

NOT CHECKED: the root, the lock, check.py; the file on FORM's head (tried at 9d60cc0fb6 plus my commits); the
siblings step listed 895 modules "not built since their sources or imports changed" as unchecked, so a clash with
one of those would not show (the names added, `LeafHolds`, `leafHolds_of_leafWide`, `width_of_gate`,
`landing_le_of_holds`, are in no other file of `security/` or `verity/` at FORM's two heads: grep).

NEXT: W groups 1 to 4, in a file of mine beside FORM's two. Groups 5 and 6 are WIDTH's: I have nothing on them.

