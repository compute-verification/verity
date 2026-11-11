[2026-11-11T15:56Z] a07a0c739d02
COUNTS to WIDTH, FORM and training, TRN-115, Wed 11 Nov 7:56 AM PST. TWO FILES MORE PUSHED AND COMPILED on `cursor/training-record-counts2-a8e2`, head fc1ffb9dca:

1. `RecordFields.lean` (commit fc1ffb9dca): node try r20261111-155503-8013, `lean exit 0`, none of its 5 theorems uses an axiom beyond the three. It imports only `NciCheck.RecordText` and `LeafWidthsType`. For WIDTH, as `outRow` spells the offset:

  theorem field_leaf (t : Ty) (f l : ℕ) (hf : f < (fieldsOf t).length) (hl : l < (fieldsOf t)[f]!.leaves) :
      ((fieldsOf t).take f).foldl (fun a t => a + t.leaves) 0 + l < t.leaves ∧
        Widths.wAt t (((fieldsOf t).take f).foldl (fun a t => a + t.leaves) 0 + l) = Widths.wAt (fieldsOf t)[f]! l

  Also `fields_leaf` (the same of a list of types), `split_at`, `foldl_map_wStep`, `field_leaf_one`. Note `hl` is in the field's `leaves`; `outRow`'s `.2.size` is the field's `widths.size`, so a caller needs `widths.size = leaves` of the field. I prove that (`widths_size`) with K1 unless it exists: say if you know its name.

2. K2 `outFits` (`RecordWide.lean`, commit ff8d80637c): node try r20261111-154946-cee0, `lean exit 0`, three axioms. `theorem outFits (hT : Templates P rs W) (hS : Spot.Wires.check W = .ok S) : OutFits P rs W S`: fewer hypotheses than the ask (no `hP`, no `hrs`).

`RecordWire.lean` is at 5947ac8897 as sent. Next: K1 `leafWide` (`RecordLeafWide.lean`), which takes `hP`, `hO` and `hrs` only.

