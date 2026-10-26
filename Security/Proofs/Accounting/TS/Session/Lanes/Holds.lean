import Proofs.Accounting.TS.Session.Lanes.Range

/-!
# Rows that hold a leaf's encodings decode to it

`decLane` and `decComb` read each field's bits at its port and first bit in the layout (`Lanes.Decode`). Where each
field's bits hold the encoding of the leaf's word there, and the words are in range, the decodes give the leaf back
(`Lanes.Range`):

* `LeafInRange`: the committed leaf's codes are s8, its scales' and y's encodings u32 and u16, its z s32;
* `LaneHolds`: lane `l`'s ports' rows hold its A rows (`laneA`) and B rows (`laneB`), the leaf's record, the public
  inputs' admit bits, salt, position and roots and its index `l`, each field at its place, and `ok = 1`; `CombHolds`:
  the combine's hold the receipt and `ok = 1`;
* `decLane_eq`, `decComb_eq`: then the decodes are `laneView` and the combine's view, at the partials rows they read
  (lane `l`'s and the combine's `p<l>` are one row, `Lanes.Partials`).
-/

namespace Accounting.TS.Lanes

open PouwWindow.Lanes Flock.SoundAtBridge Accounting.TS.Checked

/-- **An A or B row is in range**: s8 codes, and its scale's encoding a u32. -/
structure RowInRange {K : ℕ} (r : Row K) : Prop where
  codes : ∀ k, -2 ^ 7 ≤ r.codes k ∧ r.codes k < 2 ^ 7
  scale : 0 ≤ r.scale ∧ r.scale < 2 ^ 32

/-- **A leaf is in range**: its rows (`RowInRange`), its z s32 and its y's encodings u16. -/
structure LeafInRange {K : ℕ} (L : Leaf K) : Prop where
  a : ∀ r, RowInRange (L.a r)
  b : ∀ r, RowInRange (L.b r)
  z : ∀ i j, -2 ^ 31 ≤ L.z i j ∧ L.z i j < 2 ^ 31
  y : ∀ i j, 0 ≤ L.y i j ∧ L.y i j < 2 ^ 16

/-- **Lane `l`'s ports' rows hold its words**, each field's bits at its place in the layout (`decLane`'s): its A rows'
and B rows' codes (two's complement) and scales, the leaf's z (two's complement) and y cell by cell, the admit bits, the
salt, position and roots, the index `l`, and `ok = 1`. -/
structure LaneHolds (K : ℕ) (T : Tile) (rd : String → ℕ → Option Bool) (L : Leaf K) (P : LanePub) (l : Fin 32) :
    Prop where
  a : ∀ (s : Fin 4) (k : Fin K), Holds (rd s!"a{s.val}") (fieldBit (lanePorts T 0) s!"a[{s.val}].codes" + 8 * k.val) 8
    (encS 8 ((laneA L l s).codes k))
  aS : ∀ s : Fin 4, Holds (rd s!"a{s.val}") (fieldBit (lanePorts T 0) s!"a[{s.val}].scale") 32 (laneA L l s).scale.toNat
  b : ∀ (t : Fin 8) (k : Fin K), Holds (rd s!"b{t.val}") (fieldBit (lanePorts T 0) s!"b[{t.val}].codes" + 8 * k.val) 8
    (encS 8 ((laneB L l t).codes k))
  bS : ∀ t : Fin 8, Holds (rd s!"b{t.val}") (fieldBit (lanePorts T 0) s!"b[{t.val}].scale") 32 (laneB L l t).scale.toNat
  z : ∀ i j : Fin 32, Holds (rd "zy") (fieldBit (lanePorts T 0) "zy.z" + 32 * (i.val * T.c + j.val)) 32 (encS 32 (L.z i j))
  y : ∀ i j : Fin 32, Holds (rd "zy") (fieldBit (lanePorts T 0) "zy.y" + 16 * (i.val * T.c + j.val)) 16 (L.y i j).toNat
  admit : ∀ i : Fin 32, rd "admit" (fieldBit (lanePorts T 0) "admit" + i.val) = some (P.admit i)
  salt : Holds (rd "pub") (fieldBit (lanePorts T 0) "salt") digestBits P.salt
  position : Holds (rd "pub") (fieldBit (lanePorts T 0) "position") positionBits P.position
  roots : Holds (rd "pub") (fieldBit (lanePorts T 0) "roots") (2 * digestBits) P.roots
  lane : Holds (rd "pub") (fieldBit (lanePorts T 0) "lane") laneBits l.val
  ok : rd "o0" 0 = some true

/-- **The combine's ports' rows hold its words**: the receipt, and `ok = 1`. -/
structure CombHolds (nw : ℕ) (T : Tile) (rd : String → ℕ → Option Bool) (receipt : Fin nw → BitVec 32) : Prop where
  receipt : ∀ w : Fin nw, Holds (rd "receipt") (fieldBit (combinePorts T) "receipt" + 32 * w.val) 32 (receipt w).toNat
  ok : rd "o0" 0 = some true

/-- An in-range row read at its fields decodes to itself. -/
theorem row_eq {K : ℕ} {r : Row K} (hr : RowInRange r) {rd : ℕ → Option Bool} {lc ls : ℕ}
    (hc : ∀ k : Fin K, Holds rd (lc + 8 * k.val) 8 (encS 8 (r.codes k))) (hs : Holds rd ls 32 r.scale.toNat) :
    (⟨fun k => sgnBits 8 (bitsN rd (lc + 8 * k.val) 8), (bitsN rd ls 32 : ℤ)⟩ : Row K) = r := by
  obtain ⟨codes, scale⟩ := r
  rw [Row.mk.injEq]
  refine ⟨?_, ?_⟩
  · funext k
    have h := hr.codes k
    exact sgnBits_bitsN (by norm_num) (by norm_num at h ⊢; exact h.1) (by norm_num at h ⊢; exact h.2) (hc k)
  · exact bitsN_cast hr.scale.1 hr.scale.2 hs

/-- **Lane `l`'s rows decode to its view**: where its ports' rows hold its words (`LaneHolds`) of an in-range leaf,
`decLane` is `laneView` of the leaf, the public inputs, the partials row it reads and `l`. -/
theorem decLane_eq {K nw : ℕ} {T : Tile} {rd : String → ℕ → Option Bool} {L : Leaf K} {P : LanePub}
    {l : Fin 32} (hR : LeafInRange L) (h : LaneHolds K T rd L P l) :
    decLane K nw T rd = laneView L P (decLane K nw T rd).part l := by
  unfold decLane laneView
  rw [LaneView.mk.injEq]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · funext s
    exact row_eq (hR.a _) (h.a s) (h.aS s)
  · funext t
    exact row_eq (hR.b _) (h.b t) (h.bS t)
  · funext i j
    have hz := hR.z i j
    exact sgnBits_bitsN (by norm_num) (by norm_num at hz ⊢; exact hz.1) (by norm_num at hz ⊢; exact hz.2) (h.z i j)
  · funext i j
    exact bitsN_cast (hR.y i j).1 (hR.y i j).2 (h.y i j)
  · rfl
  · obtain ⟨admit, salt, position, roots, hs, hp, hr⟩ := P
    rw [LanePub.mk.injEq]
    refine ⟨?_, ?_, ?_, ?_⟩
    · funext i
      rw [h.admit i]
      dsimp only
      cases admit i <;> rfl
    · exact bitsN_eq h.salt hs
    · exact bitsN_eq h.position hp
    · exact bitsN_eq h.roots hr
  · have hl : l.val < 2 ^ laneBits := by
      have := l.isLt
      simp only [laneBits]
      omega
    apply Fin.ext
    simp only [bitsN_eq h.lane hl]
    exact Nat.mod_eq_of_lt l.isLt
  · simp [h.ok]

/-- **The combine's rows decode to its view**: where its ports' rows hold the receipt and `ok = 1` (`CombHolds`),
`decComb` is the partials rows it reads, the receipt and `ok = 1`. -/
theorem decComb_eq {nw : ℕ} {T : Tile} {rd : String → ℕ → Option Bool} {receipt : Fin nw → BitVec 32}
    (h : CombHolds nw T rd receipt) : decComb nw T rd = ⟨(decComb nw T rd).parts, receipt, true⟩ := by
  unfold decComb
  rw [CombineView.mk.injEq]
  refine ⟨?_, ?_, ?_⟩
  · rfl
  · funext w
    exact ofNat_bitsN (h.receipt w)
  · simp [h.ok]

end Accounting.TS.Lanes
