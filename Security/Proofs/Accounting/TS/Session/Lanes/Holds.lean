import Proofs.Accounting.TS.Session.Lanes.Range

/-!
# `LeafReads`' decode half: rows that hold a leaf's encodings decode to it

`decLane` and `decComb` read each field's bits at its port and first bit in the layout (`Lanes.Decode`). Where each
field's bits hold the encoding of the leaf's word there, and the words are in range, the decodes give the leaf back
(`Lanes.Range`):

* `LeafInRange`: the committed leaf's codes are s8, its scales' and y's encodings u32 and u16, its z s32;
* `LaneHolds`: lane `l`'s ports' rows hold its A rows (`laneA`) and B rows (`laneB`), the leaf's record, its partials
  row, the public inputs' admit bits, salt, position and roots and its index `l`, each field at its place, and `ok = 1`;
  `CombHolds`: the combine's hold the 32 partials rows, the receipt and `ok = 1`;
* `decLane_eq`, `decComb_eq`: then the decodes are `laneView` and the combine's view;
* `leafReads_of_holds`: so `LeafReads` at `decL` and `decC` follows from those at the registered reads (`openedReg`).

What `leafReads_of_holds` still takes is of the rows, not the decodes: that each port's opened row holds the committed
leaf's and partials' encodings (registration: `ServedRegistered`, `P-opened`, `checkComplete`), that the `admit`, `pub`
and `ok` rows hold their known content, and that the committed leaf is in range.
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
and B rows' codes (two's complement) and scales, the leaf's z (two's complement) and y cell by cell, its partials
row's words, the admit bits, the salt, position and roots, the index `l`, and `ok = 1`. -/
structure LaneHolds (K nw : ℕ) (T : Tile) (rd : String → ℕ → Option Bool) (L : Leaf K) (P : LanePub)
    (p : Fin nw → BitVec 32) (l : Fin 32) : Prop where
  a : ∀ (s : Fin 4) (k : Fin K), Holds (rd s!"a{s.val}") (fieldBit (lanePorts T 0) s!"a[{s.val}].codes" + 8 * k.val) 8
    (encS 8 ((laneA L l s).codes k))
  aS : ∀ s : Fin 4, Holds (rd s!"a{s.val}") (fieldBit (lanePorts T 0) s!"a[{s.val}].scale") 32 (laneA L l s).scale.toNat
  b : ∀ (t : Fin 8) (k : Fin K), Holds (rd s!"b{t.val}") (fieldBit (lanePorts T 0) s!"b[{t.val}].codes" + 8 * k.val) 8
    (encS 8 ((laneB L l t).codes k))
  bS : ∀ t : Fin 8, Holds (rd s!"b{t.val}") (fieldBit (lanePorts T 0) s!"b[{t.val}].scale") 32 (laneB L l t).scale.toNat
  z : ∀ i j : Fin 32, Holds (rd "zy") (fieldBit (lanePorts T 0) "zy.z" + 32 * (i.val * T.c + j.val)) 32 (encS 32 (L.z i j))
  y : ∀ i j : Fin 32, Holds (rd "zy") (fieldBit (lanePorts T 0) "zy.y" + 16 * (i.val * T.c + j.val)) 16 (L.y i j).toNat
  part : ∀ w : Fin nw, Holds (rd "partials") (fieldBit (lanePorts T 0) "partials" + 32 * w.val) 32 (p w).toNat
  admit : ∀ i : Fin 32, rd "admit" (fieldBit (lanePorts T 0) "admit" + i.val) = some (P.admit i)
  salt : P.salt < 2 ^ digestBits ∧ Holds (rd "pub") (fieldBit (lanePorts T 0) "salt") digestBits P.salt
  position : P.position < 2 ^ positionBits ∧ Holds (rd "pub") (fieldBit (lanePorts T 0) "position") positionBits P.position
  roots : P.roots < 2 ^ (2 * digestBits) ∧ Holds (rd "pub") (fieldBit (lanePorts T 0) "roots") (2 * digestBits) P.roots
  lane : Holds (rd "pub") (fieldBit (lanePorts T 0) "lane") laneBits l.val
  ok : rd "o0" 0 = some true

/-- **The combine's ports' rows hold its words**: row `x` of the partials at `p<x>`, the receipt, and `ok = 1`. -/
structure CombHolds (nw : ℕ) (T : Tile) (rd : String → ℕ → Option Bool) (parts : Fin 32 → Fin nw → BitVec 32)
    (receipt : Fin nw → BitVec 32) : Prop where
  p : ∀ (x : Fin 32) (w : Fin nw),
    Holds (rd s!"p{x.val}") (fieldBit (combinePorts T) s!"partials[{x.val}]" + 32 * w.val) 32 (parts x w).toNat
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
`decLane` is `laneView` of the leaf, the public inputs, its partials row and `l`. -/
theorem decLane_eq {K nw : ℕ} {T : Tile} {rd : String → ℕ → Option Bool} {L : Leaf K} {P : LanePub}
    {p : Fin nw → BitVec 32} {l : Fin 32} (hR : LeafInRange L) (h : LaneHolds K nw T rd L P p l) :
    decLane K nw T rd = laneView L P p l := by
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
  · funext w
    exact ofNat_bitsN (h.part w)
  · obtain ⟨admit, salt, position, roots⟩ := P
    rw [LanePub.mk.injEq]
    refine ⟨?_, ?_, ?_, ?_⟩
    · funext i
      rw [h.admit i]
      dsimp only
      cases admit i <;> rfl
    · exact bitsN_eq h.salt.2 h.salt.1
    · exact bitsN_eq h.position.2 h.position.1
    · exact bitsN_eq h.roots.2 h.roots.1
  · have hl : l.val < 2 ^ laneBits := by
      have := l.isLt
      simp only [laneBits]
      omega
    apply Fin.ext
    simp only [bitsN_eq h.lane hl]
    exact Nat.mod_eq_of_lt l.isLt
  · simp [h.ok]

/-- **The combine's rows decode to its view**: where its ports' rows hold the partials rows, the receipt and `ok = 1`
(`CombHolds`), `decComb` is that view. -/
theorem decComb_eq {nw : ℕ} {T : Tile} {rd : String → ℕ → Option Bool} {parts : Fin 32 → Fin nw → BitVec 32}
    {receipt : Fin nw → BitVec 32} (h : CombHolds nw T rd parts receipt) :
    decComb nw T rd = ⟨parts, receipt, true⟩ := by
  unfold decComb
  rw [CombineView.mk.injEq]
  refine ⟨?_, ?_, ?_⟩
  · funext x w
    exact ofNat_bitsN (h.p x w)
  · funext w
    exact ofNat_bitsN (h.receipt w)
  · simp [h.ok]

/-- **`LeafReads` from the rows**: where every proof's committed leaf is in range and, where its sessions' registrants'
rows agree, each lane's opened port bits hold its words (`LaneHolds`) and the combine's hold the partials and the receipt
(`CombHolds`), `LeafReads` holds at `decL` and `decC`. -/
theorem leafReads_of_holds {Ω : Type} {S nπ K nw : ℕ} (T : Tile)
    {sess : Pouw.RelativeSecurity.Paths Ω S → Fin S → Fin nπ → Option (Fin 32) → RowsSession}
    {pub : Pouw.RelativeSecurity.Paths Ω S → Fin S → Fin nπ → LanePub}
    {parts : Pouw.RelativeSecurity.Paths Ω S → Fin S → Fin nπ → Fin 32 → Fin nw → BitVec 32}
    {receipt : Pouw.RelativeSecurity.Paths Ω S → Fin S → Fin nπ → Fin nw → BitVec 32}
    {leaf : Pouw.RelativeSecurity.Paths Ω S → Fin S → Fin nπ → Leaf K}
    (hR : ∀ x w i, LeafInRange (leaf x w i))
    (h : ∀ x w i, RowsAgree (sess x w i) →
      (∀ l, LaneHolds K nw T (portBit (sess x w i (some l)) (openedReg (sess x w i (some l)))) (leaf x w i)
        (pub x w i) (parts x w i l) l) ∧
      CombHolds nw T (portBit (sess x w i none) (openedReg (sess x w i none))) (parts x w i) (receipt x w i)) :
    Accounting.Assumptions.TS.LeafReads sess (decL K nw T) (decC nw T) pub parts receipt leaf :=
  (leafReads_iff T sess pub parts receipt leaf).2 fun x w i hA =>
    ⟨fun l => decLane_eq (hR x w i) ((h x w i hA).1 l), decComb_eq (h x w i hA).2⟩

end Accounting.TS.Lanes
