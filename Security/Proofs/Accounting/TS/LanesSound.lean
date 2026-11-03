import Proofs.Accounting.TS.Session.Lanes.Guarantees
import Proofs.Accounting.TS.GroupSound

/-!
# A leaf's 33 linked sessions realize one unit (the proof)

`Accounting.SecurityProofs.TS.LeafLanesSound` is the group argument (`Accounting.TS.Group.group_soundAt`) at option A's
33 statements (`Option (Fin 32)`) and their layer `leafLayer`, where `Accounting.Assumptions.TS.GroupRealizes` is proved
(`Accounting.TS.Lanes.groupRealizes_lanes32`) rather than assumed. The lanes cover the leaf
(`Accounting.TS.Lanes.cover`: cell `(i, j)` is lane `4 (i mod 8) + ⌊(j mod 8)/2⌋`'s at row slot `⌊i/8⌋` and column slot
`2⌊j/8⌋ + (j mod 2)`), so a leaf whose 33 statements each hold at their committed rows is served right
(`unitServed_of`): `LeafChecks`, which the circuits computing their Definitions give (`leafChecks_of_computes`), gives
each lane's `LaneOk` at a `K` the lanes bind at, which the leaf verifier's tile check gives (`k_of_check`), and the
combine's `CombineOk`, and each lane's view agrees with the leaf on what `LaneOk` reads (`LaneAgrees`). The leaf is the
one the lanes' committed rows decode to and the receipt the combine's (`Accounting.TS.Lanes.leafOf`, `receiptOf`): each
A row as lane `4 (r mod 8)` decodes it, each B row as lane `⌊(j mod 8)/2⌋` does, each cell of the record as the one
lane whose cell it is. The four lanes that read one A row (or one B row) read one registered tree at one row in full, so
wherever the sessions' registrants' rows agree they decode it alike (`read_bits`, `decodes_of_checks`); the known
ports' fields follow from the known-read check (`laneKnownHolds_of_check`, `combKnownOk_of_check`); the partials record
is the combine's decoded one: lane `l`'s `partials` and the combine's `p<l>` are `ownOf`'s ports of one table at the
leaf's row `l`, placed from one base row for the unit (`place`), read in full (`partials_bits`). Each statement's known
content (`knownOf`) is the `okKnown` form the group layer reads, its other known rows then its `o0` row
(`knownOf_okKnown`). The union bound over the 33 sessions gives `33 δ`.

This module imports `Proofs.Flock.SoundAtBridge` (through `Proofs.Accounting.TS.GroupSound`), whose imports make `X`
a token, so `Proofs/Accounting.lean` imports it.
-/

namespace Accounting.TS.Lanes

open Accounting.TS.Checked Flock.SoundAtBridge

/-- **The 32 lanes cover the leaf**: cell `(i, j)` is lane `4 (i mod 8) + ⌊(j mod 8)/2⌋`'s at row slot `⌊i/8⌋` and
column slot `2⌊j/8⌋ + (j mod 2)`. -/
theorem cover (i j : Fin 32) : ∃ l s t, rowOf l s = i ∧ colOf l t = j := by
  have hi := i.isLt
  have hj := j.isLt
  refine ⟨⟨4 * (i.val % 8) + (j.val % 8) / 2, by omega⟩, ⟨i.val / 8, by omega⟩, ⟨2 * (j.val / 8) + j.val % 2, by omega⟩,
    Fin.ext ?_, Fin.ext ?_⟩
  · show (4 * (i.val % 8) + (j.val % 8) / 2) / 4 + 8 * (i.val / 8) = i.val
    omega
  · show 2 * ((4 * (i.val % 8) + (j.val % 8) / 2) % 4) + 8 * ((2 * (j.val / 8) + j.val % 2) / 2) +
      (2 * (j.val / 8) + j.val % 2) % 2 = j.val
    omega

/-- **The leaf verifier's tile check gives a `K` the lanes bind at**: where `PouwWindow.Lanes.Tile.check` accepts a
tile, its `K` is whole 32-wide steps and below `2^17`, where the lanes' s8 accumulator is exact. -/
theorem k_of_check {T : PouwWindow.Lanes.Tile} (h : T.check = .ok ()) : 32 ∣ T.k ∧ T.k < 2 ^ 17 := by
  unfold PouwWindow.Lanes.Tile.check at h
  by_cases h1 : T.k % 32 = 0
  · refine ⟨Nat.dvd_of_mod_eq_zero h1, Nat.lt_of_not_le fun h2 => ?_⟩
    have h2' : 131072 ≤ T.k := h2
    have h3 : ¬T.k < 64 := by omega
    simp [h1, h2', h3] at h
    cases h
  · simp [h1] at h
    cases h

/-- **A leaf whose 33 statements hold at their committed rows is served right**: at a `K` the lanes bind at, where
each session's committed rows are the port rows of a wire assignment its evaluation agrees with at every unit, each
lane's decode agrees with the leaf `L` on what `LaneOk` reads (`LaneAgrees`) and the combine's decode is the partials
record `parts`, the receipt and `ok = 1`, `LeafChecks` gives every lane's `LaneOk` and the combine's `CombineOk`, and
the lanes cover the leaf. -/
theorem unitServed_of {K nw : ℕ} {Pub : Type} {dq : List ℤ → ℤ}
    {part : Pub → Fin 32 → (Fin 4 → Row K) → (Fin 8 → Row K) → Fin nw → BitVec 32}
    {sess : Option (Fin 32) → RowsSession} {decL : RowsSession → (ℕ → ℕ → Option Bool) → LaneView K nw Pub}
    {decC : RowsSession → (ℕ → ℕ → Option Bool) → CombineView nw} {pub : Pub}
    {parts : Fin 32 → Fin nw → BitVec 32} {receipt : Fin nw → BitVec 32} {L : Leaf K}
    (hC : Accounting.Assumptions.TS.LeafChecks dq part sess decL decC) (hK32 : 32 ∣ K) (hK : K < 2 ^ 17)
    (hL : ∀ l, LaneAgrees (decL (sess (some l)) (openedReg (sess (some l)))) L pub (parts l) l)
    (hCb : decC (sess none) (openedReg (sess none)) = ⟨parts, receipt, true⟩)
    (hV : ∀ k, ∃ vals : Fin (sess k).Circ.N → Bool,
      openedReg (sess k) = portRead (sess k) vals ∧ ∀ u, ∀ g ∈ (sess k).P.io u, (sess k).Circ.eval vals g = vals g) :
    UnitServed dq part pub receipt L := by
  have hlane : ∀ l, LaneOk dq part (laneView L pub (parts l) l) := by
    intro l
    obtain ⟨vals, hv, he⟩ := hV (some l)
    have hd : LaneAgrees (decL (sess (some l)) (portRead (sess (some l)) vals)) L pub (parts l) l := by
      rw [← hv]; exact hL l
    exact hd.laneOk (hC.1 hK32 hK l vals he hd.ok)
  have hcomb : CombineOk (⟨parts, receipt, true⟩ : CombineView nw) := by
    obtain ⟨vals, hv, he⟩ := hV none
    have hd : decC (sess none) (portRead (sess none) vals) = ⟨parts, receipt, true⟩ := by
      rw [← hv]; exact hCb
    have h := hC.2 vals he ((congrArg CombineView.ok hd).trans rfl)
    rw [hd] at h
    exact h
  refine ⟨fun i j => ?_, ?_⟩
  · obtain ⟨l, s, t, rfl, rfl⟩ := cover i j
    exact (hlane l).2 s t
  · have hp : parts = fun l => part pub l (laneA L l) (laneB L l) := funext fun l => (hlane l).1
    exact hcomb.symm.trans (congrArg xorAll hp)

end Accounting.TS.Lanes

open Accounting.TS Accounting.TS.Checked Accounting.TS.Lanes Flock.SoundAtBridge PouwWindow.Lanes

/-- **The decodes read the committed leaf, from acceptance**: where each session has a unit, its served roots
registered, whose units read their one-row ports at their rows (`ReadsOwnRow`), whose registrants' rows agree at one
tree and position (`RowsAgree`) and whose known rows hold, at every proof whose sessions pass flock-verify's complete
and known-read checks at the known content of the public inputs `pub` (`knownOf`) and the ports read in full
(`readFull`), and the leaf verifier's check of their files at `ownOf` (`roots`, the unit's placement `place`,
`checkOwn`), each lane's decode agrees
with the leaf the lanes' committed rows decode to (`leafOf`) at the combine's decoded partials, and the combine's
decode is those partials, its decoded receipt (`receiptOf`) and `ok = 1`. Lane `l`'s A row at slot `s` is lane
`4⌊l/4⌋`'s at that slot and its B row at slot `t` lane `l mod 4`'s, one row of one tree each lane reads in full
(`read_bits`); its cells are its own (`cellLane_rowOf_colOf`, `laneAgrees_of`); its partials row is the combine's
(`partials_bits`, `part_eq`); its known fields are the checks' (`laneKnownHolds_of_check`, `combKnownOk_of_check`). -/
theorem Accounting.TS.Lanes.decodes_of_checks {K nw : ℕ} {sess : Option (Fin 32) → RowsSession}
    {salts : Option (Fin 32) → String → ByteArray} {pub : LanePub} {roots : String → _root_.Flock.Registered.Port}
    {place : String → String × ℕ}
    (hc : ∀ k, Flock.Registered.checkComplete (sess k).c (sess k).own = .ok () ∧
      Flock.KnownRead.check (sess k).c (knownOf pub (salts k) k ++
        Flock.KnownRead.fullRows (sess k).own (readFull (servedTile K) (k.map Fin.val))) = .ok () ∧
      checkOwn (ownOf roots place (servedTile K) (k.map Fin.val)) (sess k).own = .ok ())
    (hS : ∀ k u, (sess k).ServedRegistered u)
    (hn : ∀ k, 0 < (sess k).n) (hRo : ∀ k, ReadsOwnRow (sess k))
    (hA : RowsAgree sess) (hK : ∀ k, KnownHeld (sess k) (knownOf pub (salts k) k)) :
    ∃ parts : Fin 32 → Fin nw → BitVec 32,
      (∀ l, LaneAgrees (decL K nw (servedTile K) (sess (some l)) (openedReg (sess (some l))))
        (leafOf K (servedTile K) sess) pub (parts l) l) ∧
      decC nw (servedTile K) (sess none) (openedReg (sess none)) =
        ⟨parts, receiptOf nw (servedTile K) sess, true⟩ := by
  have hP : ∀ l : Fin 32, (decL K nw (servedTile K) (sess (some l)) (openedReg (sess (some l)))).part =
      (decC nw (servedTile K) (sess none) (openedReg (sess none))).parts l := fun l =>
    part_eq (partials_bits (T := servedTile K) (roots := roots) (place := place) (l := l.val)
      (by rw [servedTile_lanes]; exact l.isLt) (checkOwn_ok (hc (some l)).2.2) (checkOwn_ok (hc none).2.2)
      (hc (some l)).1 (hc none).1 (hc (some l)).2.1 (hc none).2.1 (uL := ⟨0, hn (some l)⟩) rfl (hS (some l) _)
      (uC := ⟨0, hn none⟩) rfl (hS none _) (hRo (some l)) (hRo none)
      (hA (some l) none ⟨0, hn (some l)⟩ ⟨0, hn none⟩))
  have hbits : ∀ (l l' : Fin 32) (nm nm' v : String) (row : ℕ), (nm, v, row) ∈ regReads (servedTile K) (some l.val) →
      (nm', v, row) ∈ regReads (servedTile K) (some l'.val) → fullValue v = true →
      portBit (sess (some l)) (openedReg (sess (some l))) nm =
        portBit (sess (some l')) (openedReg (sess (some l'))) nm' :=
    fun l l' _ _ _ _ hm hm' hv => funext fun b =>
      read_bits (roots := roots) (place := place) hm hm' (full_of hm hv) (full_of hm' hv) (checkOwn_ok (hc (some l)).2.2)
        (checkOwn_ok (hc (some l')).2.2) (hc (some l)).1 (hc (some l')).1 (hc (some l)).2.1 (hc (some l')).2.1
        (uL := ⟨0, hn (some l)⟩) rfl (hS (some l) _) (uC := ⟨0, hn (some l')⟩) rfl
        (hS (some l') _) (hRo (some l)) (hRo (some l'))
        (hA (some l) (some l') ⟨0, hn (some l)⟩ ⟨0, hn (some l')⟩) b
  refine ⟨(decC nw (servedTile K) (sess none) (openedReg (sess none))).parts, fun l => ?_, ?_⟩
  · refine laneAgrees_of sess l (fun s => ?_) (fun t => ?_) (hP l)
      (laneKnownHolds_of_check (s := sess (some l)) (servedTile K) (hK (some l))
        (u := ⟨0, hn (some l)⟩) rfl (hS (some l) _) (hc (some l)).1 (hc (some l)).2.1)
    · have hm' := a_read K (aHome l) s
      rw [rowOf_aHome] at hm'
      exact hbits l (aHome l) _ _ _ _ (a_read K l s) hm' fullValue_a
    · have hm' := b_read K (bHome l) t
      rw [colOf_bHome] at hm'
      exact hbits l (bHome l) _ _ _ _ (b_read K l t) hm' fullValue_b
  · have ho := combKnownOk_of_check (s := sess none) (hK none) (u := ⟨0, hn none⟩) rfl
      (hS none _) (hc none).1 (hc none).2.1
    refine CombineView.eta_ok _ ?_
    unfold decC
    rw [decComb_ok, ho]
    rfl

open Accounting.TS Accounting.TS.Checked Accounting.TS.Lanes Flock.SoundAtBridge PouwWindow.Lanes in
/-- **`GroupRealizes` at option A's 33 statements, from the circuits' lowerings** (the honest witness of
`accounting-ts-group-realizes` at the 32 × 32 leaf, one chunk a lane): where the lanes' and the combine's circuits
compute their pieces' compositions (`LaneLowers`, `CombineLowers`), the 33 statements realize the unit's served checks
(`UnitServed`) of the leaf their committed rows decode to (`leafOf`), at the known content of the public inputs `pub`
(`knownOf`), the ports read in full (`readFull`) and the leaf verifier's file check at `ownOf`, past its tile check:
`decodes_of_checks` gives the decodes' agreement with the leaf, `k_of_check` a `K` the lanes bind at, and
`unitServed_of` the unit's checks from `LeafChecks` (`leafChecks_of_computes`). -/
theorem Accounting.TS.Lanes.groupRealizes_lanes32 {K nw : ℕ} (sess : Option (Fin 32) → RowsSession)
    (salts : Option (Fin 32) → String → ByteArray) (pub : LanePub) (roots : String → _root_.Flock.Registered.Port)
    (place : String → String × ℕ) (hL : LaneLowers (servedTile K) (laneChain K nw) Pouw.H1Zy.zyDequant sess)
    (hC : CombineLowers sess (decC nw (servedTile K))) :
    Accounting.Assumptions.TS.GroupRealizes sess (fun k => knownOf pub (salts k) k)
      (fun k => readFull (servedTile K) (k.map Fin.val)) (fun k => ownOf roots place (servedTile K) (k.map Fin.val))
      (PouwWindow.Lanes.Tile.check ⟨32, 32, K⟩ = .ok ()) (leafOf K (servedTile K) sess)
      (UnitServed (dqOf Pouw.H1Zy.zyDequant) (partOf (laneChain K nw)) pub (receiptOf nw (servedTile K) sess)) := by
  intro hT hc hS hn hRo hA hK hV
  obtain ⟨parts, hLa, hCb⟩ := decodes_of_checks (K := K) (nw := nw) (roots := roots) (place := place) hc hS hn hRo hA hK
  have hKr := k_of_check hT
  exact unitServed_of (leafChecks_of_computes (servedTile K) (laneChain K nw) Pouw.H1Zy.zyDequant sess
    (laneComputes_of_lowers hL) (combineComputes_of_lowers hC)) hKr.1 hKr.2 hLa hCb hV

/-- **Option A's known content is the group layer's form**: statement `k`'s known rows at the public inputs `P`
(`knownOf`, `PouwWindow.Lanes.knownRows`) are its other known rows (lane `l`'s `admit` and `pub`, the combine's none)
followed by its `o0` row 1 under its salt (`Accounting.TS.Group.okKnown`). -/
theorem Accounting.TS.Lanes.knownOf_okKnown (P : LanePub) (salts : String → ByteArray) (k : Option (Fin 32)) :
    ∃ pre, knownOf P salts k = Accounting.TS.Group.okKnown pre (salts "o0") := by
  cases k with
  | none => exact ⟨#[], rfl⟩
  | some l =>
    exact ⟨#[("admit", #[(PouwWindow.Lanes.admitRow (List.ofFn P.admit), salts "admit")]),
      ("pub", #[(PouwWindow.Lanes.pubRow P.salt P.position P.roots l.val, salts "pub")])], rfl⟩

open Accounting.TS.Checked Accounting.TS.Lanes in
/-- **A leaf's 33 linked C-Flock sessions realize one unit** (`Accounting.Guarantees.TS.LeafLanesSound`):
`Accounting.TS.Group.group_soundAt` at `leafLayer` and its 33 statements, at the known content of the path's public
inputs (`knownOf`, which names `o0`), the ports read in full at the layout (`PouwWindow.Lanes.readFull`), the leaf
verifier's check of each file at `ownOf` (`roots`, `place`) and of the tile (`PouwWindow.Lanes.Tile.check`, its group
check), the served roots registered by the reads built from `own` and the one-read check
(`RowsSession.servedRegistered_canon`), and `GroupRealizes` from the 33 circuits' lowerings (`groupRealizes_lanes32`). -/
theorem Accounting.SecurityProofs.TS.LeafLanesSound : Accounting.Guarantees.TS.LeafLanesSound := by
  intro Ω _ S nπ K nw sess₀ op sess out salts pub roots place right δ hCR hLaw hδ hδ0 hL hC
  have h := Accounting.TS.Group.group_soundAt
    (leafLayer sess out (fun x w i k => knownOf (pub x w i) (salts x w i k) k)
      (fun _ _ _ k => PouwWindow.Lanes.readFull (servedTile K) (k.map Fin.val))
      (fun x w i k => PouwWindow.Lanes.ownOf (roots x w i) (place x w i) (servedTile K) (k.map Fin.val))
      (dqOf Pouw.H1Zy.zyDequant) (partOf (laneChain K nw)) pub
      (fun x w i => receiptOf nw (servedTile K) (sess x w i)) (fun x w i => leafOf K (servedTile K) (sess x w i))
      right)
    sess out (fun x w i k => knownOf (pub x w i) (salts x w i k) k)
    (fun _ _ _ k => PouwWindow.Lanes.readFull (servedTile K) (k.map Fin.val))
    (fun x w i k => PouwWindow.Lanes.ownOf (roots x w i) (place x w i) (servedTile K) (k.map Fin.val))
    (fun _ _ _ => PouwWindow.Lanes.Tile.check ⟨32, 32, K⟩ = .ok ()) "o0" δ hCR hLaw
    (fun x w i k h => (sess₀ x w i k).servedRegistered_canon (op x w i k) (PouwWindow.Lanes.checkOneRead_ok h))
    (fun x w i k => ⟨#[(PouwWindow.Lanes.okRow, salts x w i k "o0")], by
      cases k <;> simp [knownOf, PouwWindow.Lanes.knownRows]⟩) hδ hδ0
    (fun x w i hacc => ⟨(hacc none).2.2.2.2.2.2.2, fun k => ⟨(hacc k).1, (hacc k).2.1, (hacc k).2.2.1,
      (hacc k).2.2.2.1, (hacc k).2.2.2.2.1, (hacc k).2.2.2.2.2.1, (hacc k).2.2.2.2.2.2.1⟩⟩)
    (fun x w i hg hc hS hn hRo hA hK hV => Or.inl (groupRealizes_lanes32 (sess x w i) (salts x w i) (pub x w i)
      (roots x w i) (place x w i) (hL x w i) (hC x w i) hg hc hS hn hRo hA hK hV))
    (fun _ _ _ hc => Or.inr (Or.inl hc)) (fun _ _ _ hc => Or.inr (Or.inr hc))
  have e : (Fintype.card (Option (Fin 32)) : ℝ) = 33 := by norm_num [Fintype.card_option]
  rw [e] at h
  exact h
