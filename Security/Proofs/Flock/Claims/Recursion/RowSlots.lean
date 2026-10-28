import Proofs.Flock.Claims.Recursion.VStarBound
import Proofs.Flock.Claims.Recursion.VStarLinks

/-!
# The bound layout's shared rows: held off a copy's columns, and agreeing through their commits

`LayoutB`'s row slots (`rowAt`, `rowOwner`) let a check read a registered row that more than one part holds, such as
the `acc0`/`acc1` rows a `rec-residuals` part shares with the `rec-open` parts, at one value, `L.row x ws r`, wherever
the parts agree on the shared rows (`LayoutB.RowsAgree`).

* **The part holds the rows it reads** (`rowHeld_of_colsHold`, and `held_of_colsHoldB` for a round held whole): where
  the layout reads row `r` at part `s` as a frame then a run of one copy's columns (`copyPs`), and those columns hold
  bytes `b`, the part holds the frame then `b`. A copy's input columns hold what its unit reads there
  (`inCols_readElems`, `openIns_reads`), so a row slot at a port's columns holds the row the copy's unit reads.
* **The parts agree on a row through its commit and its registered root** (`rowsAgree_of_commits`): where every
  holder of row `r` records an evaluation of the frame of what it holds (with a salt) to its digest, that digest climbs
  to the row's registered root through evaluations the verifier makes itself (`Brings.setup` at the V\* statements,
  `vcalls`), and every held row has a holding owner, the parts agree on the shared rows, or two of the parts' recorded
  calls and the verifier's evaluations collide (`findColl (L.allCalls x ws ++ vcalls H b xs)`). Two holders with
  different digests climb from different leaves to one root, a collision among the verifier's evaluations
  (`VStar.chain_coll`); two different rows under one digest collide among the recorded calls, since the frame is
  injective in the row.
* **The parts read the rows the messages carry** (`LayoutB.readsRows_of_agree`, in `VStarBound`): where the messages
  carry each placed row's owner's value (`LayoutB.RowsLaid`), agreement on the rows gives `LayoutB.ReadsRows`, which
  `PartChecksB`'s routes take.

The commit is the registered-row binding: each part's hm96 slot outputs its row's commit string
(`hmSlotsCommit_of_full`), which its public file names, and the verifier opens that digest at the row's registered root
and position (`Registered.opens`). The climb's evaluations are the verifier's own, which `findColl (L.allCalls x ws)`
doesn't search: the escape here is option A, which charges a collision with them, and until it
lands no route reaches a layout whose messages carry a shared row.
-/

namespace Claims.Recursion.VStarB

open Claims (Msg)

/-- Part `s`'s recorded calls are among every part's, where `ws` gives it the witness `w`. -/
theorem LayoutB.mem_allCalls {L : LayoutB} {x : Msg} {ws : List Claims.Flock.Wit} {s : ℕ} {w : Claims.Flock.Wit}
    (hw : ws[s]? = some w) {e : Msg × Msg} (he : e ∈ L.calls x s w) : e ∈ L.allCalls x ws := by
  unfold LayoutB.allCalls
  refine List.mem_flatten.2 ⟨L.calls x s w, List.mem_iff_getElem?.2 ⟨s, ?_⟩, he⟩
  rw [List.getElem?_mapIdx, hw]
  rfl

/-- One segment at hidden-value positions reads its frame, then the bytes at those positions. -/
theorem readSegs_one (w : Claims.Flock.Wit) (pre : Msg) (ps : List ℕ) :
    readSegs w (segsAt [(pre, ps)]) = pre ++ VStar.readAt w ps := by
  rw [readSegs_segsAt]
  simp [VStar.readSegs]

theorem readAt_copyPs {p : _root_.Flock.Readable.Parsed} {t : ℕ} {ht : t < p.sts.size} {w : Claims.Flock.Wit}
    {o g u base : ℕ} {b : ByteArray} (htop : base + 8 * b.size ≤ p.sts[t].c.unit.constPos + 1)
    (hb : FlockVBridge.ColsHold (_root_.Flock.Readable.copyOf p t ht w o g u) base b) :
    VStar.readAt w (VStar.copyPs p t ht o g u base (8 * b.size)) = b.data.toList := by
  apply VStar.readAt_of_bits
  · simp only [VStar.copyPs, List.length_map, List.length_range, Array.length_toList]
    rfl
  · intro q hq
    have hq' : q < 8 * b.size := by simpa [VStar.copyPs] using hq
    simp only [VStar.copyPs, List.getElem_map, List.getElem_range]
    rw [← VStar.copyOf_eq (by omega)]
    exact hb q hq'

/-- **A part holds a shared row as a run of a copy's columns holds it**: where the layout reads row `r` at part `s` as
the frame `pre` then the copy's columns from `base` (`copyPs`), within the copy's columns, and those columns hold the
bytes `b` (`ColsHold`), the part holds `pre ++ b` as row `r`. -/
theorem rowHeld_of_colsHold {L : LayoutB} {x : Msg} {s r : ℕ} {p : _root_.Flock.Readable.Parsed} {t : ℕ}
    {ht : t < p.sts.size} {w : Claims.Flock.Wit} {o g u base : ℕ} {pre : Msg} {b : ByteArray}
    (hm : L.rowAt x s r = some (segsAt [(pre, VStar.copyPs p t ht o g u base (8 * b.size))]))
    (htop : base + 8 * b.size ≤ p.sts[t].c.unit.constPos + 1)
    (hb : FlockVBridge.ColsHold (_root_.Flock.Readable.copyOf p t ht w o g u) base b) :
    L.rowHeld x s w r = some (pre ++ b.data.toList) := by
  rw [LayoutB.rowHeld, hm, Option.map_some, readSegs_one, readAt_copyPs htop hb]

/-- **A part holds a round as a run of a copy's columns holds it**, at the bound layout: `held_of_colsHold` at
`LayoutB`, where the statement doesn't fix round `i`. -/
theorem held_of_colsHoldB {L : LayoutB} {x : Msg} {s i : ℕ} {p : _root_.Flock.Readable.Parsed} {t : ℕ}
    {ht : t < p.sts.size} {w : Claims.Flock.Wit} {o g u base : ℕ} {pre : Msg} {b : ByteArray}
    (hfx : L.fixedAt x i = none)
    (hm : L.msgAt x s i = some (segsAt [(pre, VStar.copyPs p t ht o g u base (8 * b.size))]))
    (htop : base + 8 * b.size ≤ p.sts[t].c.unit.constPos + 1)
    (hb : FlockVBridge.ColsHold (_root_.Flock.Readable.copyOf p t ht w o g u) base b) :
    L.held x s w i = some (pre ++ b.data.toList) := by
  simp only [LayoutB.held, hfx, Option.isSome_none, Bool.false_eq_true, ↓reduceIte, hm, Option.map_some]
  rw [readSegs_one, readAt_copyPs htop hb]

/-- **The verifier's own SHA-512 evaluations at the V\* statements `xs`** (the vcalls ask's `L.vcalls`, option A):
each input `Brings.setup` lists at a statement (its registered reads' leaves and climbs, and the roots it recomputes
from public data), with its output. -/
def vcalls (H : Msg → Msg) (b : RecDeploy.Brings) (xs : List Msg) : List (Msg × Msg) :=
  xs.flatMap fun xO => (b.setup xO).map fun u => (u, H u)

theorem mem_vcalls {H : Msg → Msg} {b : RecDeploy.Brings} {xs : List Msg} {xO u : Msg} (hx : xO ∈ xs)
    (hu : u ∈ b.setup xO) : (u, H u) ∈ vcalls H b xs :=
  List.mem_flatMap.2 ⟨xO, hx, List.mem_map.2 ⟨u, hu, rfl⟩⟩

/-- The verifier's evaluations are true evaluations. -/
theorem vcalls_true {H : Msg → Msg} {b : RecDeploy.Brings} {xs : List Msg} : ∀ e ∈ vcalls H b xs, H e.1 = e.2 := by
  intro e he
  obtain ⟨xO, -, he⟩ := List.mem_flatMap.1 he
  obtain ⟨u, -, rfl⟩ := List.mem_map.1 he
  rfl

/-- **The parts agree on the shared rows through their commits and the rows' registered roots, or two evaluations
collide**: where every part `s` holding row `r` records an evaluation of `frame` at what it holds and some salt, with
output its digest `dig r s`, that digest climbs `path r s` to the row's registered root `root r` with each input of the
climb an evaluation the verifier makes at one of the V\* statements `xs` (`Brings.setup`), two holders' climbs keep
values apart (`VStar.StepsApart`), `frame` is injective in the row, and every held row's owner holds it, either the
parts agree on the shared rows or two of the parts' recorded calls and the verifier's evaluations collide
(`findColl (L.allCalls x ws ++ vcalls H b xs)`, the escape option A of the vcalls ask gives). -/
theorem LayoutB.rowsAgree_of_commits {H : Msg → Msg} {L : LayoutB} {x : Msg} {ws : List Claims.Flock.Wit}
    {b : RecDeploy.Brings} {xs : List Msg} (hT : ∀ e ∈ L.allCalls x ws, H e.1 = e.2) (frame : Msg → Msg → Msg)
    (hinj : ∀ v σ v' σ', frame v σ = frame v' σ' → v = v') (dig : ℕ → ℕ → Msg)
    (hC : ∀ r s w v, ws[s]? = some w → L.rowHeld x s w r = some v → ∃ σ, (frame v σ, dig r s) ∈ L.calls x s w)
    (path : ℕ → ℕ → List (Msg → Msg)) (root : ℕ → Msg)
    (hclimb : ∀ r s w v, ws[s]? = some w → L.rowHeld x s w r = some v →
      VStar.chainH H (path r s) (dig r s) = root r)
    (hin : ∀ r s w v, ws[s]? = some w → L.rowHeld x s w r = some v →
      ∀ u ∈ VStar.chainIns H (path r s) (dig r s), ∃ xO ∈ xs, u ∈ b.setup xO)
    (hapart : ∀ r s t, VStar.StepsApart (path r s) (path r t))
    (hO : ∀ r s w v, ws[s]? = some w → L.rowHeld x s w r = some v →
      ∃ o w' v', L.rowOwner x r = some o ∧ ws[o]? = some w' ∧ L.rowHeld x o w' r = some v') :
    L.RowsAgree x ws ∨ (Claims.findColl (L.allCalls x ws ++ vcalls H b xs)).isSome := by
  have hT' : ∀ e ∈ L.allCalls x ws ++ vcalls H b xs, H e.1 = e.2 := by
    intro e he
    rcases List.mem_append.1 he with he | he
    · exact hT e he
    · exact vcalls_true e he
  by_cases hc : (Claims.findColl (L.allCalls x ws ++ vcalls H b xs)).isSome
  · exact Or.inr hc
  have hcoll : ∀ u u', u ∈ (L.allCalls x ws ++ vcalls H b xs).map Prod.fst →
      u' ∈ (L.allCalls x ws ++ vcalls H b xs).map Prod.fst → u ≠ u' → H u = H u' → False :=
    fun u u' hu hu' hne hH => hc (Claims.findColl_isSome_mem H _ hT' u u' hu hu' ⟨hne, hH⟩)
  have hset : ∀ y, (∃ xO ∈ xs, y ∈ b.setup xO) → y ∈ (L.allCalls x ws ++ vcalls H b xs).map Prod.fst := by
    rintro y ⟨xO, hx, hy⟩
    exact List.mem_map.2 ⟨(y, H y), List.mem_append_right _ (mem_vcalls hx hy), rfl⟩
  refine Or.inl fun r s w v hw hv => ?_
  obtain ⟨o, w', v', ho, hw', hv'⟩ := hO r s w v hw hv
  rw [LayoutB.row_of_owner ho hw' hv']
  obtain ⟨σ, hσ⟩ := hC r s w v hw hv
  obtain ⟨σ', hσ'⟩ := hC r o w' v' hw' hv'
  have hd : dig r s = dig r o := by
    by_contra hne
    obtain ⟨u, hu, u', hu', hne', hH⟩ := VStar.chain_coll H (hapart r s o) hne
      ((hclimb r s w v hw hv).trans (hclimb r o w' v' hw' hv').symm)
    exact hcoll u u' (hset _ (hin r s w v hw hv u hu)) (hset _ (hin r o w' v' hw' hv' u' hu')) hne' hH
  by_contra hne
  have h1 := LayoutB.mem_allCalls hw hσ
  have h2 := LayoutB.mem_allCalls hw' hσ'
  have e1 : H (frame v σ) = dig r s := hT _ h1
  have e2 : H (frame v' σ') = dig r o := hT _ h2
  exact hcoll _ _ (List.mem_map.2 ⟨_, List.mem_append_left _ h1, rfl⟩)
    (List.mem_map.2 ⟨_, List.mem_append_left _ h2, rfl⟩) (fun e => hne (hinj _ _ _ _ e))
    (e1.trans (hd.trans e2.symm))

end Claims.Recursion.VStarB
