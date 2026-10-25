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
* **The parts agree on a row through its commit** (`rowsAgree_of_commits`): where every holder of row `r` records an
  evaluation of the frame of what it holds (with a salt) to one digest `dig r`, and every held row has a holding owner,
  the parts agree on the shared rows, or two of the recorded calls collide (`findColl (L.allCalls x ws)`, `DecompB`'s
  own escape). Two different rows can't commit to one digest otherwise, since the frame is injective in the row.

That every holder's commit output is one digest per row is the registered-row binding: each part's hm96 slot outputs
its row's commit string (`hmSlotsCommit_of_full`), which its public file names, and the V\* setup opens that digest at
the row's registered root and position (`Registered.opens`). Two holders with different digests climb from different
leaves to one root, a collision among the setup's own evaluations at the two parts' statements: `Br`'s search set
(`setupCalls` over the run's statements), not `findColl (L.allCalls x ws)`.
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

/-- **The parts agree on the shared rows through their commits, or two recorded calls collide**: where every part
holding row `r` records an evaluation of `frame` at what it holds and some salt with output `dig r`, `frame` is injective
in the row, and every held row's owner holds it, either the parts agree on the shared rows or two of the calls the
parts record collide (`DecompB`'s escape). -/
theorem LayoutB.rowsAgree_of_commits {H : Msg → Msg} {L : LayoutB} {x : Msg} {ws : List Claims.Flock.Wit}
    (hT : ∀ e ∈ L.allCalls x ws, H e.1 = e.2) (frame : Msg → Msg → Msg)
    (hinj : ∀ v σ v' σ', frame v σ = frame v' σ' → v = v') (dig : ℕ → Msg)
    (hC : ∀ r s w v, ws[s]? = some w → L.rowHeld x s w r = some v → ∃ σ, (frame v σ, dig r) ∈ L.calls x s w)
    (hO : ∀ r s w v, ws[s]? = some w → L.rowHeld x s w r = some v →
      ∃ o w' v', L.rowOwner x r = some o ∧ ws[o]? = some w' ∧ L.rowHeld x o w' r = some v') :
    L.RowsAgree x ws ∨ (Claims.findColl (L.allCalls x ws)).isSome := by
  by_cases hc : (Claims.findColl (L.allCalls x ws)).isSome
  · exact Or.inr hc
  refine Or.inl fun r s w v hw hv => ?_
  obtain ⟨o, w', v', ho, hw', hv'⟩ := hO r s w v hw hv
  rw [LayoutB.row_of_owner ho hw' hv']
  obtain ⟨σ, hσ⟩ := hC r s w v hw hv
  obtain ⟨σ', hσ'⟩ := hC r o w' v' hw' hv'
  by_contra hne
  have h1 := LayoutB.mem_allCalls hw hσ
  have h2 := LayoutB.mem_allCalls hw' hσ'
  have e1 : H (frame v σ) = dig r := hT _ h1
  have e2 : H (frame v' σ') = dig r := hT _ h2
  exact hc (Claims.findColl_isSome_mem H _ hT _ _ (List.mem_map.2 ⟨_, h1, rfl⟩) (List.mem_map.2 ⟨_, h2, rfl⟩)
    ⟨fun e => hne (hinj _ _ _ _ e), e1.trans e2.symm⟩)

end Claims.Recursion.VStarB
