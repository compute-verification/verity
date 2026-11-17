module

public import Proofs.Flock.Claims.Recursion.VStarBound
public import Proofs.Flock.Claims.Recursion.VStarLinks
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.Recursion.Transport

/-! The bound layout's shared rows agree through their commits up to a collision with the verifier's own evaluations, which `findColl (L.allCalls x ws)` doesn't search; no route reaches a layout whose messages carry a shared row. -/

@[expose] public section

namespace Claims.Recursion.VStarB

open Claims (Msg)

theorem LayoutB.mem_allCalls {L : LayoutB} {x : Msg} {ws : List Claims.Flock.Wit} {s : ℕ} {w : Claims.Flock.Wit}
    (hw : ws[s]? = some w) {e : Msg × Msg} (he : e ∈ L.calls x s w) : e ∈ L.allCalls x ws := by
  unfold LayoutB.allCalls
  refine List.mem_flatten.2 ⟨L.calls x s w, List.mem_iff_getElem?.2 ⟨s, ?_⟩, he⟩
  rw [List.getElem?_mapIdx, hw]
  rfl

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

theorem rowHeld_of_colsHold {L : LayoutB} {x : Msg} {s r : ℕ} {p : _root_.Flock.Readable.Parsed} {t : ℕ}
    {ht : t < p.sts.size} {w : Claims.Flock.Wit} {o g u base : ℕ} {pre : Msg} {b : ByteArray}
    (hm : L.rowAt x s r = some (segsAt [(pre, VStar.copyPs p t ht o g u base (8 * b.size))]))
    (htop : base + 8 * b.size ≤ p.sts[t].c.unit.constPos + 1)
    (hb : FlockVBridge.ColsHold (_root_.Flock.Readable.copyOf p t ht w o g u) base b) :
    L.rowHeld x s w r = some (pre ++ b.data.toList) := by
  rw [LayoutB.rowHeld, hm, Option.map_some, readSegs_one, readAt_copyPs htop hb]

theorem held_of_colsHoldB {L : LayoutB} {x : Msg} {s i : ℕ} {p : _root_.Flock.Readable.Parsed} {t : ℕ}
    {ht : t < p.sts.size} {w : Claims.Flock.Wit} {o g u base : ℕ} {pre : Msg} {b : ByteArray}
    (hfx : L.fixedAt x i = none)
    (hm : L.msgAt x s i = some (segsAt [(pre, VStar.copyPs p t ht o g u base (8 * b.size))]))
    (htop : base + 8 * b.size ≤ p.sts[t].c.unit.constPos + 1)
    (hb : FlockVBridge.ColsHold (_root_.Flock.Readable.copyOf p t ht w o g u) base b) :
    L.held x s w i = some (pre ++ b.data.toList) := by
  simp only [LayoutB.held, hfx, Option.isSome_none, Bool.false_eq_true, ↓reduceIte, hm, Option.map_some]
  rw [readSegs_one, readAt_copyPs htop hb]

def vcalls (H : Msg → Msg) (b : RecDeploy.Brings) (xs : List Msg) : List (Msg × Msg) :=
  xs.flatMap fun xO => (b.setup xO).map fun u => (u, H u)

theorem mem_vcalls {H : Msg → Msg} {b : RecDeploy.Brings} {xs : List Msg} {xO u : Msg} (hx : xO ∈ xs)
    (hu : u ∈ b.setup xO) : (u, H u) ∈ vcalls H b xs :=
  List.mem_flatMap.2 ⟨xO, hx, List.mem_map.2 ⟨u, hu, rfl⟩⟩

theorem vcalls_true {H : Msg → Msg} {b : RecDeploy.Brings} {xs : List Msg} : ∀ e ∈ vcalls H b xs, H e.1 = e.2 := by
  intro e he
  obtain ⟨xO, -, he⟩ := List.mem_flatMap.1 he
  obtain ⟨u, -, rfl⟩ := List.mem_map.1 he
  rfl

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
