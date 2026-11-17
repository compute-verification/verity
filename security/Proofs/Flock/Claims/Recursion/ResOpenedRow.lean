module

public import Proofs.Flock.Claims.Recursion.OpenSlots
public import Proofs.Flock.Claims.Recursion.RestReadsAt
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
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.Recursion.BoundAccepts
import Proofs.Flock.Claims.Recursion.CutOpens
import Proofs.Flock.Claims.Recursion.HmOpen
import Proofs.Flock.Claims.Recursion.RestFinal

/-! The row a stream round opens is its first holder's (`opened_row`). -/

@[expose] public section

namespace RecDeploy.OpenSlots

open Flock
open Claims (Msg)
open Claims.Recursion.VStarB
open RecDeploy.DeployedR
open RecDeploy.RestRead

theorem os_rows (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) {x : Msg}
    (hx : DenseX x) {ws : List Claims.Flock.Wit} (hl : ws.length = nS x) (d : Claims.Flock.Wit) {os last}
    (hf : finalR (((DeployedR.layR elem).msgs x ws).getD (kI x - 1) []) = some (os, last)) :
    ∃ r, readX x = some r ∧ os.map rowsR =
      (slotRounds r).map fun l => l.map fun σ => (⟨(readForms (ws.getD σ.s d) σ.row).toArray⟩ : ByteArray) := by
  obtain ⟨r, hr, hlast, -⟩ := lastAt_parts elem hx hl
  have hn : nS x = r.w.parts.length := by simp [nS, hr]
  obtain ⟨L, hL⟩ : ∃ L : List (List (Msg × Msg)),
      L = (slotRounds r).map fun l => l.map fun σ => σ.opening (ws.getD σ.s d) := ⟨_, rfl⟩
  have hpl : ∀ p ∈ openPieces r, p.1 < ws.length := fun p hp => by
    rw [hl, hn]
    exact openPieces_lt hr p hp
  have hlens : openLens r = L.map fun l => l.map (·.1.length) := by
    rw [openLens_eq, hL, List.map_map]
    refine List.map_congr_left fun l _ => ?_
    simp only [Function.comp_apply, List.map_map]
    refine List.map_congr_left fun σ _ => ?_
    simp [Slot.opening, readForms_length, pieceLen]
  have hbytes : readPieces (openPieces r) ws = L.flatMap (·.flatMap fun p => p.1 ++ p.2) := by
    rw [openBytes r ws d hpl, hL, List.flatMap_map]
    refine flatMap_congr_mem _ fun l _ => ?_
    simp only [List.flatMap_map]
    exact flatMap_congr_mem _ fun σ _ => readSegs_segs _ σ
  have hsalt : ∀ l ∈ L, ∀ p ∈ l, p.2.length = 192 := by
    intro l hl p hp
    rw [hL] at hl
    obtain ⟨l', -, rfl⟩ := List.mem_map.1 hl
    obtain ⟨σ, -, rfl⟩ := List.mem_map.1 hp
    exact HmOpen.length_salt _ _ _ _
  have hf' : finalR (lastAt (restR elem) x ws) = some (os, last) := hf
  rw [hlast, finalR_trailer, hlens, hbytes, CutOpens.cutRounds_groups _ L hsalt] at hf'
  obtain ⟨rfl, -⟩ := Prod.mk.inj (Option.some.inj hf')
  refine ⟨r, hr, ?_⟩
  rw [hL]
  simp [rowsR, Slot.opening, List.map_map, Function.comp_def]

theorem capSlots_flat_length (r : XR) (rs : Array RecVStar.SchedRound) (j : ℕ) :
    ((List.range ((rs.getD (j % rs.size) default).2.1).size).flatMap (capSlotsAt r rs j)).length =
      ((rs.getD (j % rs.size) default).2.1).toList.sum := by
  rw [List.length_flatMap]
  have e : (List.range ((rs.getD (j % rs.size) default).2.1).size).map (fun e => (capSlotsAt r rs j e).length) =
      ((rs.getD (j % rs.size) default).2.1).toList := by
    apply List.ext_getElem (by simp)
    intro i h1 h2
    have hi : i < ((rs.getD (j % rs.size) default).2.1).size := by simpa using h1
    have hi' : i < (rs[j % rs.size]?.getD default).2.1.size := by simpa [Array.getD_eq_getD_getElem?] using hi
    simp [capSlotsAt, capSlots, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hi']
  rw [e]

theorem roundSlots_row {r : XR} {rs : Array RecVStar.SchedRound} {j : ℕ} {σ : Slot} (h : rowSlot r j = some σ) :
    (roundSlots r rs j)[((rs.getD (j % rs.size) default).2.1).toList.sum]? = some σ := by
  unfold roundSlots
  rw [h, Option.toList_some, List.getElem?_append_right (le_of_eq (capSlots_flat_length r rs j)),
    capSlots_flat_length, Nat.sub_self]
  rfl

theorem opened_row (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) {x : Msg}
    (hx : DenseX x) {ws : List Claims.Flock.Wit} (hl : ws.length = nS x) (d : Claims.Flock.Wit) {os last}
    (hf : finalR (((DeployedR.layR elem).msgs x ws).getD (kI x - 1) []) = some (os, last)) {r : XR}
    (hr : readX x = some r) {rs : Array RecVStar.SchedRound} (hrs : repSched r.shape = some rs) {j : ℕ}
    (hj : j < 2 * rs.size) {σ : Slot} (hσ : rowSlot r j = some σ) :
    (rowsR (os.getD (j + 1) [])).getD ((rs.getD (j % rs.size) default).2.1).toList.sum default =
      (⟨(readForms (ws.getD σ.s d) σ.row).toArray⟩ : ByteArray) := by
  obtain ⟨r', hr', hos⟩ := os_rows elem hx hl d hf
  rw [hr] at hr'
  obtain rfl := Option.some.inj hr'
  obtain ⟨N, hc, -⟩ := capAt_readX hr
  have hSR : slotRounds r =
      capSlots r (nS x - 1) (List.range N) :: (List.range (2 * rs.size)).map (roundSlots r rs) := by
    simp [slotRounds, hc, hrs]
  have e : rowsR (os.getD (j + 1) []) = (os.map rowsR).getD (j + 1) [] := by
    rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_map]
    cases os[j + 1]? <;> rfl
  rw [e, hos, hSR]
  simp only [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_cons_succ, List.getElem?_range hj,
    Option.map_some, Option.getD_some, roundSlots_row hσ]

end RecDeploy.OpenSlots
