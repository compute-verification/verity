import Definitions.Accounting.Guarantees.Sheet

/-!
# The data-sheet axioms' bridges

(I1) and (I3) from A1–A7:

* `facetCeilings_of_sheet`: each facet of the data-sheet machine is one axiom's ceiling. Issue is A2, the tensor pipe
  A3, a pipe `p` A4 scaled by `4/width_p` (its load is `≤ 0` when `width_p ≤ 0`), shared memory A5 over 32; A2 with
  counts `≥ 0` gives the cycles' sign.
* `scalarRoute_half`: at `q, q_T ≤ ½` A6's two words a lane is SCALAR-ROUTE's lanes clause, and at `b_T ≤ 0` its
  bytes clause is the split's sign.
-/

namespace Accounting.SecurityProofs.TS

open Accounting.TS Accounting.Assumptions.TS Finset

section Run

variable {K P : Type} [Fintype K] [DecidableEq P]

/-- (I1) on the data-sheet machine from A1–A5. -/
theorem facetCeilings_of_sheet (S : Sheet K P) (Q : Run K) (hn : ∀ k, 0 ≤ Q.n k) (h1 : SheetClock S Q)
    (h2 : SheetDispatch Q) (h3 : SheetTensor S Q) (h4 : SheetPipe S Q) (h5 : SheetSmem S Q) :
    FacetCeilings S.machine Q := by
  have hc : 0 ≤ Q.cycles := (sum_nonneg fun k _ => hn k).trans h2
  refine ⟨hn, hc, h1.1, h1.2, fun r => ⟨fun _ => ?_, fun h => by simp [Sheet.machine] at h⟩⟩
  cases r with
  | issue => simpa [FacetCosts.load, Sheet.machine, SheetDispatch] using h2
  | tensor => simpa [FacetCosts.load, Sheet.machine, SheetTensor] using h3
  | pipe p =>
    rcases lt_or_ge 0 (S.width p) with hwp | hwp
    · have key : S.machine.load Q.n (.pipe p) =
          4 / S.width p * ∑ k, (if S.pipe k = some p then (32 : ℝ) else 0) * Q.n k := by
        simp only [FacetCosts.load, Sheet.machine, mul_sum]
        refine sum_congr rfl fun k _ => ?_
        split_ifs <;> field_simp <;> ring
      rw [key]
      calc 4 / S.width p * ∑ k, (if S.pipe k = some p then (32 : ℝ) else 0) * Q.n k
          ≤ 4 / S.width p * (S.width p / 4 * Q.cycles) := by
            gcongr
            exact h4 p
        _ = Q.cycles := by field_simp
    · calc S.machine.load Q.n (.pipe p) ≤ 0 := by
            simp only [FacetCosts.load, Sheet.machine]
            refine sum_nonpos fun k _ => ?_
            split_ifs
            · exact mul_nonpos_of_nonpos_of_nonneg (div_nonpos_of_nonneg_of_nonpos (by norm_num) hwp) (hn k)
            · simp
        _ ≤ Q.cycles := hc
  | smem =>
    have key : S.machine.load Q.n .smem = (∑ k, S.smem k * Q.n k) / 32 := by
      simp only [FacetCosts.load, Sheet.machine, sum_div]
      refine sum_congr rfl fun k _ => ?_
      ring
    have h5' : ∑ k, S.smem k * Q.n k ≤ 32 * Q.cycles := h5
    rw [key]
    linarith

/-- (I1) from the data sheet. -/
theorem SheetFacetCeilings : Accounting.Guarantees.TS.SheetFacetCeilings := by
  intro K P _ _ S Q hn h1 h2 h3 h4 h5
  exact facetCeilings_of_sheet S Q hn h1 h2 h3 h4 h5

end Run

variable {K R T L U F X : Type} [Fintype T] [Fintype L] [Fintype U]

/-- A split run's counts are `≥ 0`. -/
theorem run_nonneg_of_split {A : Anatomy K T L U} (hs : A.Split) (k : K) : 0 ≤ A.run.n k := by
  obtain ⟨hw, hf, hx, -, hle⟩ := hs
  have := sum_nonneg fun h (_ : h ∈ univ) => hw h k
  linarith [hf k, hx k, hle k]

/-- (I3) at the ½ floor from A6. -/
theorem scalarRoute_half (Z : Model K R T L U F X) (A : Anatomy K T L U) (hs : A.Split) (h : SheetWords Z A)
    (hq : Z.q ≤ 1 / 2) (hqT : Z.qT ≤ 1 / 2) (hb : Z.bT ≤ 0) : ScalarRoute Z A := by
  obtain ⟨hnw, -, -, hwd, -⟩ := hs
  obtain ⟨-, hv, ht⟩ := h
  refine ⟨?_, fun l => ⟨?_, ?_⟩⟩
  · have := hwd .vector
    nlinarith
  · have := hwd (.table l)
    nlinarith [ht l]
  · have := hwd (.table l)
    have := hnw (.table l) (Z.bytes l)
    nlinarith

/-- (I3) at the ½ floor. -/
theorem SheetScalarRouteHalf : Accounting.Guarantees.TS.SheetScalarRouteHalf := by
  intro K R T L U F X _ _ _ Z A hs h hq hqT hb
  exact scalarRoute_half Z A hs h hq hqT hb

end Accounting.SecurityProofs.TS
