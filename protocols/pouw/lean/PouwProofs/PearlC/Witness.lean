import PouwProofs.PearlC.Prob
import Pouw.PearlC.Statements
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# TT_OUT is satisfiable

The witness protocol has Pearl-C's headline accounting and constant reference words. Its cost model has two programs:
run the honest reference (at exactly the credited work of every unit) or commit nothing. Every adversary's correct
units then cost it at least their credited work, so `TT_OUT(1/400)` holds with error 0, beside an honest reference
that runs at `W_ref` and an in-domain workload with positive credited work.
-/

namespace Pouw.PearlC

open Finset Assumptions

/-- The witness protocol. -/
def witnessP : Protocol Unit Unit Unit where
  ticket := fun _ _ _ _ _ _ _ => 0
  useful := fun _ _ _ _ _ _ _ => 0
  actOK := fun _ _ => True
  weightOK := fun _ _ => True
  credit := fun L u => (creditOf Params.headline (L.shape u) : ℝ)
  Wref := fun L u => (wrefOf Params.headline (L.shape u) : ℝ)

/-- The witness cost model: `true` runs the honest reference at the credited work, `false` commits nothing. -/
def witnessCM : CostModel Unit Unit Unit where
  Prog := Bool
  run := fun b H U act _ s => if b then honestTranscript witnessP H s U act else fun _ => none
  cost := fun b _ U _ _ _ => if b then ∑ u ∈ range U.layout.N, witnessP.credit U.layout u else 0
  calls := fun _ _ _ _ _ _ => 0

theorem creditOf_nonneg (sh : Shape) : 0 ≤ creditOf Params.headline sh := by
  unfold creditOf Params.headline
  positivity

theorem credit_le_wref (sh : Shape) : creditOf Params.headline sh ≤ wrefOf Params.headline sh := by
  unfold wrefOf
  have : (0 : ℚ) ≤ (Params.headline.qa : ℚ) * sh.m * sh.k := by positivity
  linarith

theorem witness_sum_le (U : Workload) (H : Unit → Unit) (s : Unit) (b : Bool) (act : ℕ → Codes) (σ : State) :
    ∑ u ∈ correctSet witnessP U H s (witnessCM.run b H U act σ s), witnessP.credit U.layout u ≤
      witnessCM.cost b H U act σ s := by
  classical
  cases b with
  | false =>
    have hempty : correctSet witnessP U H s (witnessCM.run false H U act σ s) = ∅ := by
      unfold correctSet
      apply filter_false_of_mem
      intro u _ h
      obtain ⟨c, hc, _⟩ := h
      simp [witnessCM] at hc
    rw [hempty, sum_empty]
    simp [witnessCM]
  | true =>
    have hsub : correctSet witnessP U H s (witnessCM.run true H U act σ s) ⊆ range U.layout.N := by
      unfold correctSet
      exact filter_subset _ _
    have hle := sum_le_sum_of_subset_of_nonneg hsub (fun u _ _ => by
      show (0 : ℝ) ≤ (creditOf Params.headline (U.layout.shape u) : ℝ)
      exact_mod_cast creditOf_nonneg _)
    simpa [witnessCM, witnessP] using hle

theorem ttOutWitness : TTOutWitness := by
  refine ⟨witnessCM, witnessP, fun _ _ => ⟨rfl, rfl⟩, ?_, ?_, ?_⟩
  · refine ⟨true, fun U _ act _ H s => ⟨by simp [witnessCM], ?_, by simp [witnessCM]⟩⟩
    show (if true = true then ∑ u ∈ range U.layout.N, witnessP.credit U.layout u else 0) ≤ _
    apply sum_le_sum
    intro u _
    show (creditOf Params.headline (U.layout.shape u) : ℝ) ≤ (wrefOf Params.headline (U.layout.shape u) : ℝ)
    exact_mod_cast credit_le_wref _
  · refine ⟨fun prog U act σ s => ⟨QTree.done (witnessCM.run prog (fun _ => ()) U act σ s), fun H => ?_⟩, ?_⟩
    · have hH : H = fun _ => () := funext fun _ => Subsingleton.elim _ _
      subst hH
      exact ⟨rfl, rfl⟩
    · intro U _ T hT q _ A hA
      rw [pr_eq_zero]
      intro x hx
      have hsum := witness_sum_le U x.1 x.2 A.online A.act (A.pre.eval x.1)
      have hcost := (hA x.1 x.2).1
      have hT' : T ≤ T / (1 - 1 / 400) := by
        rw [le_div_iff₀ (by norm_num)]
        nlinarith
      unfold Adversary.transcript at hx
      linarith
  · refine ⟨⟨⟨1, fun _ => ⟨1, 32, 1⟩, fun _ => 0⟩, fun _ _ _ => 0⟩, ⟨by norm_num, ?_, fun _ _ => trivial⟩, ?_⟩
    · intro u hu
      refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, Or.inr ⟨rfl, by norm_num⟩⟩
    · show (0 : ℝ) < ∑ u ∈ range 1, (creditOf Params.headline (⟨1, 32, 1⟩ : Shape) : ℝ)
      norm_num [creditOf, Params.headline]

end Pouw.PearlC
