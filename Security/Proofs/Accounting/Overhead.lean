import Definitions.Accounting.Guarantees.Overhead
import Proofs.Core.Game.Prob
import Mathlib.Tactic.FinCases

/-!
# Proofs of the end-to-end overhead guarantee's generic steps

* `EndToEnd`: on a run where the verifier accepts, the computation succeeds and its cost is below `s_min`, either a claim
  fails, or the claims hold and the computation is outside the class, or it is in the class with its count below `N`
  (else its cost would be at least `s_min`); the union bound at the strategy (`Verity.Game.prob_or_le`) sums the three.
  `endToEnd_at` is the same at one strategy.
* `ThresholdOfPointwise`: a circuit smaller than `ρ T₀` has `P` nowhere in the class, so its success probability is its
  probability of succeeding without `P`, at most `η`: it is not among the circuits the threshold ranges over.
* `ReferenceBound`: the reference is among them. `OverheadOfReference` divides the two bounds.

The honest and nontrivial statements are proved at constructed instances: the circuit with no gate (`emptyCircuit`),
the NOT circuit (`notCircuit`, one input gate and one operation gate that negates it), the task with one input and every
output acceptable (`unitTask`), and the NOT task (`notTask`). A circuit with no operation gate evaluates to its inputs
(`eval_of_size_eq_zero`), so on the NOT task it never succeeds at `x = 0` (`notOut_false_of_size_eq_zero`).
-/

namespace Accounting.SecurityProofs.Overhead

open Verity Verity.Game Accounting.Overhead
open scoped ENNReal

/-! ## Step 1: the event decomposition -/

/-- **The event decomposition at one strategy.** -/
theorem endToEnd_at {O : Type} (G : Game O) (s : Strategy G) (Accept Claims InClass Success : O → Prop)
    (M S : O → ℝ) (N sMin : ℝ) (δ δb η : ℝ≥0∞)
    (h1 : prob (fun o => Accept o ∧ ¬ Claims o) G s ≤ δ)
    (h2 : prob (fun o => Claims o ∧ ¬ InClass o) G s ≤ δb)
    (h3 : ∀ o, InClass o → N ≤ M o → sMin ≤ S o)
    (h4 : prob (fun o => InClass o ∧ Success o ∧ M o < N) G s ≤ η) :
    prob (fun o => Accept o ∧ Success o ∧ S o < sMin) G s ≤ δ + δb + η := by
  calc prob (fun o => Accept o ∧ Success o ∧ S o < sMin) G s
      ≤ prob (fun o => ((Accept o ∧ ¬ Claims o) ∨ (Claims o ∧ ¬ InClass o)) ∨
          (InClass o ∧ Success o ∧ M o < N)) G s := by
        refine prob_mono (fun o ⟨ha, hsu, hS⟩ => ?_) G s
        by_cases hc : Claims o
        · by_cases hi : InClass o
          · refine Or.inr ⟨hi, hsu, ?_⟩
            by_contra hM
            exact absurd (h3 o hi (not_lt.1 hM)) (not_le.2 hS)
          · exact Or.inl (Or.inr ⟨hc, hi⟩)
        · exact Or.inl (Or.inl ⟨ha, hc⟩)
    _ ≤ prob (fun o => (Accept o ∧ ¬ Claims o) ∨ (Claims o ∧ ¬ InClass o)) G s +
          prob (fun o => InClass o ∧ Success o ∧ M o < N) G s := prob_or_le _ _ G s
    _ ≤ prob (fun o => Accept o ∧ ¬ Claims o) G s + prob (fun o => Claims o ∧ ¬ InClass o) G s +
          prob (fun o => InClass o ∧ Success o ∧ M o < N) G s := add_le_add (prob_or_le _ _ G s) le_rfl
    _ ≤ δ + δb + η := add_le_add (add_le_add h1 h2) h4

theorem EndToEnd : Accounting.Guarantees.Overhead.EndToEnd :=
  fun _ G Accept Claims InClass Success M S N sMin δ δb η h1 h2 h3 h4 s =>
    endToEnd_at G s (Accept s) (Claims s) (InClass s) (Success s) (M s) (S s) N sMin (δ s) (δb s) (η s)
      (h1 s) (h2 s) (h3 s) (h4 s)

theorem EndToEnd_honest : Accounting.Guarantees.Overhead.EndToEnd_honest := by
  refine ⟨Unit, Game.ret (), fun _ _ => True, fun _ _ => True, fun _ _ => True, fun _ _ => True, fun _ _ => 0,
    fun _ _ => 0, 0, 0, fun s => ?_, fun s => ?_, fun s => ?_, fun _ _ _ _ => le_rfl, fun s => ?_⟩ <;>
    simp [prob]

theorem EndToEnd_nontrivial : Accounting.Guarantees.Overhead.EndToEnd_nontrivial := by
  refine ⟨fun h => ?_, fun h => ?_, fun h => ?_, fun h => ?_⟩
  · -- the claims fail on every run: only the joint guarantee excludes it
    have := h Unit (Game.ret ()) (fun _ _ => True) (fun _ _ => False) (fun _ _ => False) (fun _ _ => True)
      (fun _ _ => 0) (fun _ _ => 0) 1 1 (fun _ => 0) (fun _ => 0) (fun _ => 0) (fun _ _ => trivial)
      (fun s => by simp [prob]) (fun _ _ h => h.elim) (fun s => by simp [prob]) PUnit.unit
    simp [prob] at this
  · -- the claims hold and the computation is outside the class: only the bridges exclude it
    have := h Unit (Game.ret ()) (fun _ _ => True) (fun _ _ => True) (fun _ _ => False) (fun _ _ => True)
      (fun _ _ => 0) (fun _ _ => 0) 1 1 (fun _ => 0) (fun _ => 0) (fun _ => 0) (fun _ _ => trivial)
      (fun s => by simp [prob]) (fun _ _ h => h.elim) (fun s => by simp [prob]) PUnit.unit
    simp [prob] at this
  · -- the count reaches `N` at a cost below `s_min`: only the pointwise bound excludes it
    have := h Unit (Game.ret ()) (fun _ _ => True) (fun _ _ => True) (fun _ _ => True) (fun _ _ => True)
      (fun _ _ => 1) (fun _ _ => 0) 1 1 (fun _ => 0) (fun _ => 0) (fun _ => 0) (fun _ _ => trivial)
      (fun s => by simp [prob]) (fun s => by simp [prob]) (fun s => by simp [prob])
      PUnit.unit
    simp [prob] at this
  · -- the computation succeeds in the class with its count below `N`: only the coverage exception excludes it
    have := h Unit (Game.ret ()) (fun _ _ => True) (fun _ _ => True) (fun _ _ => True) (fun _ _ => True)
      (fun _ _ => 0) (fun _ _ => 0) 1 1 (fun _ => 0) (fun _ => 0) (fun _ => 0) (fun _ _ => trivial)
      (fun s => by simp [prob]) (fun s => by simp [prob])
      (fun _ _ _ h => absurd h (by norm_num)) PUnit.unit
    simp [prob] at this

/-! ## Step 8: the threshold and the overhead -/

theorem ThresholdOfPointwise : Accounting.Guarantees.Overhead.ThresholdOfPointwise := by
  intro X Y G A out P ρ T₀ η hP hcov
  refine le_iInf₂ fun C hC => ?_
  by_contra hlt
  have hnP : ∀ x, (C, x) ∈ A → ¬ P C x := fun x hx hp => hlt (hP C x hx hp)
  have hle : success G A out C ≤ η :=
    (prob_mono (fun p ⟨hA, hY⟩ => ⟨hA, hY, hnP _ hA⟩) _ _).trans (hcov C)
  exact absurd hC (not_lt.2 hle)

theorem ReferenceBound : Accounting.Guarantees.Overhead.ReferenceBound :=
  fun _ _ _ _ _ C₀ _ _ hsize hsucc => (iInf₂_le C₀ hsucc).trans hsize

theorem OverheadOfReference : Accounting.Guarantees.Overhead.OverheadOfReference := by
  intro X Y G A Aphys out P C₀ ρ T₀ η h0 htop hP hcov hsize hsucc
  have hR := ReferenceBound X Y G Aphys out C₀ T₀ η hsize hsucc
  have hT := ThresholdOfPointwise X Y G A out P ρ T₀ η hP hcov
  refine ⟨hR, ?_⟩
  calc ρ = ρ * T₀ / T₀ := (ENNReal.mul_div_cancel_right h0 htop).symm
    _ ≤ threshold G A out η / T₀ := ENNReal.div_le_div_right hT _
    _ ≤ threshold G A out η / threshold G Aphys out η := ENNReal.div_le_div_left hR _

/-! ## The constructed instances -/

/-- **The probability over a task**, as the average over its coin. -/
theorem Task.prob_eq {X Y : Type} (G : Task X Y) (E : X → Set Y → Prop) :
    G.prob E = ∑ ω, ((Fintype.card G.Ω : ℝ≥0∞))⁻¹ * (open Classical in if E (G.draw ω).1 (G.draw ω).2 then 1 else 0) :=
  rfl

/-- **The circuit with no gate.** -/
def emptyCircuit : Circuit where
  N := 0
  w := Fin.elim0
  isInput := Fin.elim0
  reads := Fin.elim0
  reads_lt g := g.elim0
  input_reads g := g.elim0
  op g := g.elim0
  op_congr g := g.elim0
  outputs := ∅

theorem emptyCircuit_size : emptyCircuit.size emptyCircuit.computed = 0 := rfl

/-- **The NOT circuit**: gate `0` an input bit, gate `1` its negation, the one output. -/
def notCircuit : Circuit where
  N := 2
  w _ := 1
  isInput g := decide (g.val = 0)
  reads g := if g.val = 1 then {0} else ∅
  reads_lt g a ha := by
    fin_cases g <;> simp_all
  input_reads g hg := by
    fin_cases g <;> simp_all
  op g v := if g.val = 1 then ~~~ (v 0) else 0
  op_congr g v v' h := by
    by_cases hg : g.val = 1
    · simp only [hg, ite_true]
      rw [h 0 (by simp [hg])]
    · simp [hg]
  outputs := {1}

theorem notCircuit_size : notCircuit.size notCircuit.computed = 1 := by decide

/-- The task with one input and every output acceptable. -/
def unitTask : Task Unit Unit := { Ω := Unit, draw := fun _ => ((), Set.univ) }

theorem unitTask_prob (E : Unit → Set Unit → Prop) :
    unitTask.prob E = open Classical in if E () Set.univ then 1 else 0 := by
  rw [Task.prob_eq]
  change ∑ _ω : Unit, ((Fintype.card Unit : ℝ≥0∞))⁻¹ * (open Classical in if E () Set.univ then 1 else 0) = _
  simp

/-- The unit task's outputs. -/
def unitOut : Circuit → Unit → Unit := fun _ _ => ()

theorem unitTask_success (A : Class Unit) (C : Circuit) :
    success unitTask A unitOut C = open Classical in if (C, ()) ∈ A then 1 else 0 := by
  rw [success, unitTask_prob]
  by_cases h : (C, ()) ∈ A <;> simp [h]

/-- Over every circuit, the unit task's threshold at `0` is `0`: the circuit with no gate succeeds always. -/
theorem unitTask_threshold_univ : threshold unitTask Set.univ unitOut 0 = 0 := by
  refine le_antisymm ((iInf₂_le emptyCircuit ?_).trans_eq ?_) zero_le
  · rw [unitTask_success]; simp
  · simp [emptyCircuit_size]

/-- Over no circuit, no circuit succeeds. -/
theorem success_empty {X Y : Type} (G : Task X Y) (out : Circuit → X → Y) (C : Circuit) : success G ∅ out C = 0 :=
  le_antisymm ((prob_mono (q := fun _ => False) (fun _ hp => hp.1) _ _).trans_eq (prob_false _ _)) zero_le

/-- Over no circuit, every threshold is `⊤`. -/
theorem threshold_empty {X Y : Type} (G : Task X Y) (out : Circuit → X → Y) (η : ℝ≥0∞) :
    threshold G ∅ out η = ⊤ :=
  iInf₂_eq_top.2 fun C hC => absurd hC (by simp [success_empty])

/-- The NOT task: a uniform input bit `x`, whose one acceptable output is `¬ x`. -/
def notTask : Task Bool Bool := { Ω := Bool, draw := fun ω => (ω, {!ω}) }

/-- The NOT task's outputs: whether some output gate's low bit is set, every input gate holding `x`. -/
def notOut : Circuit → Bool → Bool := fun C x =>
  decide (∃ o ∈ C.outputs, (C.eval (fun g => BitVec.ofNat (C.w g) x.toNat) o).getLsbD 0 = true)

theorem notTask_prob (E : Bool → Set Bool → Prop) :
    notTask.prob E = open Classical in
      2⁻¹ * (if E true {false} then 1 else 0) + 2⁻¹ * (if E false {true} then 1 else 0) := by
  rw [Task.prob_eq]
  change ∑ ω : Bool, ((Fintype.card Bool : ℝ≥0∞))⁻¹ * (open Classical in if E ω {!ω} then 1 else 0) = _
  rw [Fintype.sum_bool, Fintype.card_bool]
  simp only [Nat.cast_ofNat, mul_ite, mul_one, mul_zero]
  rfl

/-- **A circuit with no operation gate evaluates to its inputs.** -/
theorem eval_of_size_eq_zero {C : Circuit} (h : C.size C.computed = 0) (a : C.Transcript) (g : Fin C.N) :
    C.eval a g = a g := by
  have hin : C.isInput g = true := by
    by_contra hg
    have hm : g ∈ C.computed := by
      simp only [Circuit.computed, Finset.mem_filter, Finset.mem_univ, true_and]
      exact Bool.eq_false_iff.2 hg
    have := Finset.card_pos.2 ⟨g, hm⟩
    simp only [Circuit.size] at h
    omega
  unfold Circuit.eval
  rw [Circuit.run]
  simp [hin]

/-- **A circuit with no operation gate fails the NOT task at `x = 0`**: its outputs hold `0`. -/
theorem notOut_false_of_size_eq_zero {C : Circuit} (h : C.size C.computed = 0) : notOut C false = false := by
  simp [notOut, eval_of_size_eq_zero h]

/-- **The NOT circuit's output gate** holds the negation of its input gate. -/
theorem notCircuit_eval (a : notCircuit.Transcript) :
    notCircuit.eval a ⟨1, by decide⟩ = ~~~ (a ⟨0, by decide⟩) := by
  have h1 : notCircuit.isInput ⟨1, by decide⟩ = false := rfl
  have h0 : notCircuit.isInput ⟨0, by decide⟩ = true := rfl
  unfold Circuit.eval
  rw [Circuit.run]
  have hop : ∀ v, notCircuit.op ⟨1, by decide⟩ v = ~~~ (v ⟨0, by decide⟩) := fun _ => rfl
  simp only [h1, Bool.false_eq_true, ite_false, hop]
  rw [dite_eq_left (show (⟨0, by decide⟩ : Fin notCircuit.N) < ⟨1, by decide⟩ by decide), Circuit.run]
  simp only [h0, ite_true]
  rfl

/-- **The NOT circuit computes NOT.** -/
theorem notOut_notCircuit (x : Bool) : notOut notCircuit x = !x := by
  have hout : notCircuit.outputs = {⟨1, by decide⟩} := rfl
  simp only [notOut, hout, Finset.mem_singleton, exists_eq_left, notCircuit_eval]
  cases x <;> decide

theorem notCircuit_success : success notTask Set.univ notOut notCircuit = 1 := by
  rw [success, notTask_prob]
  simp only [Set.mem_univ, true_and, notOut_notCircuit, Bool.not_true, Bool.not_false, Set.mem_singleton_iff,
    ite_true, mul_one]
  exact ENNReal.inv_two_add_inv_two

theorem half_lt_one : (2⁻¹ : ℝ≥0∞) < 1 := ENNReal.inv_lt_one.2 (by norm_num)

theorem ThresholdOfPointwise_honest : Accounting.Guarantees.Overhead.ThresholdOfPointwise_honest := by
  refine ⟨⟨notCircuit, ?_⟩, fun C _ _ h => ?_, fun C => ?_⟩
  · exact notCircuit_success ▸ half_lt_one
  · simpa using h
  · change notTask.prob (fun x 𝒴 => (C, x) ∈ (Set.univ : Class Bool) ∧ notOut C x ∈ 𝒴 ∧
      ¬ 1 ≤ C.size C.computed) ≤ 2⁻¹
    rw [notTask_prob]
    by_cases hC : 1 ≤ C.size C.computed
    · simp [hC]
    · have h0 : C.size C.computed = 0 := by omega
      have : ¬ (notOut C false ∈ ({true} : Set Bool)) := by simp [notOut_false_of_size_eq_zero h0]
      simp only [Set.mem_univ, true_and, this, false_and, ite_false, mul_zero, add_zero]
      calc (2⁻¹ : ℝ≥0∞) * _ ≤ 2⁻¹ * 1 := mul_le_mul' le_rfl (by split_ifs <;> simp)
        _ = 2⁻¹ := mul_one _

theorem ThresholdOfPointwise_nontrivial : Accounting.Guarantees.Overhead.ThresholdOfPointwise_nontrivial := by
  refine ⟨fun h => ?_, fun h => ?_⟩
  · have := h Unit Unit unitTask Set.univ unitOut (fun C _ => 1 ≤ C.size C.computed) 1 1 0
      (fun C _ _ hC => by simpa using hC)
    rw [unitTask_threshold_univ] at this
    simp at this
  · have := h Unit Unit unitTask Set.univ unitOut (fun _ _ => True) 1 1 0
      (fun C => by rw [unitTask_prob]; simp)
    rw [unitTask_threshold_univ] at this
    simp at this

theorem ReferenceBound_honest : Accounting.Guarantees.Overhead.ReferenceBound_honest :=
  ⟨notCircuit, by rw [notCircuit_size]; simp, notCircuit_success ▸ half_lt_one⟩

/-- Over the circuits with at least one operation gate, the unit task's threshold at `0` is at least `1`. -/
theorem unitTask_one_le_threshold :
    (1 : ℝ≥0∞) ≤ threshold unitTask {p | 1 ≤ p.1.size p.1.computed} unitOut 0 := by
  have := ThresholdOfPointwise Unit Unit unitTask {p | 1 ≤ p.1.size p.1.computed} unitOut (fun _ _ => True) 1 1 0
    (fun C _ hC _ => by simpa using hC) (fun C => by rw [unitTask_prob]; simp)
  simpa using this

theorem ReferenceBound_nontrivial : Accounting.Guarantees.Overhead.ReferenceBound_nontrivial := by
  refine ⟨fun h => ?_, fun h => ?_⟩
  · have := h Unit Unit unitTask ∅ unitOut emptyCircuit 0 0 (by simp [emptyCircuit_size])
    rw [threshold_empty] at this
    simp at this
  · have := h Unit Unit unitTask {p | 1 ≤ p.1.size p.1.computed} unitOut notCircuit 0 0
      (by rw [unitTask_success]; simp [notCircuit_size])
    exact absurd (unitTask_one_le_threshold.trans this) (by simp)

theorem OverheadOfReference_honest : Accounting.Guarantees.Overhead.OverheadOfReference_honest := by
  refine ⟨fun C _ hC _ => ?_, fun C => ?_, notCircuit, by rw [notCircuit_size]; simp,
    notCircuit_success ▸ half_lt_one⟩
  · have : (2 : ℝ≥0∞) ≤ (C.size C.computed : ℝ≥0∞) := by exact_mod_cast hC
    simpa using this
  · change notTask.prob _ ≤ _
    rw [notTask_prob]
    simp

theorem OverheadOfReference_nontrivial : Accounting.Guarantees.Overhead.OverheadOfReference_nontrivial := by
  have hsize : ((notCircuit.size notCircuit.computed : ℕ) : ℝ≥0∞) ≤ 1 := by rw [notCircuit_size]; simp
  have hsucc : (0 : ℝ≥0∞) < success unitTask Set.univ unitOut notCircuit := by rw [unitTask_success]; simp
  have hover : overhead unitTask Set.univ Set.univ unitOut 0 = 0 := by
    rw [overhead, unitTask_threshold_univ, ENNReal.zero_div]
  refine ⟨fun h => ?_, fun h => ?_, fun h => ?_⟩
  · have := (h Unit Unit unitTask Set.univ Set.univ unitOut (fun C _ => 1 ≤ C.size C.computed) notCircuit 1 1 0
      one_ne_zero ENNReal.one_ne_top (fun C _ _ hC => by simpa using hC) hsize hsucc).2
    rw [hover] at this
    simp at this
  · have := (h Unit Unit unitTask Set.univ Set.univ unitOut (fun _ _ => True) notCircuit 1 1 0
      one_ne_zero ENNReal.one_ne_top (fun C => by rw [unitTask_prob]; simp) hsize hsucc).2
    rw [hover] at this
    simp at this
  · have := (h Unit Unit unitTask {p | 1 ≤ p.1.size p.1.computed} ∅ unitOut (fun _ _ => True) notCircuit 1 1 0
      one_ne_zero ENNReal.one_ne_top (fun C _ hC _ => by simpa using hC) (fun C => by rw [unitTask_prob]; simp)
      hsize).1
    rw [threshold_empty] at this
    simp at this

end Accounting.SecurityProofs.Overhead
