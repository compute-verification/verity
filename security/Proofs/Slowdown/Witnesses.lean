module

public import Definitions.Slowdown.Guarantees.Witnesses
public import Proofs.InferenceOnly.Rate
public import Proofs.Core
public import Proofs.InferenceOnly.Lemmas
public import Proofs.Slowdown.LoomisWhitneyBase
public import Definitions.InferenceOnly.Reads
public import Proofs.Slowdown.RemoteMonitoringPeriod
public import Proofs.Slowdown.Capability.Witnesses
public import Proofs.Core.Circuit
public import Mathlib.Analysis.SpecialFunctions.Pow.Real
public import Mathlib.Tactic.Linarith

/-!
# The inference-only policy's guarantees: proofs of their witnesses

Each witness of `Definitions.Slowdown.Guarantees.Witnesses`, at the toy (`Slowdown.Capability.Toy`) and the instances of
`Slowdown.Witness`. The facts they share come first: the toy's one proof block is a `(2, 1)`-partition of exactly two gates,
so the toy complies at `Γ = 2` and the rate `α = 1/2`, and with the token registered at `α = 0`; the threading `T1` is
`(1, 0)`-threaded and its weights incompressible at half a bit each, read off the toy's sixteen sets of gates; and `T0`
has no counted multiplication and bound `e`.
-/

@[expose] public section

open InferenceOnly

namespace Slowdown.Witness

open Verity Finset Slowdown.Capability.Toy

theorem isCK_D : P1.IsCK D 2 1 := { boolean := ⟨fun _ => rfl, by decide⟩, card := by decide, reads := by decide }

theorem isCK_Dθ : P1.IsCK Dθ 2 0 := { boolean := ⟨fun _ => rfl, by decide⟩, card := by decide, reads := by decide }

/-- The toy's one proof block is two gates reading one fresh bit, the token: it complies at `Γ = 2` and `α = 1/2`. -/
theorem compliesRate_D : CompliesRate 2 (1 / 2) D := by
  simpa only [Nat.cast_one, Nat.cast_ofNat] using SecurityProofs.compliesRate_of_complies isCK_D (by decide)

/-- With the token registered too, the toy's one proof block reads no fresh bit: it complies at `α = 0`. -/
theorem compliesRate_Dθ : CompliesRate 2 0 Dθ := by
  simpa only [Nat.cast_zero, zero_div] using SecurityProofs.compliesRate_of_complies isCK_Dθ (by decide)

theorem size_D : D.D.size D.D.computed = 2 := by decide

theorem card_T1 : T1.M.card = 1 := by decide

theorem theta_id (r : RegisteredCircuit) :
    Assumptions.ThetaRegistered r r (PartialContainment.ofContainment (idC r.D)) := fun _ _ h => h

theorem not_theta_Dθ : ¬ Assumptions.ThetaRegistered D Dθ Mθ :=
  fun h => absurd (h 1 (by decide) (by decide)) (by decide)

theorem dark_id (r : RegisteredCircuit) : (PartialContainment.ofContainment (c := r) (d := r) (idC r.D)).dark = ∅ :=
  Finset.sdiff_self _

theorem T1_threaded {k : ℝ} : T1.Threaded 2 k 1 0 := by
  have key : ∀ R : Finset (Fin 4), R ⊆ C.computed → ∀ t : Fin 1, (T1.threadReads R t).card < 1 →
      (T1.threadMults R t).card ≤ 0 * (T1.threadReads R t).card := by decide
  exact fun R hR t => key R hR.1 t

theorem T1_weights_nat : ∀ R : Finset (Fin 4), R ⊆ C.computed →
    (T1.weightsUsed R).card + (T1.threadReads R (0 : Fin 1)).card ≤ 2 * freshInput D R := by decide

theorem T1_weights {k : ℝ} : Assumptions.WeightsIncompressible D T1 2 k (1 / 2) (1 / 2) := by
  intro R hR
  have h := T1_weights_nat R hR.1
  have hs : ∑ t, (T1.threadReads R t).card = (T1.threadReads R (0 : Fin 1)).card := Fin.sum_univ_one _
  rw [Nat.cast_sum, ← Nat.cast_sum, hs]
  have h' : ((T1.weightsUsed R).card : ℝ) + (T1.threadReads R (0 : Fin 1)).card ≤ 2 * freshInput D R := by exact_mod_cast h
  linarith

theorem T1_weightsSlots {k : ℝ} : Assumptions.WeightsIncompressibleSlots D T1 σ1 2 k (1 / 2) (1 / 2) := by
  have key : ∀ R : Finset (Fin 4), R ⊆ C.computed →
      (σ1.slotsUsed R).card + (T1.threadReads R (0 : Fin 1)).card ≤ 2 * freshInput D R := by decide
  intro R hR
  have h := key R hR.1
  have hs : ∑ t, (T1.threadReads R t).card = (T1.threadReads R (0 : Fin 1)).card := Fin.sum_univ_one _
  rw [Nat.cast_sum, ← Nat.cast_sum, hs]
  have h' : ((σ1.slotsUsed R).card : ℝ) + (T1.threadReads R (0 : Fin 1)).card ≤ 2 * freshInput D R := by exact_mod_cast h
  linarith

theorem coverage_T1 : Assumptions.Coverage 1 D (2 * T1.M.card) := by
  unfold Assumptions.Coverage
  rw [size_D, card_T1]
  norm_num

theorem T0_threaded (r : RegisteredCircuit) (c : ℕ) (k : ℝ) (d e : ℕ) : (T0 r).Threaded c k d e :=
  fun _ _ t => t.elim0

theorem T0_weights (r : RegisteredCircuit) (c : ℕ) (k b_w b_a : ℝ) :
    Assumptions.WeightsIncompressible r (T0 r) c k b_w b_a := by
  intro R _
  have h0 : ∑ t, ((T0 r).threadReads R t).card = 0 := Finset.sum_eq_zero fun t _ => Fin.elim0 t
  rw [h0]
  simp [Threading.weightsUsed, Threading.mults, T0]

theorem T0_weightsSlots (r : RegisteredCircuit) (c : ℕ) (k b_w b_a : ℝ) :
    Assumptions.WeightsIncompressibleSlots r (T0 r) (σ0 r) c k b_w b_a := by
  intro R _
  have h0 : ∑ t, ((T0 r).threadReads R t).card = 0 := Finset.sum_eq_zero fun t _ => Fin.elim0 t
  rw [h0]
  simp [Threading.Slots.slotsUsed, Threading.weightsUsed, Threading.mults, T0]

theorem not_coverage_T0 : ¬ Assumptions.Coverage 1 D (2 * (T0 D).M.card) := by
  unfold Assumptions.Coverage
  rw [size_D, show (T0 D).M.card = 0 from rfl]
  norm_num

theorem bound_T0 (r : RegisteredCircuit) (k b_w : ℝ) (d e : ℕ) : (T0 r).bound k b_w d e = e := by
  simp [Threading.bound, T0]

theorem bound_T1 (k : ℝ) (hk : 0 ≤ k) : T1.bound k (1 / 2) 1 0 = k / 2 := by
  have : T1.reuse = 1 := rfl
  simp only [Threading.bound, this, Nat.cast_one, Nat.cast_zero]
  rw [max_eq_right (by positivity)]
  ring

theorem dark_Mθ : Mθ.dark = ∅ := Finset.sdiff_self _

end Slowdown.Witness

namespace Slowdown.SecurityProofs

open Verity Finset Slowdown.Capability.Toy Slowdown.Witness Slowdown.Slowdown

theorem ThreadedTraining_honest : Guarantees.ThreadedTraining_honest :=
  ⟨compliesRate_D, theta_id D, T1_threaded, T1_weights, coverage_T1⟩

theorem ThreadedTraining_nontrivial : Guarantees.ThreadedTraining_nontrivial := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, compliesRate_D, theta_id D, T0_threaded _ _ _ _ _,
    T0_weights _ _ _ _ _, not_coverage_T0, ?_⟩
  rw [bound_T0, size_D]
  norm_num

theorem ThreadedTrainingOn_nontrivial : Guarantees.ThreadedTrainingOn_nontrivial := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, compliesRate_D, theta_id D, T0_threaded _ _ _ _ _,
    T0_weights _ _ _ _ _, not_coverage_T0, ?_⟩
  rw [bound_T0, size_D, dark_id]
  norm_num

theorem ThreadedThrottled_honest : Guarantees.ThreadedThrottled_honest :=
  ⟨compliesRate_D, theta_id D, T1_threaded, T1_weights⟩

theorem ThreadedThrottled_nontrivial : Guarantees.ThreadedThrottled_nontrivial := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, compliesRate_Dθ, not_theta_Dθ, T1_threaded, T1_weights,
    ?_⟩
  rw [card_T1]
  norm_num

theorem ThreadedThrottledOn_nontrivial : Guarantees.ThreadedThrottledOn_nontrivial := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, compliesRate_Dθ, not_theta_Dθ, T1_threaded, T1_weights,
    ?_⟩
  rw [show (T1.M ∩ Mθ.S).card = 1 by decide]
  norm_num

theorem ThreadedThrottledDark_nontrivial : Guarantees.ThreadedThrottledDark_nontrivial := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, compliesRate_Dθ, not_theta_Dθ, T1_threaded, T1_weights,
    ?_⟩
  rw [card_T1, dark_Mθ]
  norm_num

theorem ThreadedThrottledSlots_honest : Guarantees.ThreadedThrottledSlots_honest :=
  ⟨compliesRate_D, theta_id D, T1_threaded, T1_weightsSlots⟩

theorem ThreadedThrottledSlots_nontrivial : Guarantees.ThreadedThrottledSlots_nontrivial := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, compliesRate_Dθ, not_theta_Dθ, T1_threaded,
    T1_weightsSlots, ?_⟩
  rw [card_T1]
  norm_num

theorem ThreadedThrottledOnSlots_nontrivial : Guarantees.ThreadedThrottledOnSlots_nontrivial := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, compliesRate_Dθ, not_theta_Dθ, T1_threaded,
    T1_weightsSlots, ?_⟩
  rw [show (T1.M ∩ Mθ.S).card = 1 by decide]
  norm_num

theorem ThreadedThrottledDarkSlots_nontrivial : Guarantees.ThreadedThrottledDarkSlots_nontrivial := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, compliesRate_Dθ, not_theta_Dθ, T1_threaded,
    T1_weightsSlots, ?_⟩
  rw [card_T1, dark_Mθ]
  norm_num

theorem LayeredTraining_honest : Guarantees.LayeredTraining_honest :=
  ⟨⟨layered1 1 2⟩, compliesRate_D, theta_id D, by norm_num [Assumptions.CapBelowColumns], T1_weights, coverage_T1⟩

theorem LayeredTraining_nontrivial : Guarantees.LayeredTraining_nontrivial := by
  refine ⟨⟨layered0 D 1 0 0 2⟩, by norm_num, by norm_num, by norm_num, by norm_num, compliesRate_D, theta_id D,
    by norm_num [Assumptions.CapBelowColumns], T0_weights _ _ _ _ _, not_coverage_T0, ?_⟩
  rw [bound_T0, size_D]
  norm_num

theorem executionPhasesHold_T1 : Assumptions.ExecutionPhasesHold T1 0 1 (1 / 2) 2 1 1 1 (1 / 2) (1 / 2) 1 0 := by
  unfold Assumptions.ExecutionPhasesHold
  rw [bound_T1 _ (by norm_num), card_T1]
  norm_num

theorem not_executionPhasesHold_T0 : ¬ Assumptions.ExecutionPhasesHold (T0 D) 1 1 (1 / 2) 0 1 1 0 (1 / 2) (1 / 2) 1 0 := by
  unfold Assumptions.ExecutionPhasesHold
  rw [bound_T0, show (T0 D).M.card = 0 from rfl]
  norm_num

theorem slowdown_T0_fails :
    ¬ (efficiency 1 1 2 0 ≤
        unregisteredShare 2 0 +
            min (residualShare 0 2) (memoryCap (T0 D).reuse 1 0 ((1 : ℕ) / (2 : ℕ) : ℝ) 1 (1 / 2) (1 / 2) * (1 + 1 / 1)) +
          declaredShare 1 (rhoT ((1 : ℕ) / (2 : ℕ) : ℝ) (1 / 2) ((T0 D).bound 1 (1 / 2) 1 0)) 2 0 + 0) := by
  rw [bound_T0, show (T0 D).reuse = 0 from rfl]
  norm_num [efficiency, unregisteredShare, residualShare, memoryCap, declaredShare, rhoT]

theorem SlowdownThreaded_honest : Guarantees.SlowdownThreaded_honest := by
  refine ⟨compliesRate_D, theta_id D, T1_threaded, T1_weights, by rw [size_D]; norm_num, by decide,
    by norm_num [Assumptions.ExecutionPhaseLength], ?_, executionPhasesHold_T1⟩
  rw [show T1.reuse = 1 from rfl]
  norm_num

theorem SlowdownThreaded_nontrivial : Guarantees.SlowdownThreaded_nontrivial := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, compliesRate_D, theta_id D, T0_threaded _ _ _ _ _, T0_weights _ _ _ _ _,
    by rw [size_D]; norm_num, by decide, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num [Assumptions.ExecutionPhaseLength], by norm_num, not_executionPhasesHold_T0, slowdown_T0_fails⟩

theorem SlowdownThreadedSlots_honest : Guarantees.SlowdownThreadedSlots_honest := by
  refine ⟨compliesRate_D, theta_id D, T1_threaded, T1_weightsSlots, by rw [size_D]; norm_num, by decide,
    by norm_num [Assumptions.ExecutionPhaseLength], ?_, executionPhasesHold_T1⟩
  rw [show T1.reuse = 1 from rfl]
  norm_num

theorem SlowdownThreadedSlots_nontrivial : Guarantees.SlowdownThreadedSlots_nontrivial := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, compliesRate_D, theta_id D, T0_threaded _ _ _ _ _, T0_weightsSlots _ _ _ _ _,
    by rw [size_D]; norm_num, by decide, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num [Assumptions.ExecutionPhaseLength], by norm_num, not_executionPhasesHold_T0, slowdown_T0_fails⟩

theorem oneExecutionPhase_withinCompute : Assumptions.WithinCompute OneWindow.cap 2 := by
  norm_num [Assumptions.WithinCompute]

theorem oneExecutionPhase_pouwSound : Assumptions.PouwSound (PMF.uniformOfFintype Bool) OneWindow.right 0 :=
  fun _ => (prob_of_forall_not _ fun _ h => h).le

theorem oneExecutionPhase_not_pouwSound : ¬ Assumptions.PouwSound (PMF.uniformOfFintype Bool) OneWindow.incorrect 0 := by
  intro h
  have h1 := h 0
  have h2 := prob_add_not (PMF.uniformOfFintype Bool) (fun ω => OneWindow.incorrect ω 0)
  rw [prob_of_forall_not (PMF.uniformOfFintype Bool) (E := fun ω => ¬ OneWindow.incorrect ω 0) fun _ h => h trivial]
    at h2
  linarith

theorem SlowdownRemote_honest : Guarantees.SlowdownRemote_honest :=
  ⟨oneExecutionPhase_withinCompute, oneExecutionPhase_pouwSound, fun _ b => b.elim, fun b => b.elim, Cut.SecurityProofs.toy_complies,
    Cut.SecurityProofs.toy_values, Cut.SecurityProofs.toy_noSharedPair, Cut.SecurityProofs.toy_hardness 2,
    Cut.SecurityProofs.toy_exports 2, by show 2 ≤ 0 + 0 + Cut.Toy.W.N + 0; rw [Cut.SecurityProofs.toy_N],
    fun _ => by norm_num [Assumptions.ResidualAtMost, incorrectCapacity]⟩

theorem SlowdownRemote_nontrivial : Guarantees.SlowdownRemote_nontrivial := by
  refine ⟨by norm_num, by norm_num, fun _ => by norm_num, by simp [Assumptions.WithinCompute], oneExecutionPhase_not_pouwSound,
    by norm_num, by norm_num, by norm_num, fun _ b => b.elim, fun _ b => b.elim,
    fun _ _ _ => by simp, by norm_num, fun b => b.elim, by norm_num, by norm_num, by norm_num,
    fun _ b => b.elim, fun b => b.elim, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, Cut.SecurityProofs.toy_complies, Cut.SecurityProofs.toy_values,
    Cut.SecurityProofs.toy_noSharedPair, Cut.SecurityProofs.toy_hardness 2, Cut.SecurityProofs.toy_exports 2,
    by norm_num, by show 100 ≤ 0 + 100 + Cut.Toy.W.N + 0; omega, by norm_num,
    fun _ => by norm_num [Assumptions.ResidualAtMost, incorrectCapacity], by norm_num, ?_⟩
  intro h
  rw [prob_of_forall_not] at h
  · norm_num at h
  · rintro - ⟨h1, -⟩
    rw [Cut.SecurityProofs.toy_workHE] at h1
    norm_num [efficiency, unregisteredShare, residualShare, lagShare] at h1

end Slowdown.SecurityProofs
