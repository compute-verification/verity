module

public import Proofs.WeightSecurity.Faulted

/-! `WeightSecurity.SecurityProofs.TheftTied : WeightSecurity.Guarantees.TheftTied`, `TheftFaulted` at a checked
population whose run records each window's extracted trace. Counting each window at its trace changes no release and so
no view (`Served.Traced.view_counted`), so `TheftFaulted` at the run counted at its traces bounds the theft, with the
probability that a release isn't faithful. Where a window's release is its trace's outputs and its trace holds its
inputs, the release is the faulted function's answer at the trace's own fault pattern (`Served.Traced.faithful_counted`,
from `Served.faithful_of_tie`), so that probability is at most `ηe + ηa`.

The witnesses are proved at the toy deployment as a population. -/

@[expose] public section

namespace WeightSecurity

namespace Served.Traced

variable {O V Tick Adv S U R Q Θ Θ' : Type}

/-- Counting a window at its trace keeps whether it was accepted, so the same windows release. -/
theorem count_counted [Fintype U] (P : Population.Dataflow U S) (f : P.Fns V) (L : ℕ) :
    ∀ l : List (Traced O V Tick Adv S), count L (l.map (counted P f)) = count L (l.map toServed)
  | [] => rfl
  | w :: l => by
    simp only [List.map_cons, count, List.length_map]
    rw [count_counted P f L l]
    rfl

/-- Counting a window at its trace keeps its release. -/
theorem reply_counted [Fintype U] (P : Population.Dataflow U S) (f : P.Fns V) (L : ℕ)
    (l : List (Traced O V Tick Adv S)) (n : ℕ) : reply L (l.map (counted P f)) n = reply L (l.map toServed) n := by
  simp only [reply, count_counted, List.getElem?_map, Option.map_map]
  rfl

/-- **Counting the windows at their traces changes no view.** -/
theorem view_counted [Fintype U] (P : Population.Dataflow U S) (f : P.Fns V)
    (Ou : Outsider R (Q × Adv) (Option ((O → V) × Tick)) Θ') (L : ℕ) (r : R) (l : List (Traced O V Tick Adv S))
    (n : ℕ) : view Ou L r (l.map (counted P f)) n = view Ou L r (l.map toServed) n := by
  simp only [view, reply_counted]

/-- **Tied and covered releases are faithful.** At a checked population, a run whose releases are their traces'
outputs and whose traces hold the windows' inputs releases, counted at its traces, the faulted function's answer at each
trace's own fault pattern (`Served.faithful_of_tie`). -/
theorem faithful_counted [Fintype U] (P : Population.Dataflow U S) (h : P.Checked) (f : P.Fns V) (eg : O → S)
    (inp : Θ → Q × Adv → S → V) (Ou : Outsider R (Q × Adv) (Option ((O → V) × Tick)) Θ') (L : ℕ) (θ : Θ) (r : R)
    (l : List (Traced O V Tick Adv S)) (ht : Tied eg L l) (hc : Covered P inp Ou L θ r l) :
    Faithful (faultedServed P f eg inp) P.faultCount Ou L θ r (l.map (counted P f)) := by
  refine faithful_of_tie P h f inp eg Ou L θ r _ fun n hn w' hw' ha => ?_
  rw [count_counted] at hn
  rw [List.getElem?_map, Option.map_eq_some_iff] at hw'
  obtain ⟨w, hw, rfl⟩ := hw'
  refine ⟨w.trace, ?_, le_rfl, ht n hn w hw ha⟩
  rw [view_counted]
  exact hc n hn w hw ha

/-- A run each of whose windows releases its trace's outputs ties its releases. -/
theorem tied_of_forall (eg : O → S) (L : ℕ) (l : List (Traced O V Tick Adv S))
    (h : ∀ w ∈ l, w.egress = fun o => w.trace (eg o)) : Tied eg L l :=
  fun _ _ w hw _ => h w (List.mem_of_getElem? hw)

end Served.Traced

namespace SecurityProofs

open Verity.Game WeightSecurity.Assumptions
open scoped ENNReal

theorem TheftTied : Guarantees.TheftTied := by
  intro Ω R Θ Q Adv O V Tick Θ' U S _ _ _ _ _ _ _ _ _ _ _ _ P f eg inp C gen Typ Stolen A Af p lam Ou T L M δ ηe ηa
    run hP hhon hfh hsim hund hte hac
  have h0 : NoFault P.faultCount (fun _ => none : P.Faults V) := by
    simp [NoFault, Population.Dataflow.faultCount, Population.Dataflow.faultSet]
  have key := TheftFaulted C (faultedServed P f eg inp) P.faultCount (fun _ => none) gen Typ Stolen A Af p lam Ou T L M
    δ (Served.Traced.countRun P f run) h0 hhon hfh hsim hund
  have hview : ∀ z : Ω × R, value (fun l => Stolen (gen z.1) (Ou.out z.2 (Served.view Ou L z.2 l T)))
      (Served.Traced.countRun P f run (gen z.1) z.2) = value (fun l => Stolen (gen z.1)
        (Ou.out z.2 (Served.view Ou L z.2 (l.map Served.Traced.toServed) T))) (run (gen z.1) z.2) := fun z => by
    simp only [Served.Traced.countRun, value_map, Served.Traced.view_counted]
  have hfa : ∀ z : Ω × R, value (fun l => ¬ Served.Faithful (faultedServed P f eg inp) P.faultCount Ou L (gen z.1)
      z.2 l) (Served.Traced.countRun P f run (gen z.1) z.2) ≤ ηe + ηa := fun z => by
    simp only [Served.Traced.countRun, value_map]
    refine (value_mono (q := fun l => ¬ Served.Traced.Tied eg L l ∨
      ¬ Served.Traced.Covered P inp Ou L (gen z.1) z.2 l) (fun l hl => ?_) _).trans
      ((value_or_le _ _ _).trans (add_le_add (hte _ _) (hac _ _)))
    by_cases ht : Served.Traced.Tied eg L l
    · by_cases hc : Served.Traced.Covered P inp Ou L (gen z.1) z.2 l
      · exact absurd (Served.Traced.faithful_counted P hP f eg inp Ou L _ _ l ht hc) hl
      · exact Or.inr hc
    · exact Or.inl ht
  calc avg (fun z : Ω × R => value (fun l => Stolen (gen z.1)
          (Ou.out z.2 (Served.view Ou L z.2 (l.map Served.Traced.toServed) T))) (run (gen z.1) z.2))
      = avg (fun z : Ω × R => value (fun l => Stolen (gen z.1) (Ou.out z.2 (Served.view Ou L z.2 l T)))
          (Served.Traced.countRun P f run (gen z.1) z.2)) := congrArg avg (funext fun z => (hview z).symm)
    _ ≤ _ := key
    _ ≤ _ := by
        rw [add_assoc _ ηe ηa]
        exact add_le_add le_rfl ((avg_mono hfa).trans (avg_le_const _))

namespace ToyTied

open WeightSecurity.ToyServed

/-- The toy population is checked: its one unit writes the one slot, which is no input, and reads nothing. -/
theorem checked : pop.Checked := by
  refine ⟨fun s => ⟨fun hs => by simp [pop] at hs, fun _ => ⟨(), by simp [pop], fun _ _ => rfl⟩⟩, fun u h => ?_⟩
  cases h with
  | single hd =>
    obtain ⟨s, -, hs⟩ := hd
    simp [pop] at hs
  | tail _ hd =>
    obtain ⟨s, -, hs⟩ := hd
    simp [pop] at hs

/-- The honest assignment gets no unit wrong. -/
theorem wrong_honest : pop.wrong fns (fun _ => 0) = ∅ := by
  ext u
  simp [Population.Dataflow.wrong, Population.Dataflow.Correct, fns]

/-- A fault pattern of size `0` names no unit. -/
theorem eq_none_of_faultCount (φ : pop.Faults (Fin 16)) (h : pop.faultCount φ = 0) : φ = fun _ => none := by
  have hu : ∀ u, u ∉ pop.faultSet φ := by
    rw [Population.Dataflow.faultCount, Finset.card_eq_zero] at h
    rw [h]
    exact Finset.notMem_empty
  funext u
  cases hφu : φ u with
  | none => rfl
  | some w => exact absurd (by simp [Population.Dataflow.faultSet, hφu]) (hu u)

theorem patternCount_le_one : patternCount (pop.faultCount (V := Fin 16)) 2 0 ≤ 1 := by
  unfold patternCount
  rw [Fintype.card_le_one_iff]
  have hz : ∀ χ : Fin 2 → pop.Faults (Fin 16), ∑ t, pop.faultCount (χ t) ≤ 0 → ∀ t, χ t = fun _ => none :=
    fun χ hχ t => eq_none_of_faultCount _ ((Finset.sum_eq_zero_iff.1 (Nat.le_zero.1 hχ)) t (Finset.mem_univ t))
  rintro ⟨φ, hφ⟩ ⟨ψ, hψ⟩
  exact Subtype.ext (funext fun t => (hz φ hφ t).trans (hz ψ hψ t).symm)

end ToyTied

theorem TheftTiedHonest : Guarantees.TheftTiedHonest := by
  classical
  obtain ⟨-, hhon, -, -, -⟩ := TheftFaultedHonest
  refine ⟨ToyTied.checked, hhon, fun p hp Pa _ => ?_, fun _ => Set.mem_univ _, fun θ r => le_of_eq ?_,
    fun θ r => le_of_eq ?_, fun θ r => le_of_eq ?_⟩
  · rw [sub_self, Real.rpow_zero, one_mul]
    exact (ToyFaulted.guess_le (faultedServed WeightSecurity.ToyServed.pop WeightSecurity.ToyServed.fns
      WeightSecurity.ToyServed.eg WeightSecurity.ToyServed.inp) (fun _ => rfl) Pa).trans (ToyFaulted.le_of_hard hp)
  · simp only [Served.Traced.countRun, WeightSecurity.ToyServed.honestTraced, value_map]
    rw [value_ret, ite_eq_right]
    rintro (⟨w, hw, -, hd⟩ | h)
    · simp only [List.map_cons, List.map_nil, List.mem_cons, List.not_mem_nil, or_false] at hw
      rcases hw with rfl | rfl <;> exact hd
    · simp [Served.lead, Served.Traced.counted, WeightSecurity.ToyServed.window, ToyTied.wrong_honest] at h
  · unfold WeightSecurity.ToyServed.honestTraced
    rw [value_ret, ite_eq_right]
    exact not_not.2 (Served.Traced.tied_of_forall _ _ _ fun w hw => by
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hw
      rcases hw with rfl | rfl <;> rfl)
  · unfold WeightSecurity.ToyServed.honestTraced
    rw [value_ret, ite_eq_right]
    exact not_not.2 fun _ _ _ _ _ s hs => absurd hs (by simp [WeightSecurity.ToyServed.pop])

theorem TheftTiedNontrivial : Guarantees.TheftTiedNontrivial := by
  classical
  intro h
  have hsteal : ∀ z : WeightSecurity.ToyServed.W × Unit, value (fun l => WeightSecurity.ToyServed.Stolen
      (WeightSecurity.ToyServed.gen z.1) (WeightSecurity.ToyServed.thief.out z.2
        (Served.view WeightSecurity.ToyServed.thief 0 z.2 (l.map Served.Traced.toServed) 2)))
      (WeightSecurity.ToyServed.leakTraced (WeightSecurity.ToyServed.gen z.1) z.2) = 1 := by
    intro z
    unfold WeightSecurity.ToyServed.leakTraced
    rw [value_ret, ite_eq_left]
    simp [WeightSecurity.ToyServed.Stolen, WeightSecurity.ToyServed.gen, WeightSecurity.ToyServed.thief,
      WeightSecurity.ToyServed.window, Served.view, Served.reply, Served.count, Outsider.view]
  have htyp : Claims.Pr (fun z : WeightSecurity.ToyServed.W × Unit =>
      WeightSecurity.ToyServed.gen z.1 ∉ (Set.univ : Set WeightSecurity.ToyServed.W)) = 0 :=
    Pouw.pr_eq_zero fun z hz => hz (Set.mem_univ _)
  have hN : (patternCount (WeightSecurity.ToyServed.pop.faultCount (V := Fin 16)) 2 0 : ℝ) ^ ((1 : ℝ) - 1) = 1 := by
    rw [sub_self, Real.rpow_zero]
  have hc : Fintype.card (faultChannel Unit Unit Unit Unit (Fin 16) Unit
      (WeightSecurity.ToyServed.pop.Faults (Fin 16)) WeightSecurity.ToyServed.pop.faultCount 2 0
        ((0 + 1) * Fintype.card Unit)).S ≤ 144 := by
    rw [card_faultChannel]
    calc patternCount (WeightSecurity.ToyServed.pop.faultCount (V := Fin 16)) 2 0 *
          Fintype.card (servedChannel Unit Unit Unit Unit (Fin 16) Unit 2 ((0 + 1) * Fintype.card Unit)).S
        ≤ 1 * 144 := Nat.mul_le_mul ToyTied.patternCount_le_one
          ((card_servedChannel_le Unit Unit Unit Unit (Fin 16) Unit 2 _).trans (by simp [Finset.sum_range_succ]))
      _ = 144 := one_mul _
  simp only [hsteal, avg_const, htyp, ENNReal.ofReal_zero, zero_add, hN, one_mul, add_zero] at h
  have hlt : ENNReal.ofReal ((Fintype.card (faultChannel Unit Unit Unit Unit (Fin 16) Unit
      (WeightSecurity.ToyServed.pop.Faults (Fin 16)) WeightSecurity.ToyServed.pop.faultCount 2 0
        ((0 + 1) * Fintype.card Unit)).S : ℝ) * (1 / 256)) < 1 := by
    rw [ENNReal.ofReal_lt_one]
    have : (Fintype.card (faultChannel Unit Unit Unit Unit (Fin 16) Unit
        (WeightSecurity.ToyServed.pop.Faults (Fin 16)) WeightSecurity.ToyServed.pop.faultCount 2 0
          ((0 + 1) * Fintype.card Unit)).S : ℝ) ≤ 144 := by
      exact_mod_cast hc
    linarith
  exact absurd h (not_le.2 hlt)

end SecurityProofs

end WeightSecurity
