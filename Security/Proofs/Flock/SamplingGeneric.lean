module

public import Definitions.Claims.SamplingGeneric
public import Proofs.Flock.SamplingGeneric.Wait
public import Proofs.Flock.SamplingGeneric.Split

/-!
# The sampled compiler in the claims core, at Flock

The compiled extractor's waits, the failure event's probability and `Compiled.KS` from the split are in
`SamplingGeneric/Wait.lean`.

`Arg.ofFlock` is Flock's `E^rep` at a decoder with Flock's errors, so its knowledge soundness is `Flock.KS`'s
(`flockArgKS`, by unfolding: its coins, extractor, `κ` and `γ` are `Flock.E`'s, `Flock.κ`'s and `Flock.γ`'s), and its
rounds are its shape's (`arg_ofFlock_wf`). So the compiler's knowledge soundness gives Flock's (`flockKS_of_compilerKS`);
it is two steps (`SamplingGeneric/Split.lean`: `KSOf`, `SessionKSOf`, composed by `compilerKS_of`).

Zero knowledge per outcome gives the leak form (`zkOutLeak`): an event's probability is the sum over the outcomes of the
outcome's and the event's (`pr_eq_sum_fiber`), each bounded by its simulator's, and the sum of the simulators' is the
number of outcomes times the simulator's that draws the outcome uniformly (`pr_prod_eq_sum`).
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Classical in
theorem pr_eq_sum_fiber {α St : Type} [Fintype α] [Fintype St] (f : α → St) (p : α → Prop) :
    Pr p = ∑ o, Pr (fun a => f a = o ∧ p a) := by
  have h : (Finset.univ.filter p).card = ∑ o, (Finset.univ.filter fun a => f a = o ∧ p a).card := by
    rw [Finset.card_eq_sum_card_fiberwise (f := f) (t := Finset.univ) fun _ _ => Finset.mem_univ _]
    refine Finset.sum_congr rfl fun o _ => ?_
    congr 1
    ext a
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    exact and_comm
  unfold Claims.Pr
  rw [h, Nat.cast_sum, Finset.sum_div]
  exact Finset.sum_congr rfl fun o _ => by congr

open Classical in
theorem pr_prod_eq_sum {St C : Type} [Fintype St] [Nonempty St] [Fintype C] (q : St → C → Prop) :
    (Fintype.card St : ℝ) * Pr (fun p : St × C => q p.1 p.2) = ∑ o, Pr (fun c => q o c) := by
  have h : (Finset.univ.filter fun p : St × C => q p.1 p.2).card =
      ∑ o, (Finset.univ.filter fun c => q o c).card := by
    rw [Finset.card_filter, Fintype.sum_prod_type]
    exact Finset.sum_congr rfl fun o _ => (Finset.card_filter _ _).symm
  have ha : (Fintype.card St : ℝ) ≠ 0 := Nat.cast_ne_zero.2 Fintype.card_ne_zero
  unfold Claims.Pr
  rw [h, Fintype.card_prod, Nat.cast_sum, Nat.cast_mul, ← Finset.sum_div, ← mul_div_assoc,
    mul_div_mul_left _ _ ha]

theorem zkOutLeak : ZKOutLeak := by
  intro W St _ _ N Ω _ _ _ _ H k P R out S ε h x w hR Θ Vs θ hk Ev
  rw [pr_eq_sum_fiber (fun ω => out x (runStar H (P w x) Vs x ω θ).msgs),
    pr_prod_eq_sum (fun o c => Ev ((S.given o).out H Vs θ x c)), ← Finset.sum_add_distrib]
  exact Finset.sum_le_sum fun o _ => h x w hR Θ Vs θ hk o Ev

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

omit [Inhabited Coin] [DecidableEq Col] in
theorem arg_ofFlock_wf (D : Claims.Flock.Deploy Coin Col T) (decode : List (ℕ × Col) → List T) :
    (Arg.ofFlock D decode).WF := fun _ => rfl

theorem flockArgKS : FlockArgKS (Coin := Coin) (Col := Col) (T := T) := by
  intro H D _ decode hKS hdec
  exact hKS decode hdec

theorem flockKS_of_compilerKS (h : CompilerKS) : FlockKS (Coin := Coin) (Col := Col) (T := T) := by
  intro H D _ decode L hKS hdec hB hS
  exact h Coin (Arg.ofFlock D decode) L H (arg_ofFlock_wf D decode) (flockArgKS H D decode hKS hdec) hB hS

end Claims.SamplingGeneric
