import Models.Accounting.Guarantees.Confidence
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Data.Fintype.Pi
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic.Linarith

/-!
# Composition and fit, proved

`Composition`: a run outside the confidence set fails some `E i`, so the event is in the union over `i` of
"`acc i` and `¬E i`". Fixing every coin but protocol `i`'s leaves a strategy against protocol `i` alone (`slice`): the
same runs, so feasible, still locally causal, and its input gates off `V i` fixed. Averaging over the fixed coins
(`card_filter_mul`: each run of coins is counted once for each value of coin `i`) gives `δ i`, and the union bound the
sum. `global_of_local` is the global form of causality: the prover's gates before `t` depend on the coins only through
what the prover reads from outside before `t`, when its inputs don't. `Fit` is additivity at the run that implements `P`.
-/

namespace Accounting.SecurityProofs

open Finset

namespace Actual

/-- Causality's global form: when the prover's (at `P`) input gates don't depend on the coins, what it does before `t`
depends on them only through what it reads from outside before `t`. -/
theorem global_of_local {Ω L : Type*} {S : Ω → Run L} (h : LocallyCausal S) (P : Finset L)
    (hP : ∀ r r', (S r).inputs P = (S r').inputs P) (t : ℕ) (r r' : Ω) (hr : (S r).heard P t = (S r').heard P t) :
    (S r).before P t = (S r').before P t :=
  h P t r r' (hP r r') hr

/-- Each run of the coins `r` is `update r' i x` for one `r'` per value `x`. -/
theorem card_filter_mul {k : ℕ} {Ω : Fin k → Type} [∀ i, Fintype (Ω i)] [∀ i, DecidableEq (Ω i)] (i : Fin k)
    (p : ((j : Fin k) → Ω j) → Prop) [DecidablePred p] :
    (univ.filter p).card * Fintype.card (Ω i) = ∑ r, (univ.filter fun x => p (Function.update r i x)).card := by
  let f : ((j : Fin k) → Ω j) × Ω i → ((j : Fin k) → Ω j) × Ω i := fun q => (Function.update q.1 i q.2, q.1 i)
  have hf : Function.Involutive f := fun q => by simp [f]
  have := Equiv.sum_comp hf.toPerm (fun q => if p q.1 then 1 else 0)
  simp only [Function.Involutive.coe_toPerm, f] at this
  simp only [card_filter, Fintype.sum_prod_type] at this ⊢
  calc (∑ r, if p r then 1 else 0) * Fintype.card (Ω i) = ∑ r, ∑ _x : Ω i, if p r then 1 else 0 := by
        rw [sum_mul]; simp [mul_comm]
    _ = _ := by convert this.symm; exact if_congr Iff.rfl rfl rfl

end Actual

open Actual in
theorem Composition : Guarantees.Composition := by
  classical
  intro L k Ω _ _ 𝓕 V W acc E δ S hE h𝓕 hS hV
  have bad : ∀ r, (∀ i, acc i ((S r).view (V i))) → S r ∉ confidence 𝓕 W E ((S r).view W) →
      ∃ i, acc i ((S r).view (V i)) ∧ ¬E i (S r) := fun r ha hc => by
    by_contra hn
    push Not at hn
    exact hc ⟨h𝓕 r, rfl, fun i => hn i (ha i)⟩
  have each : ∀ i, ((univ.filter fun r => acc i ((S r).view (V i)) ∧ ¬E i (S r)).card : ℝ) ≤
      δ i * Fintype.card ((j : Fin k) → Ω j) := fun i => by
    have hs : ∀ r, ((univ.filter fun x => acc i ((S (Function.update r i x)).view (V i)) ∧
        ¬E i (S (Function.update r i x))).card : ℝ) ≤ δ i * Fintype.card (Ω i) := fun r =>
      hE i (fun x => S (Function.update r i x)) (fun x => h𝓕 _) (fun X t x x' => hS X t _ _)
        (fun x x' => hV i _ _ fun j hj => by simp [Function.update_of_ne hj])
    rcases (Fintype.card (Ω i)).eq_zero_or_pos with h0 | hpos
    · have : IsEmpty ((j : Fin k) → Ω j) := by
        have := Fintype.card_eq_zero_iff.1 h0
        exact ⟨fun r => this.false (r i)⟩
      simp
    have key := card_filter_mul i (fun r => acc i ((S r).view (V i)) ∧ ¬E i (S r))
    have hsum := Finset.sum_le_sum fun r (_ : r ∈ univ) => hs r
    rw [← Nat.cast_sum, ← key, sum_const, card_univ, nsmul_eq_mul, Nat.cast_mul] at hsum
    have hp : (0 : ℝ) < Fintype.card (Ω i) := by exact_mod_cast hpos
    nlinarith
  calc ((univ.filter fun r => (∀ i, acc i ((S r).view (V i))) ∧
        S r ∉ confidence 𝓕 W E ((S r).view W)).card : ℝ)
      ≤ ((univ.biUnion fun i => univ.filter fun r => acc i ((S r).view (V i)) ∧ ¬E i (S r)).card : ℝ) := by
        exact_mod_cast card_le_card fun r hr => by
          obtain ⟨i, hi⟩ := bad r (mem_filter.1 hr).2.1 (mem_filter.1 hr).2.2
          exact mem_biUnion.2 ⟨i, mem_univ _, mem_filter.2 ⟨mem_univ _, hi⟩⟩
    _ ≤ ∑ i, ((univ.filter fun r => acc i ((S r).view (V i)) ∧ ¬E i (S r)).card : ℝ) := by
        exact_mod_cast card_biUnion_le
    _ ≤ (∑ i, δ i) * Fintype.card ((j : Fin k) → Ω j) := by
        rw [sum_mul]
        exact sum_le_sum fun i _ => each i

theorem Fit : Guarantees.Fit := by
  intro L 𝓕 𝓒 μ impl cert cap obs hA ⟨R, hR, hi⟩ hcap
  have h := hA R hR hi
  obtain ⟨hc, ho⟩ := hcap R hR
  have hle : (μ R : ℕ∞) ≤ (min cap obs : ℕ) := by exact_mod_cast le_min hc ho
  generalize req μ 𝓕 impl = x at h ⊢
  induction x using ENat.recTopCoe with
  | top => simp at h
  | coe n =>
    have : cert + n ≤ min cap obs := by exact_mod_cast h.trans hle
    exact_mod_cast (by omega : n ≤ min cap obs - cert)

end Accounting.SecurityProofs
