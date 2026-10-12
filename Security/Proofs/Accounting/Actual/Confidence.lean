import Models.Accounting.Guarantees.Confidence
import Proofs.Accounting.Actual.Split
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Data.Fintype.Pi
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic.Linarith

/-!
# Composition and fit, proved

`Composition`: a run outside the confidence set fails some `E i`, so the event is in the union over `i` of
"`acc i` and `¬E i`". Fixing every coin but protocol `i`'s leaves a strategy against protocol `i` alone: the same runs,
so feasible, still admissible (a restriction of a restriction), and its input gates off `V i` fixed. Averaging over the
fixed coins (`card_filter_mul`: each run of coins is counted once for each value of coin `i`) gives `δ i`, and the
union bound the sum. `stateOver_congr`: the state over continuations is a function of what the location can tell.
`global_of_admissible` is the global form of causality: what the prover does before `t` depends on the coins only
through what it reads from outside before `t`, when its inputs don't (causality from tick 0, where it can tell only its
inputs, `history_zero`). `Fit` is additivity at the run that implements `P`.
-/

namespace Accounting.SecurityProofs

open Finset

namespace Actual

/-- At tick 0 a location can tell only its input gates. -/
theorem history_zero {L : Type*} {R R' : Actual.Run L} {X : Finset L} (h : R.inputs ↑X = R'.inputs ↑X) :
    R.history X 0 = R'.history X 0 := by
  refine Prod.ext h (Prod.ext ?_ ?_) <;> ext p <;>
    simp [Actual.Run.history, Actual.Run.before, Actual.Run.crossing]

/-- The state over continuations is a function of what the location can tell: runs alike in `history X t` hold the
same state (seen), since they have the same continuations and the same gates outside `A[X, [t, ∞)]`. -/
theorem stateOver_congr {Ω L : Type*} (S : Ω → Actual.Run L) {X : Finset L} {t : ℕ} {r r' : Ω}
    (h : (S r).history X t = (S r').history X t) :
    (S r).seen '' ↑(stateOver S X t r) = (S r').seen '' ↑(stateOver S X t r') := by
  classical
  suffices sub : ∀ r r', (S r).history X t = (S r').history X t →
      (S r).seen '' ↑(stateOver S X t r) ⊆ (S r').seen '' ↑(stateOver S X t r') from
    (sub r r' h).antisymm (sub r' r h.symm)
  intro r r' h
  rintro _ ⟨a, ha, rfl⟩
  simp only [stateOver, coe_filter, mem_univ, true_and, Set.mem_ofPred_eq] at ha
  obtain ⟨hx, hreg, r'', h'', a'', ha'', hs⟩ := ha
  have hin : (S r).seen a ∈ (S r).inputs ↑X ∪ (S r).before X t := by
    by_cases hi : (S r).A.isInput a = true
    · exact Or.inl ⟨a, hi, hx, rfl⟩
    · refine Or.inr ⟨a, hx, not_le.1 fun hle => hreg ((mem_region _).2 ⟨?_, hx, hle⟩), rfl⟩
      simpa using hi
  rw [show (S r).inputs ↑X = (S r').inputs ↑X from congrArg Prod.fst h,
    show (S r).before X t = (S r').before X t from congrArg (fun p => p.2.1) h] at hin
  have hr'' : (S r'').history X t = (S r').history X t := h''.trans h
  obtain ⟨b, hbi, hbx, hb⟩ | ⟨b, hbx, hbt, hb⟩ := hin
  · refine ⟨b, ?_, hb.symm⟩
    simp only [stateOver, coe_filter, mem_univ, true_and, Set.mem_ofPred_eq]
    exact ⟨hbx, fun hm => by simp [((mem_region _).1 hm).1] at hbi, r'', hr'', a'', ha'', hs.trans hb⟩
  · refine ⟨b, ?_, hb.symm⟩
    simp only [stateOver, coe_filter, mem_univ, true_and, Set.mem_ofPred_eq]
    exact ⟨hbx, fun hm => absurd ((mem_region _).1 hm).2.2 (not_le.2 hbt), r'', hr'', a'', ha'', hs.trans hb⟩

/-- Causality's global form: in an admissible strategy, when the prover's (at `P`) input gates don't depend on the
coins, what it does before `t` (its gates and its sends) depends on them only through what it reads from outside before
`t`. -/
theorem global_of_admissible {Ω L : Type*} {S : Ω → Actual.Run L} (h : Admissible S) (P : Finset L)
    (hP : ∀ r r', (S r).inputs ↑P = (S r').inputs ↑P) (t : ℕ) (r r' : Ω)
    (hr : (S r).crossing (↑P)ᶜ ↑P (Set.Iio t) = (S r').crossing (↑P)ᶜ ↑P (Set.Iio t)) :
    (S r).acts ↑P (Set.Iio t) = (S r').acts ↑P (Set.Iio t) := by
  obtain ⟨Ω', S', ι, hc, hS⟩ := h
  have hio : Set.Iio t = Set.Ico 0 t := by ext; simp
  have hi := hP r r'
  rw [hS r, hS r'] at hi hr ⊢
  rw [hio] at hr ⊢
  exact hc P 0 t _ _ (stateOver_congr S' (history_zero hi)) hr

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
      hE i (fun x => S (Function.update r i x)) (fun x => h𝓕 _)
        (let ⟨Ω', S', ι, hc, hι⟩ := hS; ⟨Ω', S', fun x => ι (Function.update r i x), hc, fun x => hι _⟩)
        (fun x x' => hV i _ _ fun j hj => by simp [Function.update_of_ne hj])
    rcases (Fintype.card (Ω i)).eq_zero_or_pos with h0 | hpos
    · have : IsEmpty ((j : Fin k) → Ω j) := by
        have := Fintype.card_eq_zero_iff.1 h0
        exact ⟨fun r => this.false (r i)⟩
      simp
    have key := Actual.card_filter_mul i (fun r => acc i ((S r).view (V i)) ∧ ¬E i (S r))
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
