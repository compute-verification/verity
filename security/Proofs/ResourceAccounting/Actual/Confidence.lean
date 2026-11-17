module

public import Definitions.Pouw.Guarantees.Confidence
public import Proofs.ResourceAccounting.Actual.Split
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Rat.Cast.Order
import Mathlib.Tactic.Linarith.Frontend
import Mathlib.Tactic.NormNum.Abs
import Mathlib.Tactic.NormNum.DivMod
import Mathlib.Tactic.NormNum.OfScientific

/-! `Composition` and `Fit`, proved, with the facts on a location's state and content they read. -/

@[expose] public section

namespace ResourceAccounting.SecurityProofs

open Finset

namespace Actual

theorem state_zero {L : Type*} (R : Actual.Run L) (X : Finset L) : R.seen '' ↑(R.state X 0) = R.inputs ↑X := by
  classical
  ext p
  simp only [Actual.Run.state, coe_filter, mem_univ, true_and, Set.mem_image, Set.mem_ofPred_eq,
    Actual.Run.inputs]
  constructor
  · rintro ⟨a, ⟨y, hy, hay⟩ | ⟨hin, hx⟩, rfl⟩
    · obtain ⟨hxy, hi | hlt⟩ := R.holding.1 y 0 a hay
      · exact ⟨a, hi, by rw [hxy]; exact hy, rfl⟩
      · exact absurd hlt (Nat.not_lt_zero _)
    · refine ⟨a, ?_, hx, rfl⟩
      by_contra hi
      exact ((mem_into _).1 hin).1 ((mem_region _).2 ⟨by simpa using hi, hx, Set.mem_Ici.2 (Nat.zero_le _)⟩)
  · rintro ⟨a, hi, hx, rfl⟩
    exact ⟨a, Or.inl ⟨R.A.x a, hx, R.holding.2.1 a hi⟩, rfl⟩

theorem content_congr {L : Type*} [LinearOrder L] {R R' : Actual.Run L} {X : Finset L} {t : ℕ}
    (h : R.seen '' ↑(R.state X t) = R'.seen '' ↑(R'.state X t)) : R.content X t = R'.content X t := by
  have : (R.state X t).image R.seen = (R'.state X t).image R'.seen := coe_injective (by simpa only [coe_image] using h)
  unfold Actual.Run.content
  rw [this]

theorem content_length_le {L : Type*} [LinearOrder L] (R : Actual.Run L) (X : Finset L) (t : ℕ) :
    (R.content X t).length ≤ R.mem X t := by
  unfold Actual.Run.content Actual.Run.mem Verity.Circuit.width
  rw [List.length_flatMap, ((sort_perm_toList _ _).map _).sum_eq, sum_map_toList]
  refine (sum_image_le_of_nonneg fun _ _ => Nat.zero_le _).trans <|
    (sum_image_le_of_nonneg fun _ _ => Nat.zero_le _).trans (le_of_eq (sum_congr rfl fun g _ => ?_))
  simp [Actual.chunk, Actual.position, Actual.bits, Actual.Run.seen]

theorem causalFrom_comp {Ω Ω' L : Type*} [LinearOrder L] {S : Ω → Actual.Run L} (h : CausalFrom S) (f : Ω' → Ω) :
    CausalFrom fun x => S (f x) :=
  fun X t u r r' => h X t u (f r) (f r')

theorem global_of_causal {Ω L : Type*} [LinearOrder L] {S : Ω → Actual.Run L} (h : CausalFrom S) (P : Finset L)
    (hP : ∀ r r', (S r).inputs ↑P = (S r').inputs ↑P) (t : ℕ) (r r' : Ω)
    (hr : (S r).crossing (↑P)ᶜ ↑P (Set.Iio t) = (S r').crossing (↑P)ᶜ ↑P (Set.Iio t)) :
    (S r).acts P (Set.Iio t) = (S r').acts P (Set.Iio t) := by
  have hio : Set.Iio t = Set.Ico 0 t := by ext; simp
  rw [hio] at hr ⊢
  exact h P 0 t r r' (content_congr (by rw [state_zero, state_zero, hP r r'])) hr

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
  intro L _ k Ω _ _ 𝓕 V acc E δ S hE h𝓕 hS hV
  have bad : ∀ r, (∀ i, acc i ((S r).view (V i))) → ¬(∀ i, E i (fun x => S (Function.update r i x)) (r i)) →
      ∃ i, acc i ((S r).view (V i)) ∧ ¬E i (fun x => S (Function.update r i x)) (r i) := fun r ha hc => by
    push Not at hc
    obtain ⟨i, hi⟩ := hc
    exact ⟨i, ha i, hi⟩
  have each : ∀ i, ((univ.filter fun r => acc i ((S r).view (V i)) ∧
      ¬E i (fun x => S (Function.update r i x)) (r i)).card : ℝ) ≤ δ i * Fintype.card ((j : Fin k) → Ω j) :=
      fun i => by
    have hs : ∀ r, ((univ.filter fun x => acc i ((S (Function.update r i x)).view (V i)) ∧
        ¬E i (fun x' => S (Function.update r i x')) x).card : ℝ) ≤ δ i * Fintype.card (Ω i) := fun r =>
      hE i (fun x => S (Function.update r i x)) (fun x => h𝓕 _) (Actual.causalFrom_comp hS _)
        (fun x x' => hV i _ _ fun j hj => by simp [Function.update_of_ne hj])
    rcases (Fintype.card (Ω i)).eq_zero_or_pos with h0 | hpos
    · have : IsEmpty ((j : Fin k) → Ω j) := by
        have := Fintype.card_eq_zero_iff.1 h0
        exact ⟨fun r => this.false (r i)⟩
      simp
    have key := Actual.card_filter_mul i (fun r => acc i ((S r).view (V i)) ∧
      ¬E i (fun x => S (Function.update r i x)) (r i))
    simp only [Function.update_idem, Function.update_self] at key
    have hsum := Finset.sum_le_sum fun r (_ : r ∈ univ) => hs r
    rw [← Nat.cast_sum, ← key, sum_const, card_univ, nsmul_eq_mul, Nat.cast_mul] at hsum
    have hp : (0 : ℝ) < Fintype.card (Ω i) := by exact_mod_cast hpos
    nlinarith
  calc ((univ.filter fun r => (∀ i, acc i ((S r).view (V i))) ∧
        ¬∀ i, E i (fun x => S (Function.update r i x)) (r i)).card : ℝ)
      ≤ ((univ.biUnion fun i => univ.filter fun r => acc i ((S r).view (V i)) ∧
          ¬E i (fun x => S (Function.update r i x)) (r i)).card : ℝ) := by
        exact_mod_cast card_le_card fun r hr => by
          obtain ⟨i, hi⟩ := bad r (mem_filter.1 hr).2.1 (mem_filter.1 hr).2.2
          exact mem_biUnion.2 ⟨i, mem_univ _, mem_filter.2 ⟨mem_univ _, hi⟩⟩
    _ ≤ ∑ i, ((univ.filter fun r => acc i ((S r).view (V i)) ∧
          ¬E i (fun x => S (Function.update r i x)) (r i)).card : ℝ) := by
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

end ResourceAccounting.SecurityProofs
