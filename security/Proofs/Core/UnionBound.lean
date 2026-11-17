module

public import Definitions.Claims
public import Definitions.Core.Interaction
public import Definitions.Pouw.Basic.Prob
public import Proofs.Pouw.Prob
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
import Mathlib.Tactic.ENatToNat
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
import Mathlib.Topology.Sheaves.Init

/-! The union bound over a product of coins, under which the protocols may adapt to each other's coins. -/

@[expose] public section

namespace Pouw

open Finset

section Split

variable {X P R : Type} [Fintype X] [Fintype P] [Fintype R]

theorem pr_le_of_split (e : X ≃ P × R) (E : X → Prop) {δ : ℝ} (hδ : 0 ≤ δ)
    (h : ∀ p, pr (fun r => E (e.symm (p, r))) ≤ δ) : pr E ≤ δ := by
  rw [← pr_equiv e.symm E]
  exact pr_prod_le _ δ hδ h

theorem pr_split_fst_le (e : X ≃ P × R) (E : P → Prop) : pr (fun x => E (e x).1) ≤ pr E :=
  (pr_equiv e fun y => E y.1).trans_le (pr_fst_le E)

end Split

theorem Pr_eq_pr {α : Type} [Fintype α] (E : α → Prop) : Claims.Pr E = pr E := rfl

section Product

variable {I : Type} [Fintype I] [DecidableEq I] {Ω : I → Type} [∀ i, Fintype (Ω i)]

/-- The coins split at the coordinate `i`: the others, and `i`'s. -/
def splitAt (i : I) : ((j : I) → Ω j) ≃ ((j : { j // j ≠ i }) → Ω j) × Ω i :=
  (Equiv.piSplitAt i Ω).trans (Equiv.prodComm _ _)

omit [Fintype I] [∀ i, Fintype (Ω i)] in
theorem splitAt_symm_eq_update (i : I) (g : (j : { j // j ≠ i }) → Ω j) (x x₀ : Ω i) :
    (splitAt i).symm (g, x) = Function.update ((splitAt i).symm (g, x₀)) i x := by
  funext j
  by_cases hj : j = i
  · subst hj
    simp [splitAt, Equiv.piSplitAt_symm_apply]
  · simp [splitAt, Equiv.piSplitAt_symm_apply, hj]

theorem pr_le_of_update [∀ i, Nonempty (Ω i)] (i : I) (E : ((j : I) → Ω j) → Prop) {δ : ℝ}
    (h : ∀ ξ, pr (fun x : Ω i => E (Function.update ξ i x)) ≤ δ) : pr E ≤ δ := by
  refine pr_le_of_split (splitAt i) E ((pr_nonneg _).trans (h (Classical.arbitrary _))) fun g => ?_
  obtain ⟨x₀⟩ : Nonempty (Ω i) := inferInstance
  have key : (fun x => E ((splitAt i).symm (g, x))) =
      fun x => E (Function.update ((splitAt i).symm (g, x₀)) i x) :=
    funext fun x => by rw [splitAt_symm_eq_update i g x x₀]
  rw [key]
  exact h _

theorem union_bound_update {A : Type} [Fintype A] [∀ i, Nonempty (Ω i)] (own : A → I)
    (Bad : A → ((i : I) → Ω i) → Prop) (δ : A → ℝ)
    (h : ∀ a ξ, pr (fun x : Ω (own a) => Bad a (Function.update ξ (own a) x)) ≤ δ a) :
    pr (fun ξ => ∃ a, Bad a ξ) ≤ ∑ a, δ a :=
  (pr_mono fun _ ⟨a, ha⟩ => ⟨a, mem_univ a, ha⟩).trans <|
    (Proofs.pr_exists_le_sum univ Bad).trans (sum_le_sum fun a _ => pr_le_of_update (own a) (Bad a) (h a))

theorem union_bound_product [∀ i, Nonempty (Ω i)] (Bad : (i : I) → Ω i → Prop) (δ : I → ℝ)
    (h : ∀ i, pr (Bad i) ≤ δ i) :
    pr (fun ξ : (i : I) → Ω i => ∃ i, Bad i (ξ i)) ≤ ∑ i, δ i :=
  union_bound_update id (fun i ξ => Bad i (ξ i)) δ fun i ξ => by simpa using h i

theorem union_bound_monitoringPeriod {J A : Type} [Fintype J] [Fintype A] [∀ i, Nonempty (Ω i)] (own : J → A → I)
    (Bad : J → A → ((i : I) → Ω i) → Prop) (δ : J → A → ℝ)
    (h : ∀ j a ξ, pr (fun x : Ω (own j a) => Bad j a (Function.update ξ (own j a) x)) ≤ δ j a) :
    pr (fun ξ => ∃ j a, Bad j a ξ) ≤ ∑ j, ∑ a, δ j a := by
  have := union_bound_update (fun q : J × A => own q.1 q.2) (fun q => Bad q.1 q.2) (fun q => δ q.1 q.2)
    fun q => h q.1 q.2
  rw [Fintype.sum_prod_type] at this
  exact (pr_mono fun _ ⟨j, a, hb⟩ => ⟨(j, a), hb⟩).trans this

theorem union_bound_monitoringPeriod_with {J A : Type} [Fintype J] [Fintype A] [∀ i, Nonempty (Ω i)]
    (F : ((i : I) → Ω i) → Prop) {η : ℝ} (hF : pr F ≤ η) (own : J → A → I)
    (Bad : J → A → ((i : I) → Ω i) → Prop) (δ : J → A → ℝ)
    (h : ∀ j a ξ, pr (fun x : Ω (own j a) => Bad j a (Function.update ξ (own j a) x)) ≤ δ j a) :
    pr (fun ξ => F ξ ∨ ∃ j a, Bad j a ξ) ≤ η + ∑ j, ∑ a, δ j a :=
  (pr_or_le _ _).trans (add_le_add hF (union_bound_monitoringPeriod own Bad δ h))

theorem pr_restrict_le (S : I → Prop) [DecidablePred S] (E : ((i : { i // S i }) → Ω i) → Prop) :
    pr (fun ξ : (i : I) → Ω i => E fun i => ξ i) ≤ pr E :=
  pr_split_fst_le (Equiv.piEquivPiSubtypeProd S Ω) E

end Product

section Past

variable {I : Type} [Fintype I] [LinearOrder I] {Ω : I → Type} [∀ i, Fintype (Ω i)]

/-- The coins split before `t`: the coins before `t`, and `t`'s and the later ones. -/
def splitPast (t : I) : ((i : I) → Ω i) ≃ ((i : { i // i < t }) → Ω i) × ((i : { i // ¬ i < t }) → Ω i) :=
  Equiv.piEquivPiSubtypeProd (· < t) Ω

theorem pr_le_of_past [∀ i, Nonempty (Ω i)] (t : I) (E : ((i : I) → Ω i) → Prop) {δ : ℝ}
    (h : ∀ past, pr (fun rest => E ((splitPast t).symm (past, rest))) ≤ δ) : pr E ≤ δ :=
  pr_le_of_split (splitPast t) E ((pr_nonneg _).trans (h (Classical.arbitrary _))) h

theorem union_bound_past [∀ i, Nonempty (Ω i)] (Bad : I → ((i : I) → Ω i) → Prop) (δ : I → ℝ)
    (h : ∀ t past, pr (fun rest => Bad t ((splitPast t).symm (past, rest))) ≤ δ t) :
    pr (fun ξ => ∃ t, Bad t ξ) ≤ ∑ t, δ t :=
  (pr_mono fun _ ⟨t, ht⟩ => ⟨t, mem_univ t, ht⟩).trans <|
    (Proofs.pr_exists_le_sum univ Bad).trans (sum_le_sum fun t _ => pr_le_of_past t (Bad t) (h t))

end Past

end Pouw
