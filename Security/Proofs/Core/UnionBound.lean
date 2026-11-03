import Proofs.Pouw.Prob
import Definitions.Claims

/-!
# The union bound over a product of coins

Every protocol decides on its own coins, and the coins of a period, or of a monitoring period's periods, are their
product: a uniform `ξ : (i : I) → Ω i`. A protocol's guarantee bounds its failure event by `δ` over its own coordinate,
at every value of the others (`Accounting.PeriodRun.Sound`'s form), or over its own and the later coordinates, at every
value of the earlier ones. Either way the event may read every coin, so the protocols may adapt to each other's coins.
The probability over the product that some protocol fails is then at most the sum of the `δ`s.

* `pr_le_of_split`: when the coins split as `P × R`, a bound over `R` at every value of `P` bounds the whole;
  `pr_le_of_update` is the split at one coordinate, `pr_le_of_past` at the coordinates before one.
* `union_bound_update`: each protocol `a` fails with at most `δ a` over its coordinate `own a`, at every value of the
  other coins; some protocol fails with at most `∑ δ a`. `union_bound_product` is the independent case, each failure
  event reading only its own coordinate.
* `union_bound_past`: the adaptive form, coins drawn in sequence, each step's failure at most `δ t` conditionally on
  the coins before it.
* `union_bound_monitoringPeriod`: over a monitoring period's periods, each period's protocols in
  `union_bound_update`'s form: `∑_j ∑_a δ j a`. `union_bound_monitoringPeriod_with` adds one more event bounded over the
  whole product, such as acceptance over a monitoring period (`FlockSoundness.Audit.Law.pooled_escape_log`, in
  `Claims.Pr`, which is `pr`: `Pr_eq_pr`); `pr_restrict_le` carries a bound proved over a block of the coordinates to the
  whole product.
-/

namespace Pouw

open Finset

section Split

variable {X P R : Type} [Fintype X] [Fintype P] [Fintype R]

/-- **A bound at every value of part of the coins.** When the coins split as `P × R` (`e`) and, at every value of the
`P` part, the probability over the `R` part is at most `δ ≥ 0`, so is the probability over all of them. -/
theorem pr_le_of_split (e : X ≃ P × R) (E : X → Prop) {δ : ℝ} (hδ : 0 ≤ δ)
    (h : ∀ p, pr (fun r => E (e.symm (p, r))) ≤ δ) : pr E ≤ δ := by
  rw [← pr_equiv e.symm E]
  exact pr_prod_le _ δ hδ h

/-- An event that reads only the `P` part of a split has at most its probability over `P`. -/
theorem pr_split_fst_le (e : X ≃ P × R) (E : P → Prop) : pr (fun x => E (e x).1) ≤ pr E :=
  (pr_equiv e fun y => E y.1).trans_le (pr_fst_le E)

end Split

/-- `Claims.Pr`, the claims' uniform probability, is `pr`. -/
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

/-- **A bound at one coordinate.** If, at every value of the other coins, the probability over coordinate `i` of `E`
is at most `δ`, so is the probability over all the coins. -/
theorem pr_le_of_update [∀ i, Nonempty (Ω i)] (i : I) (E : ((j : I) → Ω j) → Prop) {δ : ℝ}
    (h : ∀ ξ, pr (fun x : Ω i => E (Function.update ξ i x)) ≤ δ) : pr E ≤ δ := by
  refine pr_le_of_split (splitAt i) E ((pr_nonneg _).trans (h (Classical.arbitrary _))) fun g => ?_
  obtain ⟨x₀⟩ : Nonempty (Ω i) := inferInstance
  have key : (fun x => E ((splitAt i).symm (g, x))) =
      fun x => E (Function.update ((splitAt i).symm (g, x₀)) i x) :=
    funext fun x => by rw [splitAt_symm_eq_update i g x x₀]
  rw [key]
  exact h _

/-- **The union bound over the product of the protocols' coins.** Protocol `a` draws coordinate `own a` of the coins
and fails (`Bad a`, an event on all of them) with probability at most `δ a` over its coordinate, at every value of the
other coins. Then some protocol fails with probability at most `∑ a, δ a`. -/
theorem union_bound_update {A : Type} [Fintype A] [∀ i, Nonempty (Ω i)] (own : A → I)
    (Bad : A → ((i : I) → Ω i) → Prop) (δ : A → ℝ)
    (h : ∀ a ξ, pr (fun x : Ω (own a) => Bad a (Function.update ξ (own a) x)) ≤ δ a) :
    pr (fun ξ => ∃ a, Bad a ξ) ≤ ∑ a, δ a :=
  (pr_mono fun _ ⟨a, ha⟩ => ⟨a, mem_univ a, ha⟩).trans <|
    (Proofs.pr_exists_le_sum univ Bad).trans (sum_le_sum fun a _ => pr_le_of_update (own a) (Bad a) (h a))

/-- **The union bound over a product of independent coins.** Protocol `i` fails on its own coin with probability at
most `δ i`; under the product of the coins, some protocol fails on its coordinate with probability at most
`∑ i, δ i`. -/
theorem union_bound_product [∀ i, Nonempty (Ω i)] (Bad : (i : I) → Ω i → Prop) (δ : I → ℝ)
    (h : ∀ i, pr (Bad i) ≤ δ i) :
    pr (fun ξ : (i : I) → Ω i => ∃ i, Bad i (ξ i)) ≤ ∑ i, δ i :=
  union_bound_update id (fun i ξ => Bad i (ξ i)) δ fun i ξ => by simpa using h i

/-- **The union bound over a monitoring period's periods.** Period `j`'s protocol `a` draws coordinate `own j a` and
fails with probability at most `δ j a` over it, at every value of the other coins (so a period may adapt to every other
period's coins). Then some protocol of some period fails with probability at most `∑ j, ∑ a, δ j a`. -/
theorem union_bound_monitoringPeriod {J A : Type} [Fintype J] [Fintype A] [∀ i, Nonempty (Ω i)] (own : J → A → I)
    (Bad : J → A → ((i : I) → Ω i) → Prop) (δ : J → A → ℝ)
    (h : ∀ j a ξ, pr (fun x : Ω (own j a) => Bad j a (Function.update ξ (own j a) x)) ≤ δ j a) :
    pr (fun ξ => ∃ j a, Bad j a ξ) ≤ ∑ j, ∑ a, δ j a := by
  have := union_bound_update (fun q : J × A => own q.1 q.2) (fun q => Bad q.1 q.2) (fun q => δ q.1 q.2)
    fun q => h q.1 q.2
  rw [Fintype.sum_prod_type] at this
  exact (pr_mono fun _ ⟨j, a, hb⟩ => ⟨(j, a), hb⟩).trans this

/-- **A monitoring period with one more event**, bounded over the whole product (`pr F ≤ η`), such as every period's
draw accepting (`FlockSoundness.Audit.Law.pooled_escape_log`, carried over by `pr_restrict_le`): that event or some
protocol of some period failing has probability at most `η + ∑ j, ∑ a, δ j a`. -/
theorem union_bound_monitoringPeriod_with {J A : Type} [Fintype J] [Fintype A] [∀ i, Nonempty (Ω i)]
    (F : ((i : I) → Ω i) → Prop) {η : ℝ} (hF : pr F ≤ η) (own : J → A → I)
    (Bad : J → A → ((i : I) → Ω i) → Prop) (δ : J → A → ℝ)
    (h : ∀ j a ξ, pr (fun x : Ω (own j a) => Bad j a (Function.update ξ (own j a) x)) ≤ δ j a) :
    pr (fun ξ => F ξ ∨ ∃ j a, Bad j a ξ) ≤ η + ∑ j, ∑ a, δ j a :=
  (pr_or_le _ _).trans (add_le_add hF (union_bound_monitoringPeriod own Bad δ h))

/-- An event that reads only the coordinates in `S` has at most its probability over those coordinates. -/
theorem pr_restrict_le (S : I → Prop) [DecidablePred S] (E : ((i : { i // S i }) → Ω i) → Prop) :
    pr (fun ξ : (i : I) → Ω i => E fun i => ξ i) ≤ pr E :=
  pr_split_fst_le (Equiv.piEquivPiSubtypeProd S Ω) E

end Product

section Past

variable {I : Type} [Fintype I] [LinearOrder I] {Ω : I → Type} [∀ i, Fintype (Ω i)]

/-- The coins split before `t`: the coins before `t`, and `t`'s and the later ones. -/
def splitPast (t : I) : ((i : I) → Ω i) ≃ ((i : { i // i < t }) → Ω i) × ((i : { i // ¬ i < t }) → Ω i) :=
  Equiv.piEquivPiSubtypeProd (· < t) Ω

/-- **A bound conditioned on the past.** If, at every value of the coins before `t`, the probability over `t`'s and
the later coins of `E` is at most `δ`, so is the probability over all the coins. -/
theorem pr_le_of_past [∀ i, Nonempty (Ω i)] (t : I) (E : ((i : I) → Ω i) → Prop) {δ : ℝ}
    (h : ∀ past, pr (fun rest => E ((splitPast t).symm (past, rest))) ≤ δ) : pr E ≤ δ :=
  pr_le_of_split (splitPast t) E ((pr_nonneg _).trans (h (Classical.arbitrary _))) h

/-- **The adaptive union bound.** The coins are drawn in the order of `I`, and step `t` fails (`Bad t`, an event on
all the coins) with probability at most `δ t` over its own and the later coins, conditionally on the coins before it.
Then some step fails with probability at most `∑ t, δ t`. -/
theorem union_bound_past [∀ i, Nonempty (Ω i)] (Bad : I → ((i : I) → Ω i) → Prop) (δ : I → ℝ)
    (h : ∀ t past, pr (fun rest => Bad t ((splitPast t).symm (past, rest))) ≤ δ t) :
    pr (fun ξ => ∃ t, Bad t ξ) ≤ ∑ t, δ t :=
  (pr_mono fun _ ⟨t, ht⟩ => ⟨t, mem_univ t, ht⟩).trans <|
    (Proofs.pr_exists_le_sum univ Bad).trans (sum_le_sum fun t _ => pr_le_of_past t (Bad t) (h t))

end Past

end Pouw
