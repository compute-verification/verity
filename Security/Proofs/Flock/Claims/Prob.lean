import Definitions.Claims

/-!
# Uniform averages over finite coin spaces

Lemmas on `Claims.Pr` and `Claims.Ex`: bounds, congruence, events of some coordinates of a product, a product of
independent events, and the uniform round guess.
-/

namespace Claims

theorem Pr_nonneg {α : Type} [Fintype α] (p : α → Prop) : 0 ≤ Pr p := by
  unfold Pr; positivity

theorem Pr_le_one {α : Type} [Fintype α] [Nonempty α] (p : α → Prop) : Pr p ≤ 1 := by
  classical
  unfold Pr
  rw [div_le_one (by exact_mod_cast Fintype.card_pos)]
  exact_mod_cast Finset.card_filter_le _ _

theorem Pr_congr {α : Type} [Fintype α] {p q : α → Prop} (h : ∀ a, p a ↔ q a) : Pr p = Pr q := by
  have : p = q := funext fun a => propext (h a)
  subst this; rfl

theorem Pr_mono {α : Type} [Fintype α] {p q : α → Prop} (h : ∀ a, p a → q a) : Pr p ≤ Pr q := by
  classical
  unfold Pr
  refine div_le_div_of_nonneg_right ?_ (by positivity)
  exact_mod_cast Finset.card_le_card fun a ha => by simp only [Finset.mem_filter] at ha ⊢; exact ⟨ha.1, h a ha.2⟩

theorem Pr_false {α : Type} [Fintype α] : Pr (fun _ : α => False) = 0 := by simp [Pr]

/-- `Pr` as an average of indicators. -/
theorem Pr_eq_sum {α : Type} [Fintype α] (p : α → Prop) [DecidablePred p] :
    Pr p = (∑ a, if p a then (1 : ℝ) else 0) / Fintype.card α := by
  unfold Pr
  congr 1
  rw [Finset.sum_boole]
  congr 2
  exact Finset.filter_congr_decidable _ _ _

theorem Ex_nonneg {α : Type} [Fintype α] {f : α → ℝ} (h : ∀ a, 0 ≤ f a) : 0 ≤ Ex f := by
  unfold Ex
  exact div_nonneg (Finset.sum_nonneg fun a _ => h a) (by positivity)

theorem Ex_mono {α : Type} [Fintype α] {f g : α → ℝ} (h : ∀ a, f a ≤ g a) : Ex f ≤ Ex g := by
  unfold Ex
  exact div_le_div_of_nonneg_right (Finset.sum_le_sum fun a _ => h a) (by positivity)

theorem Ex_add {α : Type} [Fintype α] (f g : α → ℝ) : Ex (fun a => f a + g a) = Ex f + Ex g := by
  unfold Ex; rw [Finset.sum_add_distrib, add_div]

theorem Ex_div {α : Type} [Fintype α] (f : α → ℝ) (c : ℝ) : Ex (fun a => f a / c) = Ex f / c := by
  unfold Ex; rw [← Finset.sum_div]; ring

theorem Ex_const {α : Type} [Fintype α] [Nonempty α] (c : ℝ) : Ex (fun _ : α => c) = c := by
  unfold Ex
  have : (0 : ℝ) < Fintype.card α := by exact_mod_cast Fintype.card_pos
  rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  field_simp

/-- An event of the first coordinate has the same probability over a product. -/
theorem Pr_fst {α β : Type} [Fintype α] [Fintype β] [Nonempty β] (p : α → Prop) :
    Pr (fun s : α × β => p s.1) = Pr p := by
  classical
  rw [Pr_eq_sum (fun s : α × β => p s.1), Pr_eq_sum p, Fintype.sum_prod_type, Fintype.card_prod]
  have : (0 : ℝ) < Fintype.card β := by exact_mod_cast Fintype.card_pos
  have e : ∀ a : α, (∑ b : β, if p (a, b).1 then (1 : ℝ) else 0) = Fintype.card β * (if p a then 1 else 0) := by
    intro a; split_ifs <;> simp
  rw [Finset.sum_congr rfl fun a _ => e a, ← Finset.mul_sum]
  push_cast
  field_simp

/-- An event of the first two coordinates has the same probability over a triple. -/
theorem Pr_drop {α β γ : Type} [Fintype α] [Fintype β] [Fintype γ] [Nonempty γ] (p : α → β → Prop) :
    Pr (fun s : α × β × γ => p s.1 s.2.1) = Pr (fun s : α × β => p s.1 s.2) := by
  classical
  rw [Pr_eq_sum (fun s : α × β × γ => p s.1 s.2.1), Pr_eq_sum (fun s : α × β => p s.1 s.2), Fintype.sum_prod_type,
    Fintype.sum_prod_type, Fintype.card_prod, Fintype.card_prod, Fintype.card_prod]
  have : (0 : ℝ) < Fintype.card γ := by exact_mod_cast Fintype.card_pos
  have e : ∀ a b, (∑ c : γ, if p (a, b, c).1 (a, b, c).2.1 then (1 : ℝ) else 0) =
      Fintype.card γ * (if p (a, b).1 (a, b).2 then 1 else 0) := by
    intro a b; split_ifs <;> simp_all
  simp only [Fintype.sum_prod_type, e, ← Finset.mul_sum]
  push_cast
  field_simp

/-- **Independent coordinates**: an event of the first two coordinates and one of the first and third, over a triple,
has probability the average over the first of the product of the two conditional probabilities. -/
theorem Pr_indep {α β γ : Type} [Fintype α] [Fintype β] [Fintype γ] (p : α → β → Prop) (q : α → γ → Prop) :
    Pr (fun s : α × β × γ => p s.1 s.2.1 ∧ q s.1 s.2.2) = Ex fun a => Pr (p a) * Pr (q a) := by
  classical
  unfold Ex
  rw [Pr_eq_sum (fun s : α × β × γ => p s.1 s.2.1 ∧ q s.1 s.2.2), Fintype.sum_prod_type, Fintype.card_prod,
    Fintype.card_prod, Finset.sum_div, Finset.sum_div]
  refine Finset.sum_congr rfl fun a _ => ?_
  rw [Pr_eq_sum (p a), Pr_eq_sum (q a), div_mul_div_comm, Finset.sum_mul_sum, Fintype.sum_prod_type]
  push_cast
  rw [div_div]
  congr 1
  · refine Finset.sum_congr rfl fun b _ => Finset.sum_congr rfl fun c _ => ?_
    split_ifs <;> simp_all
  · ring

/-- **A product of independent events**: over `m` independent coordinates, `Pr[∀ i, q (ρ i)] = Pr[q]^m`. -/
theorem Pr_pi {δ : Type} [Fintype δ] (m : ℕ) (q : δ → Prop) :
    Pr (fun ρ : Fin m → δ => ∀ i, q (ρ i)) = Pr q ^ m := by
  classical
  rw [Pr_eq_sum (fun ρ : Fin m → δ => ∀ i, q (ρ i)), Pr_eq_sum q, div_pow, Fintype.card_fun, Fintype.card_fin]
  push_cast
  congr 1
  rw [← Fin.prod_const, Finset.prod_univ_sum]
  refine Finset.sum_congr (by simp) fun ρ _ => ?_
  rw [Finset.prod_boole]
  simp

/-- **The round guess**: `Pr[p]` over a uniform round `j` and the rest is the average over `j`. -/
theorem Pr_guess {k : ℕ} {β : Type} [Fintype β] (p : Fin k × β → Prop) :
    Pr p = (∑ r : Fin k, Pr fun b : β => p (r, b)) / k := by
  classical
  rw [Pr_eq_sum p, Fintype.sum_prod_type, Fintype.card_prod, Fintype.card_fin]
  simp only [Pr_eq_sum (fun b : β => p (_, b))]
  rw [← Finset.sum_div]
  push_cast
  rw [div_div, mul_comm (k : ℝ)]

end Claims
