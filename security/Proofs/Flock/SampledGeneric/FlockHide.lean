module

public import Proofs.Flock.Soundness.ZK.RealLeaves
public import Proofs.Flock.Soundness.Discharge.ZkCoins.Leaf

/-! hm96's hiding commitment to `N` leaves, by a hybrid over the leaves. -/

@[expose] public section

namespace FlockSoundness.SampledGeneric

open Game ZK Discharge.ZkCoins
open scoped ENNReal

theorem prCoin_unit_prod {Ω : Type} [Fintype Ω] (P : Ω → Prop) :
    prCoin (fun p : Unit × Ω => P p.2) = prCoin P :=
  (prCoin_prod_split fun (_ : Unit) (y : Ω) => P y).trans (avg_const _)

variable {ι Y B C D : Type} [Fintype ι] [DecidableEq ι] [Fintype Y] [Nonempty Y] [AddCommGroup B] [Fintype B]

theorem hm96_commit_ideal_close (My : Y → B) (c : Y → C) {δ₁ : ℝ≥0∞} (hT1 : Claims.Hm96Hiding My c δ₁)
    (H : B × C → D) (X : ι → B) (Γ : (ι → D) → ℝ≥0∞) (hΓ : ∀ d, Γ d ≤ 1) :
    avg (fun y : ι → Y => Γ fun i => H (X i + My (y i), c (y i))) ≤
        avg (fun u : ι → B × Y => Γ fun i => H ((u i).1, c (u i).2)) + Fintype.card ι * δ₁ ∧
      avg (fun u : ι → B × Y => Γ fun i => H ((u i).1, c (u i).2)) ≤
        avg (fun y : ι → Y => Γ fun i => H (X i + My (y i), c (y i))) + Fintype.card ι * δ₁ := by
  classical
  have : Nonempty B := ⟨0⟩
  let g : (ι → Y) → ι → D := fun y i => H (X i + My (y i), c (y i))
  let g' : (ι → B × Y) → ι → D := fun u i => H ((u i).1, c (u i).2)
  let S : Finset (ι → D) := Finset.univ.image g ∪ Finset.univ.image g'
  have hS : ∀ y, g y ∈ S := fun y => Finset.mem_union_left _ (Finset.mem_image_of_mem g (Finset.mem_univ y))
  have hS' : ∀ u, g' u ∈ S := fun u => Finset.mem_union_right _ (Finset.mem_image_of_mem g' (Finset.mem_univ u))
  have toIdeal : ∀ P : (ι → D) → Prop,
      prCoin (fun y => P (g y)) ≤ prCoin (fun u => P (g' u)) + Fintype.card ι * δ₁ := fun P =>
    calc prCoin (fun y => P (g y))
        = prCoin (fun p : Unit × (ι → Y) => P (g p.2)) := (prCoin_unit_prod fun y => P (g y)).symm
      _ ≤ prCoin (fun p : Unit × (ι → B × Y) => P (g' p.2)) + Fintype.card ι * δ₁ :=
          ideal_leaves_swap_fintype My c hT1 H (Ω := Unit) (fun _ => X) (fun _ L => P L)
      _ = prCoin (fun u => P (g' u)) + Fintype.card ι * δ₁ := by rw [prCoin_unit_prod fun u => P (g' u)]
  have toReal : ∀ P : (ι → D) → Prop,
      prCoin (fun u => P (g' u)) ≤ prCoin (fun y => P (g y)) + Fintype.card ι * δ₁ := fun P =>
    calc prCoin (fun u => P (g' u))
        = prCoin (fun p : Unit × (ι → B × Y) => P (g' p.2)) := (prCoin_unit_prod fun u => P (g' u)).symm
      _ ≤ prCoin (fun p : Unit × (ι → Y) => P (g p.2)) + Fintype.card ι * δ₁ :=
          real_leaves_swap_fintype My c hT1 H (Ω := Unit) (fun _ => X) (fun _ L => P L)
      _ = prCoin (fun y => P (g y)) + Fintype.card ι * δ₁ := by rw [prCoin_unit_prod fun y => P (g y)]
  exact ⟨avg_le_of_prCoin g g' S hS hS' toIdeal Γ hΓ, avg_le_of_prCoin g' g S hS' hS toReal Γ hΓ⟩

/-- The `2` is needed (`hm96_commit_two_traces_not_card`). -/
theorem hm96_commit_two_traces (My : Y → B) (c : Y → C) {δ₁ : ℝ≥0∞} (hT1 : Claims.Hm96Hiding My c δ₁)
    (H : B × C → D) (X X' : ι → B) (Γ : (ι → D) → ℝ≥0∞) (hΓ : ∀ d, Γ d ≤ 1) :
    avg (fun y : ι → Y => Γ fun i => H (X i + My (y i), c (y i))) ≤
      avg (fun y : ι → Y => Γ fun i => H (X' i + My (y i), c (y i))) + 2 * Fintype.card ι * δ₁ :=
  calc avg (fun y : ι → Y => Γ fun i => H (X i + My (y i), c (y i)))
      ≤ avg (fun u : ι → B × Y => Γ fun i => H ((u i).1, c (u i).2)) + Fintype.card ι * δ₁ :=
        (hm96_commit_ideal_close My c hT1 H X Γ hΓ).1
    _ ≤ (avg (fun y : ι → Y => Γ fun i => H (X' i + My (y i), c (y i))) + Fintype.card ι * δ₁) +
          Fintype.card ι * δ₁ :=
        add_le_add (hm96_commit_ideal_close My c hT1 H X' Γ hΓ).2 le_rfl
    _ = avg (fun y : ι → Y => Γ fun i => H (X' i + My (y i), c (y i))) + 2 * Fintype.card ι * δ₁ := by
        rw [add_assoc, ← two_mul, ← mul_assoc]

theorem hm96Hiding_const_half : Claims.Hm96Hiding (fun _ : Unit => (0 : ZMod 2)) (fun _ : Unit => ()) (1 / 2) := by
  intro E
  by_cases hE : E (0, ())
  · have hc : (Fintype.card (ZMod 2 × Unit) : ℝ≥0∞) = 2 := by simp [ZMod.card]
    have hsum : (2 : ℝ≥0∞)⁻¹ ≤ Claims.PrENNReal (fun p : ZMod 2 × Unit => E (p.1, ())) := by
      unfold Claims.PrENNReal
      calc (2 : ℝ≥0∞)⁻¹
          = (Fintype.card (ZMod 2 × Unit) : ℝ≥0∞)⁻¹ * @ite _ (E (0, ())) (Classical.propDecidable _) 1 0 := by
            rw [hc, ite_eq_left hE, mul_one]
        _ ≤ _ := Finset.single_le_sum (f := fun p : ZMod 2 × Unit => (Fintype.card (ZMod 2 × Unit) : ℝ≥0∞)⁻¹ *
              @ite _ (E (p.1, ())) (Classical.propDecidable _) 1 0) (fun _ _ => zero_le) (Finset.mem_univ (0, ()))
    calc Claims.PrENNReal (fun _ : Unit => E (0, ()))
        = 1 := by simp [Claims.PrENNReal, hE]
      _ = 2⁻¹ + 1 / 2 := by rw [one_div, ENNReal.inv_two_add_inv_two]
      _ ≤ _ := add_le_add hsum le_rfl
  · calc Claims.PrENNReal (fun _ : Unit => E (0, ()))
        = 0 := by simp [Claims.PrENNReal, hE]
      _ ≤ _ := zero_le

theorem hm96_commit_two_traces_not_card :
    ¬ ∀ (ι Y B C D : Type) [Fintype ι] [DecidableEq ι] [Fintype Y] [Nonempty Y] [AddCommGroup B] [Fintype B]
      (My : Y → B) (c : Y → C) (δ₁ : ℝ≥0∞), Claims.Hm96Hiding My c δ₁ →
      ∀ (H : B × C → D) (X X' : ι → B) (Γ : (ι → D) → ℝ≥0∞), (∀ d, Γ d ≤ 1) →
        avg (fun y : ι → Y => Γ fun i => H (X i + My (y i), c (y i))) ≤
          avg (fun y : ι → Y => Γ fun i => H (X' i + My (y i), c (y i))) + Fintype.card ι * δ₁ := by
  intro h
  have h10 : (1 : ZMod 2) ≠ 0 := by decide
  let Γ : (Unit → ZMod 2 × Unit) → ℝ≥0∞ := fun d => if (d ()).1 = 0 then 1 else 0
  have hΓ : ∀ d, Γ d ≤ 1 := fun d => by by_cases hd : (d ()).1 = 0 <;> simp [Γ, hd]
  have e : ∀ x : ZMod 2, avg (fun _ : Unit → Unit => Γ fun _ => id (x + 0, ())) = if x = 0 then 1 else 0 := by
    intro x
    by_cases hx : x = 0 <;> simp [avg, Γ, hx]
  have h0 := h Unit Unit (ZMod 2) Unit (ZMod 2 × Unit) (fun _ => 0) (fun _ => ()) (1 / 2) hm96Hiding_const_half id
    (fun _ => 0) (fun _ => 1) Γ hΓ
  rw [e, e, ite_eq_left rfl, ite_eq_right h10, zero_add, Fintype.card_unit, Nat.cast_one, one_mul] at h0
  exact absurd h0 (not_le.2 (ENNReal.half_lt_self one_ne_zero ENNReal.one_ne_top))

end FlockSoundness.SampledGeneric
