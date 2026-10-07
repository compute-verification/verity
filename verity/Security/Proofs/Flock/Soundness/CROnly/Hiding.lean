import Proofs.Flock.Soundness.CROnly.Hm96Defs
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Algebra.Order.Chebyshev

/-!
# hm96 hiding from the universal hash

`Assumptions.Hm96Hiding My c δ₁` holds at `δ₁ = hidingGap My c`, with no hypothesis (`hm96Hiding_gap`): every test's
advantage is at most the statistical distance of the two pairs. So `Hm96Hiding` is no assumption once `δ₁` is the gap,
and what is left is the gap's size at the pinned key.

The leftover hash lemma bounds it on average over the keys of a universal family (`sum_hidingGap_le`):
$$\mathbb E_k\,\mathrm{gap}(M_k, c)\le\tfrac12\sqrt{|B|\cdot|c(Y)|/|Y|},$$
by Cauchy–Schwarz over (key, pair) and the collision count `Σ_k Σ_v μ_k(v)² ≤ |K|/|Y| + |K|·Σ_v ν(v)²`. Markov's inequality
bounds the fraction of keys whose gap exceeds `ε` (`card_hidingGap_gt`).
-/

namespace FlockSoundness.CROnly

open Finset Game
open scoped ENNReal

/-- Two indicator sums of two distributions on `S` differ by at most half their ℓ¹ distance. -/
theorem sum_ind_le_tv {V : Type} (S : Finset V) (e : V → Prop) [DecidablePred e] (μ ν : V → ℝ)
    (h : ∑ v ∈ S, μ v = ∑ v ∈ S, ν v) :
    ∑ v ∈ S, (if e v then (1 : ℝ) else 0) * μ v ≤
      ∑ v ∈ S, (if e v then (1 : ℝ) else 0) * ν v + 1 / 2 * ∑ v ∈ S, |μ v - ν v| := by
  have hpt : ∀ v ∈ S, 2 * ((if e v then (1 : ℝ) else 0) * μ v) ≤
      2 * ((if e v then (1 : ℝ) else 0) * ν v) + |μ v - ν v| + (μ v - ν v) := fun v _ => by
    split_ifs
    · linarith [le_abs_self (μ v - ν v)]
    · linarith [neg_abs_le (μ v - ν v)]
  have hs := sum_le_sum hpt
  have e1 : ∑ v ∈ S, 2 * ((if e v then (1 : ℝ) else 0) * μ v) =
      2 * ∑ v ∈ S, (if e v then (1 : ℝ) else 0) * μ v := by rw [mul_sum]
  have e2 : ∑ v ∈ S, (2 * ((if e v then (1 : ℝ) else 0) * ν v) + |μ v - ν v| + (μ v - ν v)) =
      2 * ∑ v ∈ S, (if e v then (1 : ℝ) else 0) * ν v + ∑ v ∈ S, |μ v - ν v| +
        (∑ v ∈ S, μ v - ∑ v ∈ S, ν v) := by
    rw [sum_add_distrib, sum_add_distrib, sum_sub_distrib, mul_sum]
  rw [e1, e2, h, sub_self] at hs
  linarith

section Push

variable {Ω V : Type} [Fintype Ω] [DecidableEq V]

theorem sum_card_fiber_mul (g : Ω → V) (S : Finset V) (hS : ∀ x, g x ∈ S) (f : V → ℝ) :
    ∑ v ∈ S, (#{x | g x = v} : ℝ) * f v = ∑ x, f (g x) := by
  rw [← sum_fiberwise_of_maps_to' (s := univ) (t := S) (fun x _ => hS x) f]
  refine sum_congr rfl fun v _ => ?_
  rw [sum_const, nsmul_eq_mul]

theorem sum_card_fiber_div [Nonempty Ω] (g : Ω → V) (S : Finset V) (hS : ∀ x, g x ∈ S) :
    ∑ v ∈ S, (#{x | g x = v} : ℝ) / Fintype.card Ω = 1 := by
  have h : ∑ v ∈ S, (#{x | g x = v} : ℝ) * 1 = ∑ x : Ω, (1 : ℝ) := sum_card_fiber_mul g S hS fun _ => 1
  rw [← sum_div, div_eq_one_iff_eq (Nat.cast_ne_zero.2 Fintype.card_ne_zero)]
  simpa only [mul_one, sum_const, card_univ, nsmul_eq_mul] using h

theorem prCoin_eq_ofReal [Nonempty Ω] (p : Ω → Prop) [DecidablePred p] :
    prCoin p = ENNReal.ofReal (((univ.filter p).card : ℝ) / Fintype.card Ω) := by
  rw [prCoin_eq_card, ENNReal.ofReal_div_of_pos (Nat.cast_pos.2 Fintype.card_pos), ENNReal.ofReal_natCast,
    ENNReal.ofReal_natCast]

theorem card_filter_comp_div (g : Ω → V) (S : Finset V) (hS : ∀ x, g x ∈ S) (E : V → Prop) [DecidablePred E] :
    (#{x | E (g x)} : ℝ) / Fintype.card Ω =
      ∑ v ∈ S, (if E v then (1 : ℝ) else 0) * ((#{x | g x = v} : ℝ) / Fintype.card Ω) := by
  have h : ∑ v ∈ S, (#{x | g x = v} : ℝ) * (if E v then (1 : ℝ) else 0) = ∑ x, if E (g x) then (1 : ℝ) else 0 :=
    sum_card_fiber_mul g S hS fun v => if E v then (1 : ℝ) else 0
  rw [natCast_card_filter, ← h, sum_div]
  refine sum_congr rfl fun v _ => ?_
  ring

/-- **A test's advantage is at most the statistical distance**: for two pushforwards `g`, `g'` of uniform coins into
`S`, every event's probabilities differ by at most half the ℓ¹ distance of their masses on `S`. -/
theorem prCoin_comp_le {Ω' : Type} [Fintype Ω'] [Nonempty Ω] [Nonempty Ω'] (g : Ω → V) (g' : Ω' → V) (S : Finset V)
    (hS : ∀ x, g x ∈ S) (hS' : ∀ x, g' x ∈ S) (E : V → Prop) :
    prCoin (fun x => E (g x)) ≤ prCoin (fun x => E (g' x)) + ENNReal.ofReal (1 / 2 * ∑ v ∈ S,
      |(#{x | g x = v} : ℝ) / Fintype.card Ω - (#{x | g' x = v} : ℝ) / Fintype.card Ω'|) := by
  classical
  rw [prCoin_eq_ofReal (fun x => E (g x)), prCoin_eq_ofReal (fun x => E (g' x)), card_filter_comp_div g S hS E,
    card_filter_comp_div g' S hS' E]
  have hν0 : 0 ≤ ∑ v ∈ S, (if E v then (1 : ℝ) else 0) * ((#{x | g' x = v} : ℝ) / Fintype.card Ω') :=
    sum_nonneg fun v _ => mul_nonneg (by split_ifs <;> norm_num) (by positivity)
  have hd0 : 0 ≤ 1 / 2 * ∑ v ∈ S,
      |(#{x | g x = v} : ℝ) / Fintype.card Ω - (#{x | g' x = v} : ℝ) / Fintype.card Ω'| :=
    mul_nonneg (by norm_num) (sum_nonneg fun _ _ => abs_nonneg _)
  rw [← ENNReal.ofReal_add hν0 hd0]
  refine ENNReal.ofReal_le_ofReal (sum_ind_le_tv S E _ _ ?_)
  rw [sum_card_fiber_div g S hS, sum_card_fiber_div g' S hS']

end Push

section Gap

variable {Y B C : Type} [Fintype Y] [Fintype B] [DecidableEq B] [DecidableEq C]

theorem hidingGap_nonneg (My : Y → B) (c : Y → C) : 0 ≤ hidingGap My c :=
  mul_nonneg (by norm_num) (sum_nonneg fun _ _ => abs_nonneg _)

/-- **hm96 hiding at its gap**: `Hm96Hiding` holds with `δ₁` the statistical distance of `(My y, c y)` from `(U, c y)`,
with no hypothesis. -/
theorem hm96Hiding_gap [Nonempty Y] [Nonempty B] (My : Y → B) (c : Y → C) :
    Assumptions.Hm96Hiding My c (ENNReal.ofReal (hidingGap My c)) := fun E =>
  prCoin_comp_le (fun y => (My y, c y)) (fun p : B × Y => (p.1, c p.2)) ((univ : Finset B) ×ˢ univ.image c)
    (fun y => mem_product.2 ⟨mem_univ _, mem_image_of_mem c (mem_univ y)⟩)
    (fun p => mem_product.2 ⟨mem_univ _, mem_image_of_mem c (mem_univ p.2)⟩) E

omit [DecidableEq B] [DecidableEq C] in
/-- Hiding is monotone in `δ₁`. -/
theorem hm96Hiding_mono {My : Y → B} {c : Y → C} {δ δ' : ℝ≥0∞} (h : Assumptions.Hm96Hiding My c δ) (hδ : δ ≤ δ') :
    Assumptions.Hm96Hiding My c δ' := fun E => (h E).trans (add_le_add le_rfl hδ)

theorem card_uniform_pair (c : Y → C) (b : B) (z : C) :
    #{p : B × Y | (p.1, c p.2) = (b, z)} = #{y | c y = z} := by
  refine card_nbij' (fun p => p.2) (fun y => (b, y)) ?_ ?_ ?_ ?_
  · intro p hp
    simp only [mem_coe, mem_filter, mem_univ, true_and] at hp ⊢
    exact congrArg Prod.snd hp
  · intro y hy
    simp only [mem_coe, mem_filter, mem_univ, true_and] at hy ⊢
    rw [hy]
  · intro p hp
    simp only [mem_coe, mem_filter, mem_univ, true_and] at hp
    exact Prod.ext (congrArg Prod.fst hp).symm rfl
  · intro y _
    rfl

variable {K : Type} [Fintype K] [Nonempty Y] [Nonempty B]

/-- **The leftover hash lemma, for one hash `c` of the salt beside the key's**: over a universal family `M`, the hiding
gap's sum over the keys is at most `|K|·½·√(|B|·|c(Y)|/|Y|)`. -/
theorem sum_hidingGap_le (M : K → Y → B) (hU : Universal M) (c : Y → C) :
    ∑ k, hidingGap (M k) c ≤
      Fintype.card K * (1 / 2 * Real.sqrt (Fintype.card B * #(univ.image c) / Fintype.card Y)) := by
  classical
  have hNY : (0 : ℝ) < Fintype.card Y := Nat.cast_pos.2 Fintype.card_pos
  have hNB : (0 : ℝ) < Fintype.card B := Nat.cast_pos.2 Fintype.card_pos
  have hY0 : (Fintype.card Y : ℝ) ≠ 0 := hNY.ne'
  have hB0 : (Fintype.card B : ℝ) ≠ 0 := hNB.ne'
  have hS : ∀ k y, (M k y, c y) ∈ (univ : Finset B) ×ˢ univ.image c := fun k y =>
    mem_product.2 ⟨mem_univ _, mem_image_of_mem c (mem_univ y)⟩
  let m : C → ℝ := fun z => (#{y | c y = z} : ℝ)
  let μ : K → B × C → ℝ := fun k v => (#{y | (M k y, c y) = v} : ℝ) / Fintype.card Y
  let ν : B × C → ℝ := fun v => (#{p : B × Y | (p.1, c p.2) = v} : ℝ) / Fintype.card (B × Y)
  have hν : ∀ v, ν v = m v.2 / (Fintype.card B * Fintype.card Y) := by
    rintro ⟨b, z⟩
    simp only [ν, m, card_uniform_pair c b z, Fintype.card_prod, Nat.cast_mul]
  have hgap : ∀ k, hidingGap (M k) c = 1 / 2 * ∑ v ∈ (univ : Finset B) ×ˢ univ.image c, |μ k v - ν v| :=
    fun _ => rfl
  have hfib : ∀ k (f : B × C → ℝ), ∑ v ∈ (univ : Finset B) ×ˢ univ.image c, μ k v * f v =
      (∑ y, f (M k y, c y)) / Fintype.card Y := by
    intro k f
    have h : ∑ v ∈ (univ : Finset B) ×ˢ univ.image c, (#{y | (M k y, c y) = v} : ℝ) * f v =
        ∑ y, f (M k y, c y) := sum_card_fiber_mul (fun y => (M k y, c y)) _ (hS k) f
    rw [← h, sum_div]
    refine sum_congr rfl fun v _ => ?_
    simp only [μ]
    ring
  have hfibc : ∀ f : C → ℝ, ∑ z ∈ univ.image c, m z * f z = ∑ y, f (c y) := fun f =>
    sum_card_fiber_mul c (univ.image c) (fun y => mem_image_of_mem c (mem_univ y)) f
  -- `Q = Σ_v ν(v)² = Σ_v μ_k(v)·ν(v)`, the uniform pair's collision mass
  set Q : ℝ := (∑ y, m (c y)) / (Fintype.card B * (Fintype.card Y : ℝ) ^ 2) with hQ
  have hμν : ∀ k, ∑ v ∈ (univ : Finset B) ×ˢ univ.image c, μ k v * ν v = Q := by
    intro k
    rw [hfib k ν]
    simp only [hν]
    rw [← sum_div, hQ]
    ring
  have hνν : ∑ v ∈ (univ : Finset B) ×ˢ univ.image c, ν v ^ 2 = Q := by
    rw [sum_product]
    simp only [hν]
    rw [sum_const, card_univ, nsmul_eq_mul]
    have h2 : ∑ z ∈ univ.image c, (m z / (Fintype.card B * Fintype.card Y)) ^ 2 =
        ∑ z ∈ univ.image c, m z * (m z / ((Fintype.card B : ℝ) * Fintype.card Y) ^ 2) :=
      sum_congr rfl fun z _ => by ring
    rw [h2, hfibc, ← sum_div, hQ, ← mul_div_assoc,
      div_eq_div_iff (pow_ne_zero 2 (mul_ne_zero hB0 hY0)) (mul_ne_zero hB0 (pow_ne_zero 2 hY0))]
    ring
  have hx : ∀ k, ∑ v ∈ (univ : Finset B) ×ˢ univ.image c, (μ k v - ν v) ^ 2 =
      ∑ v ∈ (univ : Finset B) ×ˢ univ.image c, μ k v * μ k v - Q := by
    intro k
    have h : ∀ v ∈ (univ : Finset B) ×ˢ univ.image c,
        (μ k v - ν v) ^ 2 = μ k v * μ k v - 2 * (μ k v * ν v) + ν v ^ 2 := fun v _ => by ring
    rw [sum_congr rfl h, sum_add_distrib, sum_sub_distrib, ← mul_sum, hμν k, hνν]
    ring
  have hμμ : ∀ k, ∑ v ∈ (univ : Finset B) ×ˢ univ.image c, μ k v * μ k v =
      (∑ y, (#{y' | (M k y', c y') = (M k y, c y)} : ℝ)) / (Fintype.card Y : ℝ) ^ 2 := by
    intro k
    rw [hfib k (μ k)]
    simp only [μ]
    rw [← sum_div]
    ring
  -- the collision count, by universality
  have hpair : ∀ y y' : Y, (#{k | (M k y', c y') = (M k y, c y)} : ℝ) ≤
      (if y' = y then (Fintype.card K : ℝ) else 0) +
        (if c y' = c y then (Fintype.card K : ℝ) / Fintype.card B else 0) := by
    intro y y'
    by_cases hyy : y' = y
    · rw [ite_eq_left hyy, ite_eq_left (congrArg c hyy)]
      have h1 : (#{k | (M k y', c y') = (M k y, c y)} : ℝ) ≤ Fintype.card K := by
        exact_mod_cast (card_filter_le _ _).trans_eq card_univ
      have h2 : (0 : ℝ) ≤ Fintype.card K / Fintype.card B := by positivity
      linarith
    · rw [ite_eq_right hyy]
      by_cases hc : c y' = c y
      · rw [ite_eq_left hc, zero_add]
        have h1 : #{k | (M k y', c y') = (M k y, c y)} = #{k | M k y' = M k y} := by
          congr 1
          ext k
          simp [hc]
        rw [h1, le_div_iff₀ hNB]
        exact_mod_cast hU y' y hyy
      · rw [ite_eq_right hc, zero_add]
        have h1 : #{k | (M k y', c y') = (M k y, c y)} = 0 := by
          rw [card_eq_zero, filter_eq_empty_iff]
          intro k _ h
          exact hc (congrArg Prod.snd h)
        rw [h1, Nat.cast_zero]
  have hrow : ∀ y : Y, ∑ y', ((if y' = y then (Fintype.card K : ℝ) else 0) +
      (if c y' = c y then (Fintype.card K : ℝ) / Fintype.card B else 0)) =
      Fintype.card K + Fintype.card K / Fintype.card B * m (c y) := by
    intro y
    rw [sum_add_distrib, sum_ite_eq', ite_eq_left (mem_univ y), ← sum_filter, sum_const, nsmul_eq_mul]
    simp only [m]
    ring
  have hcol : ∀ y : Y, ∑ k, (#{y' | (M k y', c y') = (M k y, c y)} : ℝ) ≤
      Fintype.card K + Fintype.card K / Fintype.card B * m (c y) := by
    intro y
    have hswap : ∑ k, (#{y' | (M k y', c y') = (M k y, c y)} : ℝ) =
        ∑ y', (#{k | (M k y', c y') = (M k y, c y)} : ℝ) := by
      simp only [natCast_card_filter]
      exact sum_comm
    rw [hswap, ← hrow y]
    exact sum_le_sum fun y' _ => hpair y y'
  have hX : ∑ k, ∑ v ∈ (univ : Finset B) ×ˢ univ.image c, (μ k v - ν v) ^ 2 ≤
      Fintype.card K / Fintype.card Y := by
    have h1 : ∑ y, ∑ k, (#{y' | (M k y', c y') = (M k y, c y)} : ℝ) ≤
        Fintype.card Y * Fintype.card K + Fintype.card K / Fintype.card B * ∑ y, m (c y) := by
      refine (sum_le_sum fun y _ => hcol y).trans_eq ?_
      rw [sum_add_distrib, sum_const, card_univ, nsmul_eq_mul, ← mul_sum]
    have h2 : ∑ k, ∑ v ∈ (univ : Finset B) ×ˢ univ.image c, (μ k v - ν v) ^ 2 =
        (∑ y, ∑ k, (#{y' | (M k y', c y') = (M k y, c y)} : ℝ)) / (Fintype.card Y : ℝ) ^ 2 -
          Fintype.card K * Q := by
      simp only [hx, hμμ]
      rw [sum_sub_distrib, ← sum_div, sum_const, card_univ, nsmul_eq_mul, sum_comm]
    rw [h2, hQ]
    have h3 : (∑ y, ∑ k, (#{y' | (M k y', c y') = (M k y, c y)} : ℝ)) / (Fintype.card Y : ℝ) ^ 2 ≤
        (Fintype.card Y * Fintype.card K + Fintype.card K / Fintype.card B * ∑ y, m (c y)) /
          (Fintype.card Y : ℝ) ^ 2 :=
      div_le_div_of_nonneg_right h1 (by positivity)
    have h4 : (Fintype.card Y * Fintype.card K + Fintype.card K / Fintype.card B * ∑ y, m (c y)) /
          (Fintype.card Y : ℝ) ^ 2 - Fintype.card K * ((∑ y, m (c y)) /
            (Fintype.card B * (Fintype.card Y : ℝ) ^ 2)) = Fintype.card K / Fintype.card Y := by
      field_simp
      ring
    linarith
  -- Cauchy–Schwarz over (key, pair)
  have hCS : (∑ k, ∑ v ∈ (univ : Finset B) ×ˢ univ.image c, |μ k v - ν v|) ^ 2 ≤
      (Fintype.card K * #((univ : Finset B) ×ˢ univ.image c)) *
        ∑ k, ∑ v ∈ (univ : Finset B) ×ˢ univ.image c, (μ k v - ν v) ^ 2 := by
    have h := sq_sum_le_card_mul_sum_sq (s := (univ : Finset K) ×ˢ ((univ : Finset B) ×ˢ univ.image c))
      (f := fun kv => |μ kv.1 kv.2 - ν kv.2|)
    rw [sum_product, sum_product, card_product, card_univ] at h
    simpa only [sq_abs, Nat.cast_mul] using h
  have hA0 : 0 ≤ ∑ k, ∑ v ∈ (univ : Finset B) ×ˢ univ.image c, |μ k v - ν v| :=
    sum_nonneg fun _ _ => sum_nonneg fun _ _ => abs_nonneg _
  have hSc : (#((univ : Finset B) ×ˢ univ.image c) : ℝ) = Fintype.card B * #(univ.image c) := by
    rw [card_product, card_univ, Nat.cast_mul]
  have hK0 : (0 : ℝ) ≤ Fintype.card K := Nat.cast_nonneg _
  have hA : ∑ k, ∑ v ∈ (univ : Finset B) ×ˢ univ.image c, |μ k v - ν v| ≤
      Fintype.card K * Real.sqrt (Fintype.card B * #(univ.image c) / Fintype.card Y) := by
    have h1 : (∑ k, ∑ v ∈ (univ : Finset B) ×ˢ univ.image c, |μ k v - ν v|) ^ 2 ≤
        (Fintype.card K : ℝ) ^ 2 * (Fintype.card B * #(univ.image c) / Fintype.card Y) := by
      refine hCS.trans ?_
      rw [hSc]
      calc (Fintype.card K * (Fintype.card B * #(univ.image c))) *
            ∑ k, ∑ v ∈ (univ : Finset B) ×ˢ univ.image c, (μ k v - ν v) ^ 2
          ≤ (Fintype.card K * (Fintype.card B * #(univ.image c))) * (Fintype.card K / Fintype.card Y) :=
            mul_le_mul_of_nonneg_left hX (by positivity)
        _ = _ := by ring
    calc ∑ k, ∑ v ∈ (univ : Finset B) ×ˢ univ.image c, |μ k v - ν v|
        = Real.sqrt ((∑ k, ∑ v ∈ (univ : Finset B) ×ˢ univ.image c, |μ k v - ν v|) ^ 2) :=
          (Real.sqrt_sq hA0).symm
      _ ≤ Real.sqrt ((Fintype.card K : ℝ) ^ 2 * (Fintype.card B * #(univ.image c) / Fintype.card Y)) :=
          Real.sqrt_le_sqrt h1
      _ = _ := by rw [Real.sqrt_mul (sq_nonneg _), Real.sqrt_sq hK0]
  calc ∑ k, hidingGap (M k) c = 1 / 2 * ∑ k, ∑ v ∈ (univ : Finset B) ×ˢ univ.image c, |μ k v - ν v| := by
        rw [mul_sum]
        exact sum_congr rfl fun k _ => hgap k
    _ ≤ 1 / 2 * (Fintype.card K * Real.sqrt (Fintype.card B * #(univ.image c) / Fintype.card Y)) :=
        mul_le_mul_of_nonneg_left hA (by norm_num)
    _ = _ := by ring

/-- **Markov's inequality for the gap**: the keys whose gap exceeds `ε` number at most `1/ε` times the gap's sum
bound. -/
theorem card_hidingGap_gt (M : K → Y → B) (hU : Universal M) (c : Y → C) {ε : ℝ} :
    (#{k | ε < hidingGap (M k) c} : ℝ) * ε ≤
      Fintype.card K * (1 / 2 * Real.sqrt (Fintype.card B * #(univ.image c) / Fintype.card Y)) := by
  calc (#{k | ε < hidingGap (M k) c} : ℝ) * ε = ∑ k ∈ univ.filter (fun k => ε < hidingGap (M k) c), ε := by
        rw [sum_const, nsmul_eq_mul]
    _ ≤ ∑ k ∈ univ.filter (fun k => ε < hidingGap (M k) c), hidingGap (M k) c :=
        sum_le_sum fun k hk => (mem_filter.1 hk).2.le
    _ ≤ ∑ k, hidingGap (M k) c :=
        sum_le_sum_of_subset_of_nonneg (filter_subset _ _) fun k _ _ => hidingGap_nonneg _ _
    _ ≤ _ := sum_hidingGap_le M hU c

end Gap

end FlockSoundness.CROnly
