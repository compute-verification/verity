import Proofs.Core.HongKung

/-!
# Products of two operands: Loomis–Whitney and the amortized cube cap

The combinatorics of a matrix product's products, for any guarantee that counts them. A product of `C = A · B` by the
standard algorithm is a triple `(i, l, j)`: its `A` entry is `π₁₂ = (i, l)`, its `B` entry `π₂₃ = (l, j)` and its `C`
entry `π₁₃ = (i, j)`.

* `card_sq_le_proj`: **discrete Loomis–Whitney**, `|V|² ≤ |π₁₂ V| |π₂₃ V| |π₁₃ V|` for a finite set of triples `V`
  (`Verity.Accel.HongKung.loomis_whitney`), and `card_le_sqrt_proj`, `|V| ≤ √(…)`.
* `sqrt_mul_le`: **AM–GM's linearization**, `√(x y z) ≤ (b_A x + b_B y)² / (16 μ b_A b_B b_C) + μ b_C z` for
  `x, y, z ≥ 0` and `b_A, b_B, b_C, μ > 0`. It is linear in the entries `z`, so a bound on their sum over blocks bounds
  the sum of the products.
* `sq_add_le_max`: a convex function of the bits a block spends on entries is at most its value at an end,
  `(K - t)² c₀ + m t ≤ max (K² c₀) (m K)` for `0 ≤ t ≤ K`.
* `sum_card_le`: **the amortized cap**. Blocks `u` with products `T_u`, each paying `R_u` bits for operands beside the
  `H_u` operand bits it holds (`b_A |π₁₂ T_u| + b_B |π₂₃ T_u| ≤ R_u + H_u`) and `X_u` bits of other reads
  (`R_u + X_u ≤ k`), with the entries charged only across all blocks (`b_C Σ_u |π₁₃ T_u| ≤ E + Σ_u X_u`), have, for
  every `μ > 0` and every weight `s` on the other reads,
  `Σ_u |T_u| + s Σ_u X_u ≤ Σ_u max ((k + H_u)² / (16 μ b_A b_B b_C)) ((μ + s)(k + H_u)) + μ E`.
* `sum_card_le_cap`: at `H = 0` and `s = 0`, with `μ* = √k / (4 √(b_A b_B b_C))` (`max_at_opt`),
  `Σ_u |T_u| ≤ B k √k / (4 √(b_A b_B b_C)) + μ* E`: `k^{3/2} / (4 √(b_A b_B b_C))` products a block, amortized.
* `sqrt_mul_le_third`: the per-block form `√(x y z) ≤ (k/3) √(k/3) / √(b_A b_B b_C)` when
  `b_A x + b_B y + b_C z ≤ k`, which holds of a block that pays `b_C` bits for every entry it touches. When only reads
  are charged a block that starts an entry pays nothing for it, and the amortized cap, `3√3/4 ≈ 1.299` times this
  one, is attained by blocks of two kinds, one reading operands and the other partial sums (`split_attains`).
-/

namespace Verity.Combinatorics

open Finset

section LoomisWhitney

variable {α β γ : Type*} [DecidableEq α] [DecidableEq β] [DecidableEq γ]

/-- **Discrete Loomis–Whitney**: `|V|² ≤ |π₁₂ V| |π₂₃ V| |π₁₃ V|`. -/
theorem card_sq_le_proj (V : Finset (α × β × γ)) :
    #V ^ 2 ≤ #(V.image fun x => (x.1, x.2.1)) * #(V.image fun x => (x.2.1, x.2.2)) *
      #(V.image fun x => (x.1, x.2.2)) :=
  Accel.HongKung.loomis_whitney V

/-- A finite set of triples has at most `√(|π₁₂ V| |π₂₃ V| |π₁₃ V|)` elements. -/
theorem card_le_sqrt_proj (V : Finset (α × β × γ)) :
    (#V : ℝ) ≤ √((#(V.image fun x => (x.1, x.2.1)) : ℝ) * #(V.image fun x => (x.2.1, x.2.2)) *
      #(V.image fun x => (x.1, x.2.2))) := by
  rw [Real.le_sqrt (Nat.cast_nonneg _) (by positivity)]
  exact_mod_cast card_sq_le_proj V

end LoomisWhitney

/-- **AM–GM's linearization**: `√(x y z) ≤ (b_A x + b_B y)² / (16 μ b_A b_B b_C) + μ b_C z`. -/
theorem sqrt_mul_le {x y z bA bB bC μ : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hA : 0 < bA) (hB : 0 < bB)
    (hC : 0 < bC) (hμ : 0 < μ) :
    √(x * y * z) ≤ (bA * x + bB * y) ^ 2 / (16 * μ * bA * bB * bC) + μ * bC * z := by
  have hxy : 4 * bA * bB * (x * y) ≤ (bA * x + bB * y) ^ 2 := by nlinarith [sq_nonneg (bA * x - bB * y)]
  have hq : x * y / (4 * μ * bC) ≤ (bA * x + bB * y) ^ 2 / (16 * μ * bA * bB * bC) := by
    rw [div_le_div_iff₀ (by positivity) (by positivity)]
    have h4 : 0 ≤ 4 * μ * bC := by positivity
    nlinarith [mul_le_mul_of_nonneg_right hxy h4]
  have hab : x * y / (4 * μ * bC) * (μ * bC * z) = x * y * z / 4 := by
    field_simp
  have hamgm : √(x * y * z) ≤ x * y / (4 * μ * bC) + μ * bC * z := by
    rw [Real.sqrt_le_left (by positivity)]
    nlinarith [sq_nonneg (x * y / (4 * μ * bC) - μ * bC * z)]
  linarith

/-- A convex function of the bits `t` a block spends on entries is at most its value at an end:
`(K - t)² c₀ + m t ≤ max (K² c₀) (m K)` for `0 ≤ t ≤ K`. -/
theorem sq_add_le_max {K t c₀ m : ℝ} (ht : 0 ≤ t) (htK : t ≤ K) (hc : 0 ≤ c₀) :
    (K - t) ^ 2 * c₀ + m * t ≤ max (K ^ 2 * c₀) (m * K) := by
  have h1 : (K - t) ^ 2 * c₀ ≤ K * (K - t) * c₀ := by
    have : (K - t) ^ 2 ≤ K * (K - t) := by nlinarith [mul_nonneg ht (sub_nonneg.2 htK)]
    exact mul_le_mul_of_nonneg_right this hc
  rcases le_total m (K * c₀) with h | h
  · have : m * t ≤ K * c₀ * t := mul_le_mul_of_nonneg_right h ht
    calc (K - t) ^ 2 * c₀ + m * t ≤ K * (K - t) * c₀ + K * c₀ * t := by linarith
      _ = K ^ 2 * c₀ := by ring
      _ ≤ _ := le_max_left _ _
  · have : K * (K - t) * c₀ ≤ m * (K - t) := by nlinarith [mul_nonneg (sub_nonneg.2 htK) (sub_nonneg.2 h)]
    calc (K - t) ^ 2 * c₀ + m * t ≤ m * (K - t) + m * t := by linarith
      _ = m * K := by ring
      _ ≤ _ := le_max_right _ _

section Amortized

variable {α β γ ι : Type*} [DecidableEq α] [DecidableEq β] [DecidableEq γ] [Fintype ι]

/-- **The amortized cap**: blocks whose operands cost `b_A` and `b_B` bits each, beside the operand bits they hold, and
whose entries are paid only across all blocks, `b_C Σ_u |π₁₃ T_u| ≤ E + Σ_u X_u`, have, for every `μ > 0` and every
weight `s`, `Σ_u |T_u| + s Σ_u X_u ≤ Σ_u max ((k + H_u)² / (16 μ b_A b_B b_C)) ((μ + s)(k + H_u)) + μ E`. -/
theorem sum_card_le (T : ι → Finset (α × β × γ)) (R H X : ι → ℝ) {k bA bB bC μ s E : ℝ} (hA : 0 < bA)
    (hB : 0 < bB) (hC : 0 < bC) (hμ : 0 < μ) (hR : ∀ u, 0 ≤ R u) (hH : ∀ u, 0 ≤ H u)
    (hX : ∀ u, 0 ≤ X u)
    (hops : ∀ u, bA * #((T u).image fun x => (x.1, x.2.1)) + bB * #((T u).image fun x => (x.2.1, x.2.2)) ≤
      R u + H u)
    (hk : ∀ u, R u + X u ≤ k)
    (hz : bC * ∑ u, (#((T u).image fun x => (x.1, x.2.2)) : ℝ) ≤ E + ∑ u, X u) :
    ∑ u, (#(T u) : ℝ) + s * ∑ u, X u ≤
      ∑ u, max ((k + H u) ^ 2 / (16 * μ * bA * bB * bC)) ((μ + s) * (k + H u)) + μ * E := by
  obtain ⟨c₀, hc₀⟩ : ∃ c : ℝ, c = 1 / (16 * μ * bA * bB * bC) := ⟨_, rfl⟩
  have hc : 0 ≤ c₀ := by rw [hc₀]; positivity
  have hdiv : ∀ a : ℝ, a / (16 * μ * bA * bB * bC) = a * c₀ := fun a => by rw [hc₀, mul_one_div]
  have hblock : ∀ u, (#(T u) : ℝ) ≤ (R u + H u) ^ 2 * c₀ + μ * bC * #((T u).image fun x => (x.1, x.2.2)) := by
    intro u
    have hnn : 0 ≤ bA * (#((T u).image fun x => (x.1, x.2.1)) : ℝ) +
        bB * #((T u).image fun x => (x.2.1, x.2.2)) :=
      add_nonneg (mul_nonneg hA.le (Nat.cast_nonneg _)) (mul_nonneg hB.le (Nat.cast_nonneg _))
    calc (#(T u) : ℝ) ≤ _ := card_le_sqrt_proj (T u)
      _ ≤ _ := sqrt_mul_le (Nat.cast_nonneg _) (Nat.cast_nonneg _) (Nat.cast_nonneg _) hA hB hC hμ
      _ ≤ (R u + H u) ^ 2 * c₀ + μ * bC * #((T u).image fun x => (x.1, x.2.2)) := by
          rw [hdiv]
          linarith [mul_le_mul_of_nonneg_right (pow_le_pow_left₀ hnn (hops u) 2) hc]
  have hper : ∀ u, (R u + H u) ^ 2 * c₀ + (μ + s) * X u ≤
      max ((k + H u) ^ 2 / (16 * μ * bA * bB * bC)) ((μ + s) * (k + H u)) := by
    intro u
    have h1 : (R u + H u) ^ 2 * c₀ ≤ ((k + H u) - X u) ^ 2 * c₀ := by
      gcongr
      · linarith [hR u, hH u]
      · linarith [hk u]
    have h2 := sq_add_le_max (K := k + H u) (m := μ + s) (hX u) (by linarith [hR u, hH u, hk u]) hc
    rw [hdiv]
    linarith
  calc ∑ u, (#(T u) : ℝ) + s * ∑ u, X u
      ≤ ∑ u, ((R u + H u) ^ 2 * c₀ + μ * bC * #((T u).image fun x => (x.1, x.2.2))) + s * ∑ u, X u := by
        gcongr with u
        exact hblock u
    _ = ∑ u, (R u + H u) ^ 2 * c₀ + μ * (bC * ∑ u, (#((T u).image fun x => (x.1, x.2.2)) : ℝ)) +
          s * ∑ u, X u := by
        rw [sum_add_distrib, mul_sum, mul_sum, mul_sum]
        congr 2
        exact sum_congr rfl fun u _ => by ring
    _ ≤ ∑ u, (R u + H u) ^ 2 * c₀ + μ * (E + ∑ u, X u) + s * ∑ u, X u := by gcongr
    _ = ∑ u, ((R u + H u) ^ 2 * c₀ + (μ + s) * X u) + μ * E := by
        rw [sum_add_distrib, ← mul_sum]
        ring
    _ ≤ ∑ u, max ((k + H u) ^ 2 / (16 * μ * bA * bB * bC)) ((μ + s) * (k + H u)) + μ * E := by
        gcongr with u
        exact hper u

end Amortized

/-- At `μ* = √k / (4 √b)` the two terms of the cap agree: `max (k² / (16 μ* b)) (μ* k) = k √k / (4 √b)`. -/
theorem max_at_opt {k b : ℝ} (hk : 0 < k) (hb : 0 < b) :
    max (k ^ 2 / (16 * (√k / (4 * √b)) * b)) (√k / (4 * √b) * k) = k * √k / (4 * √b) := by
  obtain ⟨a, ha, rfl⟩ : ∃ a, 0 < a ∧ a ^ 2 = k := ⟨√k, Real.sqrt_pos.2 hk, Real.sq_sqrt hk.le⟩
  obtain ⟨c, hc, rfl⟩ : ∃ c, 0 < c ∧ c ^ 2 = b := ⟨√b, Real.sqrt_pos.2 hb, Real.sq_sqrt hb.le⟩
  rw [Real.sqrt_sq ha.le, Real.sqrt_sq hc.le]
  have h1 : (a ^ 2) ^ 2 / (16 * (a / (4 * c)) * c ^ 2) = a ^ 2 * a / (4 * c) := by
    field_simp
    ring
  have h2 : a / (4 * c) * a ^ 2 = a ^ 2 * a / (4 * c) := by
    field_simp
  rw [h1, h2, max_self]

section Amortized

variable {α β γ ι : Type*} [DecidableEq α] [DecidableEq β] [DecidableEq γ] [Fintype ι]

/-- **The amortized cap at `μ*`**, no operand bits held: `Σ_u |T_u| ≤ B k √k / (4 √(b_A b_B b_C)) + μ* E`, with `B` the
number of blocks and `μ* = √k / (4 √(b_A b_B b_C))`. -/
theorem sum_card_le_cap (T : ι → Finset (α × β × γ)) (R X : ι → ℝ) {k bA bB bC E : ℝ} (hk0 : 0 < k)
    (hA : 0 < bA) (hB : 0 < bB) (hC : 0 < bC) (hR : ∀ u, 0 ≤ R u) (hX : ∀ u, 0 ≤ X u)
    (hops : ∀ u, bA * #((T u).image fun x => (x.1, x.2.1)) + bB * #((T u).image fun x => (x.2.1, x.2.2)) ≤ R u)
    (hk : ∀ u, R u + X u ≤ k)
    (hz : bC * ∑ u, (#((T u).image fun x => (x.1, x.2.2)) : ℝ) ≤ E + ∑ u, X u) :
    ∑ u, (#(T u) : ℝ) ≤
      Fintype.card ι * (k * √k / (4 * √(bA * bB * bC))) + √k / (4 * √(bA * bB * bC)) * E := by
  have hb : 0 < bA * bB * bC := by positivity
  have hμ : 0 < √k / (4 * √(bA * bB * bC)) := by
    have := Real.sqrt_pos.2 hk0
    have := Real.sqrt_pos.2 hb
    positivity
  have h := sum_card_le T R (fun _ => 0) X (s := 0) (E := E) hA hB hC hμ hR (fun _ => le_rfl) hX
    (fun u => by simpa using hops u) hk hz
  simp only [add_zero, zero_mul] at h
  rw [mul_assoc (16 * _) bA bB, mul_assoc (16 * _) (bA * bB) bC, max_at_opt hk0 hb, sum_const, card_univ,
    nsmul_eq_mul] at h
  exact h

end Amortized

/-- **The per-block form**: a block that pays `b_C` bits for each entry it touches, `b_A x + b_B y + b_C z ≤ k`, has
`√(x y z) ≤ (k/3) √(k/3) / √(b_A b_B b_C)`. -/
theorem sqrt_mul_le_third {x y z bA bB bC k : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hA : 0 < bA)
    (hB : 0 < bB) (hC : 0 < bC) (h : bA * x + bB * y + bC * z ≤ k) :
    √(x * y * z) ≤ k / 3 * √(k / 3) / √(bA * bB * bC) := by
  have ha : 0 ≤ bA * x := by positivity
  have hb : 0 ≤ bB * y := by positivity
  have hc : 0 ≤ bC * z := by positivity
  have hk : 0 ≤ k := by linarith
  have hb3 : 0 < bA * bB * bC := by positivity
  have hamgm : 27 * ((bA * x) * (bB * y) * (bC * z)) ≤ (bA * x + bB * y + bC * z) ^ 3 := by
    nlinarith [mul_nonneg (add_nonneg (add_nonneg ha hb) hc) (sq_nonneg (bA * x - bB * y)),
      mul_nonneg (add_nonneg (add_nonneg ha hb) hc) (sq_nonneg (bB * y - bC * z)),
      mul_nonneg (add_nonneg (add_nonneg ha hb) hc) (sq_nonneg (bA * x - bC * z)),
      mul_nonneg ha (sq_nonneg (bB * y - bC * z)), mul_nonneg hb (sq_nonneg (bA * x - bC * z)),
      mul_nonneg hc (sq_nonneg (bA * x - bB * y))]
  have hcube : (bA * x + bB * y + bC * z) ^ 3 ≤ k ^ 3 := by
    exact pow_le_pow_left₀ (by positivity) h 3
  have hsq : (k / 3 * √(k / 3) / √(bA * bB * bC)) ^ 2 = (k / 3) ^ 3 / (bA * bB * bC) := by
    rw [div_pow, mul_pow, Real.sq_sqrt (by positivity), Real.sq_sqrt hb3.le]
    ring
  rw [Real.sqrt_le_left (by positivity), hsq, le_div_iff₀ hb3]
  nlinarith

/-- **The split attains the amortized cap** where the per-block form fails: at `b_A = b_B = b_C = 1` and `k = 16`,
one block reading the 8 + 8 operands of a cube of `4 · 2 · 4 = 32` products and handing out its 16 partial sums, and
one reading them, do 32 products in 2 blocks, `16 = k^{3/2} / 4` a block, above `(k/3)^{3/2} ≈ 12.3`. -/
theorem split_attains :
    (32 : ℝ) / 2 = 16 * √16 / (4 * √(1 * 1 * 1)) ∧ 16 / 3 * √(16 / 3) / √(1 * 1 * 1) < (32 : ℝ) / 2 := by
  have h16 : √(16 : ℝ) = 4 := by
    rw [show (16 : ℝ) = 4 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]
  refine ⟨by rw [h16]; norm_num, ?_⟩
  have h3 : √(16 / 3 : ℝ) < 7 / 3 := by
    rw [Real.sqrt_lt' (by norm_num)]
    norm_num
  rw [mul_one, mul_one, Real.sqrt_one, div_one]
  nlinarith

end Verity.Combinatorics
