import Proofs.Flock.Soundness.Audit.Work
import Proofs.Flock.Soundness.Audit.Influence

/-!
# The stratified law's harm bound

`IsHarmBound L h δ H` (`Audit/Harm.lean`) is `harm_bound`'s specification: every wrong-unit set that escapes with
probability at least `δ` has harm at most `H`. This file proves it for the stratified law (and so the work law), at `H` in
closed form. Stratum `s` has `n_s` units of which `k_s` are drawn uniformly without replacement, and a set `B` holds
`c_s = |B ∩ stratum s|` of them.

* `choose_div_le_exp`, `stratified_escape_le_exp`: **the closed form**. `B` escapes with probability at most
  `exp(−∑ s k_s c_s / n_s)`, since `C(n − c, k)/C(n, k) ≤ (1 − c/n)^k ≤ exp(−k c / n)`. `work_escape_le_exp` is the work
  law's, whose `work_escape_le` bounds the same product through weighted AM-GM at the rule's sizes instead.
* `stratified_risk_le`: a set escaping with probability at least `δ > 0` has `c_s ≤ n_s − k_s` in every stratum and
  `∑ s k_s c_s / n_s ≤ ln(1/δ)`.
* `stratified_isHarmBound_dual`, **the harm bound at a price**: with harm `g_s` per unit of stratum `s`, every `τ ≥ 0`
  gives the harm bound `τ ln(1/δ) + ∑ s (n_s − k_s) max(0, g_s − τ k_s / n_s)`. That is weak duality for the fractional
  knapsack `max ∑ s g_s x_s` over `0 ≤ x_s ≤ n_s − k_s` with `∑ s k_s x_s / n_s ≤ ln(1/δ)`.
* `stratified_isHarmBound_fill`, **the knapsack's value**: a fill `x` that some `τ ≥ 0` separates (`x_s = n_s − k_s` where
  `g_s > τ k_s / n_s`, `x_s = 0` where `g_s < τ k_s / n_s`, and `∑ s k_s x_s / n_s = ln(1/δ)` unless `τ = 0`) attains the
  bound at `τ`, so its value `∑ s g_s x_s` is a harm bound. The greedy fill, strata taken whole in decreasing
  `g_s n_s / k_s` until the budget `ln(1/δ)` runs out, the last one fractionally, is separated by that last stratum's
  `g_s n_s / k_s`, or by `0` when the budget never runs out. This is `density_bound` in the sampling optimizer
  (`verity.core.protocols.sampled_proofs.optimize`).
* `work_isHarmBound_fill`: the same under the work law with harm the credited work `w_s`, so its value bounds
  `workOf σ w B`.
* `IsHarmBound.natFloor`, `stratified_isHarmBound_bits`, `Partition.audit_exfiltration_stratified`: **the exfiltration
  form**. With harm a unit's output bits `b_s`, the floor of the fill's value is `audit_exfiltration`'s `H`.
* `stratumRisk`, `le_stratumRisk`, `stratified_isHarmBound_exact`: the same weak duality with each stratum's exact risk
  `ln(C(n_s, k_s)/C(n_s − c, k_s))` in place of `k_s c / n_s`, which is at most it. Core's
  `verity.core.protocols.profile.Stratified.harm_bound` is that knapsack's value over the exact risk increments: below the
  closed form's, and a harm bound by this theorem at the price its greedy fill stops at, not by the closed form.
-/

namespace FlockSoundness.Audit

open Game Finset
open scoped ENNReal NNReal

namespace Law

variable {n m : ℕ}

/-! ## The closed form -/

/-- The hypergeometric escape of `c` of `N` units under a uniform `k`-subset is at most `exp(−k c / N)`: it is at most
`(1 − c/N)^k` (`choose_div_le_pow`), and `1 − x ≤ exp(−x)`. -/
theorem choose_div_le_exp {N c k : ℕ} (hk : k ≤ N) (hc : c ≤ N) :
    ((N - c).choose k : ℝ) / (N.choose k : ℝ) ≤ Real.exp (-((k : ℝ) * c / N)) := by
  rcases Nat.eq_zero_or_pos N with h0 | hN
  · subst h0
    obtain rfl : k = 0 := by omega
    obtain rfl : c = 0 := by omega
    simp
  have h : ((N - c).choose k : ℝ) / (N.choose k : ℝ) ≤ (((N - c : ℕ) : ℝ) / N) ^ k := by
    exact_mod_cast NNReal.coe_le_coe.2 (choose_div_le_pow (m := c) hk hN)
  refine h.trans ?_
  have hN' : (0 : ℝ) < N := by exact_mod_cast hN
  have h1 : ((N - c : ℕ) : ℝ) / N = 1 - (c : ℝ) / N := by
    rw [Nat.cast_sub hc]
    field_simp
  rw [h1]
  have h0 : 0 ≤ 1 - (c : ℝ) / N := by
    rw [sub_nonneg, div_le_one hN']
    exact_mod_cast hc
  calc (1 - (c : ℝ) / N) ^ k ≤ Real.exp (-((c : ℝ) / N)) ^ k :=
        pow_le_pow_left₀ h0 (by linarith [Real.add_one_le_exp (-((c : ℝ) / N))]) k
    _ = Real.exp (-((k : ℝ) * c / N)) := by
        rw [← Real.exp_nat_mul]
        congr 1
        ring

/-- A hypergeometric factor in `ℝ≥0∞` is its real value. -/
theorem choose_div_eq_ofReal {N c k : ℕ} (hk : k ≤ N) :
    ((N - c).choose k : ℝ≥0∞) / (N.choose k : ℝ≥0∞) =
      ENNReal.ofReal (((N - c).choose k : ℝ) / (N.choose k : ℝ)) := by
  have hC : (0 : ℝ) < N.choose k := by exact_mod_cast Nat.choose_pos hk
  rw [ENNReal.ofReal_div_of_pos hC, ENNReal.ofReal_natCast, ENNReal.ofReal_natCast]

/-- A product of hypergeometric factors, `∏ s C(N_s − c_s, k_s)/C(N_s, k_s)`, is at most `exp(−∑ s k_s c_s / N_s)`. -/
theorem prod_choose_div_le_exp (N c k : Fin m → ℕ) (hk : ∀ s, k s ≤ N s) (hc : ∀ s, c s ≤ N s) :
    ∏ s, ((N s - c s).choose (k s) : ℝ≥0∞) / ((N s).choose (k s) : ℝ≥0∞) ≤
      ENNReal.ofReal (Real.exp (-∑ s, (k s : ℝ) * c s / N s)) := by
  have hnn : ∀ s, 0 ≤ ((N s - c s).choose (k s) : ℝ) / ((N s).choose (k s) : ℝ) := fun s => by positivity
  rw [Finset.prod_congr rfl fun s _ => choose_div_eq_ofReal (c := c s) (hk s),
    ← ENNReal.ofReal_prod_of_nonneg fun s _ => hnn s]
  refine ENNReal.ofReal_le_ofReal ?_
  rw [← Finset.sum_neg_distrib, Real.exp_sum]
  exact Finset.prod_le_prod₀ (fun s _ => hnn s) fun s _ => choose_div_le_exp (hk s) (hc s)

/-- A product of hypergeometric factors at least `δ > 0` has every factor positive, so `c_s + k_s ≤ N_s`, and
`∑ s k_s c_s / N_s ≤ ln(1/δ)`. -/
theorem risk_le_of_le_prod (N c k : Fin m → ℕ) (hk : ∀ s, k s ≤ N s) (hc : ∀ s, c s ≤ N s) {δ : ℝ≥0∞}
    (hδ : 0 < δ) (h : δ ≤ ∏ s, ((N s - c s).choose (k s) : ℝ≥0∞) / ((N s).choose (k s) : ℝ≥0∞)) :
    (∀ s, c s + k s ≤ N s) ∧ ∑ s, (k s : ℝ) * c s / N s ≤ Real.log δ.toReal⁻¹ := by
  refine ⟨fun s => ?_, ?_⟩
  · by_contra hlt
    have h0 : ((N s - c s).choose (k s) : ℝ≥0∞) / ((N s).choose (k s) : ℝ≥0∞) = 0 := by
      rw [Nat.choose_eq_zero_of_lt (by have := hc s; omega), Nat.cast_zero, ENNReal.zero_div]
    exact absurd (le_zero_iff.1 (h.trans_eq (Finset.prod_eq_zero (Finset.mem_univ s) h0))) hδ.ne'
  · have hle := h.trans (prod_choose_div_le_exp N c k hk hc)
    have hδt : δ.toReal ≤ Real.exp (-∑ s, (k s : ℝ) * c s / N s) :=
      ENNReal.toReal_le_of_le_ofReal (Real.exp_pos _).le hle
    have hpos : 0 < δ.toReal := ENNReal.toReal_pos hδ.ne' (ne_top_of_le_ne_top ENNReal.ofReal_ne_top hle)
    have hl := Real.log_le_log hpos hδt
    rw [Real.log_exp] at hl
    rw [Real.log_inv]
    linarith

/-- The exact form: with every `c_s + k_s ≤ N_s`, a product of hypergeometric factors at least `δ > 0` has
`∑ s ln(C(N_s, k_s)/C(N_s − c_s, k_s)) ≤ ln(1/δ)`. -/
theorem risk_exact_le_of_le_prod (N c k : Fin m → ℕ) (hk : ∀ s, k s ≤ N s) (hcap : ∀ s, c s + k s ≤ N s)
    {δ : ℝ≥0∞} (hδ : 0 < δ) (h : δ ≤ ∏ s, ((N s - c s).choose (k s) : ℝ≥0∞) / ((N s).choose (k s) : ℝ≥0∞)) :
    ∑ s, Real.log (((N s).choose (k s) : ℝ) / ((N s - c s).choose (k s) : ℝ)) ≤ Real.log δ.toReal⁻¹ := by
  have hr : ∀ s, 0 < ((N s - c s).choose (k s) : ℝ) / ((N s).choose (k s) : ℝ) := fun s =>
    div_pos (by exact_mod_cast Nat.choose_pos (by have := hcap s; omega))
      (by exact_mod_cast Nat.choose_pos (hk s))
  rw [Finset.prod_congr rfl fun s _ => choose_div_eq_ofReal (c := c s) (hk s),
    ← ENNReal.ofReal_prod_of_nonneg fun s _ => (hr s).le] at h
  have hpos : 0 < δ.toReal := ENNReal.toReal_pos hδ.ne' (ne_top_of_le_ne_top ENNReal.ofReal_ne_top h)
  have hl := Real.log_le_log hpos (ENNReal.toReal_le_of_le_ofReal (Finset.prod_nonneg fun s _ => (hr s).le) h)
  rw [Real.log_prod (fun s _ => (hr s).ne')] at hl
  have he : ∀ s, Real.log (((N s).choose (k s) : ℝ) / ((N s - c s).choose (k s) : ℝ)) =
      -Real.log (((N s - c s).choose (k s) : ℝ) / ((N s).choose (k s) : ℝ)) := fun s => by
    rw [← Real.log_inv, inv_div]
  rw [Finset.sum_congr rfl fun s _ => he s, Finset.sum_neg_distrib, Real.log_inv]
  linarith

/-! ## The fractional knapsack -/

/-- **Weak duality**: counts with `c_s + k_s ≤ N_s` and `∑ s k_s c_s / N_s ≤ b` have `∑ s g_s c_s` at most
`τ b + ∑ s (N_s − k_s) max(0, g_s − τ k_s / N_s)`, for every `τ ≥ 0`. -/
theorem sum_le_dual (N c k : Fin m → ℕ) (g : Fin m → ℝ) {b τ : ℝ} (hτ : 0 ≤ τ) (hcap : ∀ s, c s + k s ≤ N s)
    (hb : ∑ s, (k s : ℝ) * c s / N s ≤ b) :
    ∑ s, g s * c s ≤ τ * b + ∑ s, ((N s - k s : ℕ) : ℝ) * max 0 (g s - τ * k s / N s) := by
  have hsplit : ∑ s, g s * c s = τ * ∑ s, (k s : ℝ) * c s / N s + ∑ s, (g s - τ * k s / N s) * c s := by
    rw [Finset.mul_sum, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun s _ => by ring
  rw [hsplit]
  refine add_le_add (mul_le_mul_of_nonneg_left hb hτ) (Finset.sum_le_sum fun s _ => ?_)
  have hc : (c s : ℝ) ≤ ((N s - k s : ℕ) : ℝ) := by exact_mod_cast (by have := hcap s; omega : c s ≤ N s - k s)
  calc (g s - τ * k s / N s) * c s ≤ max 0 (g s - τ * k s / N s) * c s :=
        mul_le_mul_of_nonneg_right (le_max_right _ _) (Nat.cast_nonneg _)
    _ ≤ max 0 (g s - τ * k s / N s) * ((N s - k s : ℕ) : ℝ) :=
        mul_le_mul_of_nonneg_left hc (le_max_left _ _)
    _ = _ := mul_comm _ _

/-- **Complementary slackness**: a fill `x` that `τ` separates (`x_s = N_s − k_s` where `g_s > τ k_s / N_s`, `x_s = 0`
where `g_s < τ k_s / N_s`, and `∑ s k_s x_s / N_s = b` unless `τ = 0`) has value `∑ s g_s x_s` equal to the dual bound
at `τ`. -/
theorem fill_eq_dual (N k : Fin m → ℕ) (g x : Fin m → ℝ) {b τ : ℝ}
    (hfull : ∀ s, τ * k s / N s < g s → x s = ((N s - k s : ℕ) : ℝ))
    (hempty : ∀ s, g s < τ * k s / N s → x s = 0)
    (hspent : τ = 0 ∨ ∑ s, (k s : ℝ) * x s / N s = b) :
    ∑ s, g s * x s = τ * b + ∑ s, ((N s - k s : ℕ) : ℝ) * max 0 (g s - τ * k s / N s) := by
  have hsplit : ∑ s, g s * x s = τ * ∑ s, (k s : ℝ) * x s / N s + ∑ s, (g s - τ * k s / N s) * x s := by
    rw [Finset.mul_sum, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun s _ => by ring
  have hτb : τ * ∑ s, (k s : ℝ) * x s / N s = τ * b := by
    rcases hspent with h | h
    · simp [h]
    · rw [h]
  rw [hsplit, hτb]
  congr 1
  refine Finset.sum_congr rfl fun s _ => ?_
  rcases lt_trichotomy (g s) (τ * k s / N s) with hlt | heq | hgt
  · rw [hempty s hlt, max_eq_left (by linarith)]
    ring
  · rw [heq, sub_self, max_self]
    ring
  · rw [hfull s hgt, max_eq_right (by linarith)]
    ring

/-- A set's harm, with harm `g_s` per unit of stratum `s`, is `∑ s g_s c_s`. -/
theorem sum_comp_eq (σ : Fin n → Fin m) (g : Fin m → ℝ) (B : Finset (Fin n)) :
    ∑ u ∈ B, g (σ u) = ∑ s, g s * (B.filter (σ · = s)).card := by
  rw [← Finset.sum_fiberwise B σ fun u => g (σ u)]
  refine Finset.sum_congr rfl fun s _ => ?_
  rw [Finset.sum_congr rfl fun u hu => congrArg g (Finset.mem_filter.1 hu).2, Finset.sum_const, nsmul_eq_mul,
    mul_comm]

theorem card_filter_le_stratum (σ : Fin n → Fin m) (s : Fin m) (B : Finset (Fin n)) :
    (B.filter (σ · = s)).card ≤ (stratum σ s).card := by
  rw [filter_eq_inter_stratum]
  exact card_le_card inter_subset_right

/-! ## The stratified law -/

/-- **The closed form**: drawing `k_s` of the `n_s` units of stratum `s` uniformly without replacement, a set with `c_s`
units in stratum `s` escapes with probability at most `exp(−∑ s k_s c_s / n_s)`. -/
theorem stratified_escape_le_exp (σ : Fin n → Fin m) (k : Fin m → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card)
    (B : Finset (Fin n)) :
    (stratified σ k hk).escape B ≤
      ENNReal.ofReal (Real.exp (-∑ s, (k s : ℝ) * (B.filter (σ · = s)).card / (stratum σ s).card)) := by
  rw [stratified_escape]
  exact prod_choose_div_le_exp (fun s => (stratum σ s).card) (fun s => (B.filter (σ · = s)).card) k hk
    (fun s => card_filter_le_stratum σ s B)

/-- **The work law in closed form**: a set with `c_s` units in stratum `s` escapes with probability at most
`exp(−∑ s k_s c_s / n_s)`, `k_s = workK σ w f K s`, at any floors. -/
theorem work_escape_le_exp (σ : Fin n → Fin m) (w f : Fin m → ℕ) (K : ℕ) (B : Finset (Fin n)) :
    (work σ w f K).escape B ≤
      ENNReal.ofReal (Real.exp (-∑ s, (workK σ w f K s : ℝ) * (B.filter (σ · = s)).card / (stratum σ s).card)) :=
  stratified_escape_le_exp σ (workK σ w f K) (workK_le σ w f K) B

/-- A set escaping with probability at least `δ > 0` holds at most `n_s − k_s` units of stratum `s`, and its closed-form
risk `∑ s k_s c_s / n_s` is at most `ln(1/δ)`. -/
theorem stratified_risk_le (σ : Fin n → Fin m) (k : Fin m → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card) {δ : ℝ≥0∞}
    (hδ : 0 < δ) {B : Finset (Fin n)} (hB : δ ≤ (stratified σ k hk).escape B) :
    (∀ s, (B.filter (σ · = s)).card + k s ≤ (stratum σ s).card) ∧
      ∑ s, (k s : ℝ) * (B.filter (σ · = s)).card / (stratum σ s).card ≤ Real.log δ.toReal⁻¹ := by
  rw [stratified_escape] at hB
  exact risk_le_of_le_prod (fun s => (stratum σ s).card) (fun s => (B.filter (σ · = s)).card) k hk
    (fun s => card_filter_le_stratum σ s B) hδ hB

/-- **The harm bound at a price.** With harm `g_s` per unit of stratum `s`, for every `τ ≥ 0`,
`τ ln(1/δ) + ∑ s (n_s − k_s) max(0, g_s − τ k_s / n_s)` is a harm bound at `δ > 0`. -/
theorem stratified_isHarmBound_dual (σ : Fin n → Fin m) (k : Fin m → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card)
    (g : Fin m → ℝ) {δ : ℝ≥0∞} (hδ : 0 < δ) {τ : ℝ} (hτ : 0 ≤ τ) :
    IsHarmBound (stratified σ k hk) (fun u => g (σ u)) δ (τ * Real.log δ.toReal⁻¹ +
      ∑ s, (((stratum σ s).card - k s : ℕ) : ℝ) * max 0 (g s - τ * k s / (stratum σ s).card)) := by
  intro B hB
  obtain ⟨hcap, hrisk⟩ := stratified_risk_le σ k hk hδ hB
  show ∑ u ∈ B, g (σ u) ≤ _
  rw [sum_comp_eq]
  exact sum_le_dual (fun s => (stratum σ s).card) (fun s => (B.filter (σ · = s)).card) k g hτ hcap hrisk

/-- **The knapsack's value is a harm bound.** With harm `g_s` per unit of stratum `s`, a fill `x` that some `τ ≥ 0`
separates has value `∑ s g_s x_s`, which is a harm bound at `δ > 0`. The greedy fill (`density_bound`) is one. -/
theorem stratified_isHarmBound_fill (σ : Fin n → Fin m) (k : Fin m → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card)
    (g x : Fin m → ℝ) {δ : ℝ≥0∞} (hδ : 0 < δ) {τ : ℝ} (hτ : 0 ≤ τ)
    (hfull : ∀ s, τ * k s / (stratum σ s).card < g s → x s = (((stratum σ s).card - k s : ℕ) : ℝ))
    (hempty : ∀ s, g s < τ * k s / (stratum σ s).card → x s = 0)
    (hspent : τ = 0 ∨ ∑ s, (k s : ℝ) * x s / (stratum σ s).card = Real.log δ.toReal⁻¹) :
    IsHarmBound (stratified σ k hk) (fun u => g (σ u)) δ (∑ s, g s * x s) := by
  rw [fill_eq_dual (fun s => (stratum σ s).card) k g x hfull hempty hspent]
  exact stratified_isHarmBound_dual σ k hk g hδ hτ

/-- **Free compute under the work law**: with harm a unit's credited work `w_s`, a separated fill's value
`∑ s w_s x_s` bounds the work `workOf σ w B` of every set escaping with probability at least `δ > 0`. -/
theorem work_isHarmBound_fill (σ : Fin n → Fin m) (w f : Fin m → ℕ) (K : ℕ) (x : Fin m → ℝ) {δ : ℝ≥0∞} (hδ : 0 < δ)
    {τ : ℝ} (hτ : 0 ≤ τ)
    (hfull : ∀ s, τ * workK σ w f K s / (stratum σ s).card < w s →
      x s = (((stratum σ s).card - workK σ w f K s : ℕ) : ℝ))
    (hempty : ∀ s, (w s : ℝ) < τ * workK σ w f K s / (stratum σ s).card → x s = 0)
    (hspent : τ = 0 ∨ ∑ s, (workK σ w f K s : ℝ) * x s / (stratum σ s).card = Real.log δ.toReal⁻¹) :
    IsHarmBound (work σ w f K) (fun u => (w (σ u) : ℝ)) δ (∑ s, (w s : ℝ) * x s) :=
  stratified_isHarmBound_fill σ (workK σ w f K) (workK_le σ w f K) (fun s => (w s : ℝ)) x hδ hτ hfull hempty hspent

/-- Harm in whole bits: the floor of a harm bound is one. -/
theorem _root_.FlockSoundness.Audit.IsHarmBound.natFloor {L : Law n} {b : Fin n → ℕ} {δ : ℝ≥0∞} {H : ℝ}
    (hH : IsHarmBound L (fun u => (b u : ℝ)) δ H) : IsHarmBound L (fun u => (b u : ℝ)) δ (⌊H⌋₊ : ℕ) := by
  intro B hB
  have h : ((∑ u ∈ B, b u : ℕ) : ℝ) ≤ H := by
    push_cast
    exact hH B hB
  show ∑ u ∈ B, (b u : ℝ) ≤ ((⌊H⌋₊ : ℕ) : ℝ)
  exact_mod_cast Nat.le_floor h

/-- **The exfiltration form**: with `b_s` output bits per unit of stratum `s`, the floor of a separated fill's value is a
harm bound at `δ > 0`, the `H` that `audit_exfiltration` takes. -/
theorem stratified_isHarmBound_bits (σ : Fin n → Fin m) (k : Fin m → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card)
    (b : Fin m → ℕ) (x : Fin m → ℝ) {δ : ℝ≥0∞} (hδ : 0 < δ) {τ : ℝ} (hτ : 0 ≤ τ)
    (hfull : ∀ s, τ * k s / (stratum σ s).card < b s → x s = (((stratum σ s).card - k s : ℕ) : ℝ))
    (hempty : ∀ s, (b s : ℝ) < τ * k s / (stratum σ s).card → x s = 0)
    (hspent : τ = 0 ∨ ∑ s, (k s : ℝ) * x s / (stratum σ s).card = Real.log δ.toReal⁻¹) :
    IsHarmBound (stratified σ k hk) (fun u => (b (σ u) : ℝ)) δ (⌊∑ s, (b s : ℝ) * x s⌋₊ : ℕ) :=
  IsHarmBound.natFloor (b := fun u => b (σ u))
    (stratified_isHarmBound_fill σ k hk (fun s => (b s : ℝ)) x hδ hτ hfull hempty hspent)

/-! ## The exact risk -/

/-- Stratum `s`'s exact risk with `c` of its units wrong, `ln(C(n_s, k_s)/C(n_s − c, k_s))`: minus the log of its escape. -/
noncomputable def stratumRisk (σ : Fin n → Fin m) (k : Fin m → ℕ) (s : Fin m) (c : ℕ) : ℝ :=
  Real.log (((stratum σ s).card.choose (k s) : ℝ) / (((stratum σ s).card - c).choose (k s) : ℝ))

/-- The exact risk is at least the closed form's: `k_s c / n_s ≤ stratumRisk σ k s c` for `c + k_s ≤ n_s`. -/
theorem le_stratumRisk (σ : Fin n → Fin m) (k : Fin m → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card) (s : Fin m) {c : ℕ}
    (hc : c + k s ≤ (stratum σ s).card) :
    (k s : ℝ) * c / (stratum σ s).card ≤ stratumRisk σ k s c := by
  have hr : 0 < (((stratum σ s).card - c).choose (k s) : ℝ) / ((stratum σ s).card.choose (k s) : ℝ) :=
    div_pos (by exact_mod_cast Nat.choose_pos (by omega)) (by exact_mod_cast Nat.choose_pos (hk s))
  have hl := Real.log_le_log hr (choose_div_le_exp (hk s) (by omega : c ≤ (stratum σ s).card))
  rw [Real.log_exp] at hl
  have he : stratumRisk σ k s c =
      -Real.log ((((stratum σ s).card - c).choose (k s) : ℝ) / ((stratum σ s).card.choose (k s) : ℝ)) := by
    unfold stratumRisk
    rw [← Real.log_inv, inv_div]
  rw [he]
  linarith

/-- **The harm bound at a price, exact.** For every harm `h`, every `τ ≥ 0` and every `c` bounding each stratum's
`∑ u ∈ A, h u − τ · stratumRisk σ k s |A|` over its sets `A` with `|A| + k_s ≤ n_s`, `τ ln(1/δ) + ∑ s c_s` is a harm
bound at `δ > 0`. -/
theorem stratified_isHarmBound_exact (σ : Fin n → Fin m) (k : Fin m → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card)
    (h : Fin n → ℝ) {δ : ℝ≥0∞} (hδ : 0 < δ) {τ : ℝ} (hτ : 0 ≤ τ) (c : Fin m → ℝ)
    (hc : ∀ s (A : Finset (Fin n)), A ⊆ stratum σ s → A.card + k s ≤ (stratum σ s).card →
      ∑ u ∈ A, h u - τ * stratumRisk σ k s A.card ≤ c s) :
    IsHarmBound (stratified σ k hk) h δ (τ * Real.log δ.toReal⁻¹ + ∑ s, c s) := by
  intro B hB
  obtain ⟨hcap, -⟩ := stratified_risk_le σ k hk hδ hB
  rw [stratified_escape] at hB
  have hrisk : ∑ s, stratumRisk σ k s (B.filter (σ · = s)).card ≤ Real.log δ.toReal⁻¹ :=
    risk_exact_le_of_le_prod (fun s => (stratum σ s).card) (fun s => (B.filter (σ · = s)).card) k hk hcap hδ hB
  calc ∑ u ∈ B, h u = ∑ s, (∑ u ∈ B.filter (σ · = s), h u - τ * stratumRisk σ k s (B.filter (σ · = s)).card) +
        τ * ∑ s, stratumRisk σ k s (B.filter (σ · = s)).card := by
          rw [← Finset.sum_fiberwise B σ h, Finset.mul_sum, ← Finset.sum_add_distrib]
          exact Finset.sum_congr rfl fun s _ => by ring
    _ ≤ ∑ s, c s + τ * Real.log δ.toReal⁻¹ :=
        add_le_add (Finset.sum_le_sum fun s _ => hc s _ (fun u hu => mem_stratum.2 (mem_filter.1 hu).2) (hcap s))
          (mul_le_mul_of_nonneg_left hrisk hτ)
    _ = _ := add_comm _ _

end Law

namespace Partition

variable {C : Circuit} {n m : ℕ} (P : Partition C n)

/-- **Exfiltration under the stratified law**: `audit_exfiltration` at the harm bound of a separated fill of the output
bits `b_s` per unit of stratum `s`. A prover that picks its strategy from a uniform message and a receiver that sees only
what is delivered on `D` succeed together with probability at most
`2^⌊∑ s b_s x_s⌋ · ∑_{i ≤ j} C(n, i) / |M| + δ + ε_ks + δ_link + δ_in`, where `miss (j + 1) < δ`. -/
theorem audit_exfiltration_stratified {σs : Fin n → Fin m} {k : Fin m → ℕ}
    {hk : ∀ s, k s ≤ (Law.stratum σs s).card} {Reg : Type} {session : Finset (Fin n) → Reg → Game Bool}
    {X : (R : Reg) → Cont (Law.stratified σs k hk) Reg session R → (Fin C.N → Bool)}
    {ext : (S : Finset (Fin n)) → (R : Reg) → Strategy (session S R) → Fin n → Option (Fin C.N → Bool)}
    {a : Fin C.N → Bool} {εks δlink δin : ℝ≥0∞} {D : Finset (Fin C.N)} {M : Type} [Fintype M] [Nonempty M]
    [DecidableEq M] (hks : (P.analysis X ext).KnowledgeSound εks) (hlink : (P.analysis X ext).LinkSound δlink)
    (hin : AnchorsSound X a δin) (hD : D ⊆ P.committed) (b : Fin m → ℕ) (K : Finset (Fin n) → Finset (Fin C.N))
    (hK : ∀ B, K B ⊆ P.committed) (hsep : ∀ B, P.Separates (K B) B D) (hw : ∀ B, (K B).card ≤ ∑ u ∈ B, b (σs u))
    (x : Fin m → ℝ) {δ : ℝ≥0∞} (hδ : 0 < δ) {τ : ℝ} (hτ : 0 ≤ τ)
    (hfull : ∀ s, τ * k s / (Law.stratum σs s).card < b s → x s = (((Law.stratum σs s).card - k s : ℕ) : ℝ))
    (hempty : ∀ s, (b s : ℝ) < τ * k s / (Law.stratum σs s).card → x s = 0)
    (hspent : τ = 0 ∨ ∑ s, (k s : ℝ) * x s / (Law.stratum σs s).card = Real.log δ.toReal⁻¹)
    {j : ℕ} (hj : (Law.stratified σs k hk).miss (j + 1) < δ)
    (σ : M → Strategy (audit (Law.stratified σs k hk) Reg session)) (g : (D → Bool) → M) :
    avg (fun msg => prob (fun o => o.1 = true ∧ g (out D (X (reg (σ msg)) (cont (σ msg)))) = msg)
        (audit (Law.stratified σs k hk) Reg session) (σ msg)) ≤
      ((2 ^ ⌊∑ s, (b s : ℝ) * x s⌋₊ * ∑ i ∈ range (j + 1), n.choose i : ℕ) : ℝ≥0∞) / (Fintype.card M : ℝ≥0∞) +
        (δ + εks + δlink + δin) :=
  P.audit_exfiltration hks hlink hin hD (fun u => b (σs u)) K hK hsep hw
    (Law.stratified_isHarmBound_bits σs k hk b x hδ hτ hfull hempty hspent) hj σ g

end Partition

end FlockSoundness.Audit
