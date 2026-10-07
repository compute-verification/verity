import Definitions.Pous.P2.Cost
import Proofs.Pous.P2.Domain

/-!
# Slack rows: an adversary with `λ` extra state bits per block

`MeetsWithSlack sch reveal λ D Q k ε` (`Security/Definitions/Pous/P2/Cost.lean`) is `Meets` with the state bound
`⌊ρ·|C|⌋` raised by `λ·B` bits. At `λ = 0` it is `Meets` (`meetsWithSlack_zero`). The route is `meets_m1p_of_row`'s at a
state of `⌊ρ·|C|⌋ + λ·B` bits and a target error `2^-e` (`P2RowE`, `slack_of_row`). The rows at `B = 2^19` are certified
with the largest `g` the exact 1% inequality allows (`row_b19_*`), and the families with a uniform `g = ⌊c·B⌋ + 1`
(`row_uniform_gen`). -/

namespace PousMargin

open Pous Pous.Assumptions PousP2 PousP2p Finset
open scoped ENNReal

theorem meetsWithSlack_zero {M : IdealModel} (sch : Scheme M) (reveal : Reveal) (D Q k : ℕ) (ε : ℝ≥0∞) :
    MeetsWithSlack sch reveal 0 D Q k ε ↔ Meets sch reveal D Q k ε := by
  simp only [MeetsWithSlack, Meets, zero_mul, add_zero]

/-- A bound at a smaller `ε` gives the bound at a larger one. -/
theorem meetsWithSlack_mono {M : IdealModel} {sch : Scheme M} {lam D Q k : ℕ} {ε ε' : ℝ≥0∞}
    (h : MeetsWithSlack sch .sequential lam D Q k ε) (hε : ε ≤ ε') (hε' : ε' ≤ Params.εMax) :
    MeetsWithSlack sch .sequential lam D Q k ε' :=
  ⟨hε', h.2.1, h.2.2.1, h.2.2.2.1, fun W A₁ A₂ => (h.2.2.2.2 W A₁ A₂).trans (add_le_add le_rfl hε)⟩

theorem two_inv_pow_le {e e' : ℕ} (h : e ≤ e') : (2⁻¹ : ℝ≥0∞) ^ e' ≤ (2⁻¹ : ℝ≥0∞) ^ e :=
  pow_le_pow_right_of_le_one' (by norm_num) h

theorem eps129_le : (2⁻¹ : ℝ≥0∞) ^ 129 ≤ Params.εMax := two_inv_pow_le (by norm_num [Params.secParam])

/-- `P2Row` with the target `|Hist|·Γ ≤ 2^-e` in place of `2^-128`. -/
def P2RowE (B ℓ S T k e : ℕ) : Prop :=
  ∃ g, 1 ≤ g ∧ g ≤ B ∧ (((g - 1 : ℕ) : ℚ) / B) ^ k ≤ 1 / 100 ∧
    logHistGammaM1 B S T g k (2 ^ ℓ) ≤ -(e : ℝ)

/-- `hist_gamma_le` at a target `2^-e`. -/
theorem hist_gamma_le_e (B ℓ S T g k e : ℕ) (hk : 1 ≤ k) (hgB : g ≤ B)
    (h : logHistGammaM1 B S T g k (2 ^ ℓ) ≤ -(e : ℝ)) :
    (∑ j ∈ Finset.range k, (B : ℝ≥0∞) ^ j) * gammaA2 B ℓ S T g ≤ (2⁻¹ : ℝ≥0∞) ^ e := by
  set hist : ℝ := ((histCount B k : ℕ) : ℝ)
  set c : ℝ := ((B.choose g : ℕ) : ℝ)
  set s : ℝ := ∑ h ∈ Finset.range (g + 1), (((g + 1) * T).choose h : ℝ)
  have hhist : 0 < hist := Nat.cast_pos.2 (histCount_pos B k hk)
  have hc : 0 < c := Nat.cast_pos.2 (Nat.choose_pos hgB)
  have hs : 0 < s := by
    simp only [s]
    rw [Finset.sum_range_succ']
    have : (0 : ℝ) ≤ ∑ h ∈ Finset.range g, (((g + 1) * T).choose (h + 1) : ℝ) := by positivity
    simp only [Nat.choose_zero_right, Nat.cast_one]
    linarith
  set X : ℝ := hist * ((2 : ℝ) ^ S * c * s / (2 : ℝ) ^ (ℓ * g))
  have hX0 : 0 < X := by positivity
  have hlogX : Real.logb 2 X = logHistGammaM1 B S T g k (2 ^ ℓ) := by
    unfold logHistGammaM1
    simp only [X]
    rw [Real.logb_mul hhist.ne' (by positivity), Real.logb_div (by positivity) (by positivity),
      Real.logb_mul (by positivity) hs.ne', Real.logb_mul (by positivity) hc.ne', Real.logb_pow, Real.logb_pow,
      Real.logb_self_eq_one one_lt_two, Nat.cast_pow, Nat.cast_ofNat, Real.logb_pow,
      Real.logb_self_eq_one one_lt_two]
    push_cast
    ring
  have hXle : X ≤ (2⁻¹ : ℝ) ^ e := by
    rw [← Real.rpow_logb (by norm_num : (0 : ℝ) < 2) (by norm_num) hX0, hlogX]
    calc (2 : ℝ) ^ logHistGammaM1 B S T g k (2 ^ ℓ) ≤ (2 : ℝ) ^ (-(e : ℝ)) :=
          Real.rpow_le_rpow_of_exponent_le (by norm_num) h
      _ = (2⁻¹ : ℝ) ^ e := by
          rw [Real.rpow_neg (by norm_num), Real.rpow_natCast, inv_pow]
  have hL : (∑ j ∈ Finset.range k, (B : ℝ≥0∞) ^ j) * gammaA2 B ℓ S T g = ENNReal.ofReal X := by
    have hsum : (∑ j ∈ Finset.range k, (B : ℝ≥0∞) ^ j) = ENNReal.ofReal hist := by
      simp only [hist, histCount, Nat.cast_sum, Nat.cast_pow]
      rw [ENNReal.ofReal_sum_of_nonneg (fun _ _ => by positivity)]
      refine Finset.sum_congr rfl fun j _ => ?_
      rw [ENNReal.ofReal_pow (Nat.cast_nonneg _), ENNReal.ofReal_natCast]
    have hs' : (∑ h ∈ Finset.range (g + 1), (((g + 1) * T).choose h : ℝ≥0∞)) = ENNReal.ofReal s := by
      simp only [s]
      rw [ENNReal.ofReal_sum_of_nonneg (fun _ _ => by positivity)]
      exact Finset.sum_congr rfl fun h _ => (ENNReal.ofReal_natCast _).symm
    simp only [X, gammaA2, hsum, hs']
    rw [ENNReal.ofReal_mul hhist.le, ENNReal.ofReal_div_of_pos (by positivity), ENNReal.ofReal_mul (by positivity),
      ENNReal.ofReal_mul (by positivity), ENNReal.ofReal_pow (by norm_num), ENNReal.ofReal_pow (by norm_num),
      ENNReal.ofReal_ofNat, ENNReal.ofReal_natCast]
    simp only [c, ENNReal.ofReal_natCast]
  rw [hL, ← ENNReal.ofReal_ofNat 2, ← ENNReal.ofReal_inv_of_pos (by norm_num), ← ENNReal.ofReal_pow (by norm_num)]
  exact ENNReal.ofReal_le_ofReal hXle

/-- **`m1p` meets the requirement with `lam` bits of slack per block at `ε = 2^-e`, given a row at the raised state
plus one bit.** `meets_m1p_of_row` with the state `⌊ρ·|C|⌋ + lam·B` and the target `2^-e`, `e ≥ 128`. -/
theorem slack_of_row (B w p b c Q k lam e : ℕ) [NeZero B] (hb : 2 ^ b ≤ p) (hp2 : p ≤ 2 ^ w)
    (hc : 2 ^ w ≤ p + c) (hBc : 2 * B * c ≤ 2 ^ w) (hsp : 20 * w ≤ 21 * b) (hk : 1 ≤ k) (he : 128 ≤ e)
    (hrow : P2RowE B w ((m1p B w p b).stateBound Params.ρ + lam * B + 1) Q k e) (D : ℕ) :
    MeetsWithSlack (m1p B w p b) .sequential lam D Q k ((2⁻¹ : ℝ≥0∞) ^ e) := by
  obtain ⟨g, -, hgB, hpow, hlog⟩ := hrow
  have hp0 : 0 < p := lt_of_lt_of_le (Nat.two_pow_pos _) hb
  have hbw : b ≤ w := le_of_two_pow_le (hb.trans hp2)
  refine ⟨two_inv_pow_le he, m1p_correct B w p b hbw hp2, m1p_codeHoldsW B w p b hbw, m1p_spaceBound B w p b hsp,
    ?_⟩
  show AuditSecure (m1p B w p b) .sequential ((m1p B w p b).stateBound Params.ρ + lam * B) D Q k _ _
  intro W A₁ A₂
  have hgc : 2 * g * c ≤ 2 ^ w := le_trans (by gcongr) hBc
  have hΓ : (∑ j ∈ Finset.range k, (B : ℝ≥0∞) ^ j) *
      gammaP B p ((m1p B w p b).stateBound Params.ρ + lam * B) Q g ≤ (2⁻¹ : ℝ≥0∞) ^ e :=
    (mul_le_mul_right (gammaP_le B w p c _ Q g hp0 hc hgc) _).trans (hist_gamma_le_e B w _ Q g k e hk hgB hlog)
  have h := PousTrackA.a3_prime_lam (m1p B w p b) _ D Q k g e _ (fun _ _ => True)
    (lemmaA_m1p B w p b ((m1p B w p b).stateBound Params.ρ + lam * B) D Q g hb hp2) hΓ W A₁ A₂
  refine h.trans ?_
  simp only [not_true_eq_false, pr_false, add_zero]
  have hB0 : (0 : ℝ) < B := by exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne B)
  have hδ : ((((g - 1 : ℕ) : ℝ≥0∞) / (m1p B w p b).B) ^ k) ≤ ENNReal.ofReal Params.δ := by
    change ((((g - 1 : ℕ) : ℝ≥0∞) / (B : ℝ≥0∞)) ^ k) ≤ ENNReal.ofReal ((1 / 100 : ℚ) : ℝ)
    rw [← ENNReal.ofReal_natCast, ← ENNReal.ofReal_natCast B, ← ENNReal.ofReal_div_of_pos hB0,
      ← ENNReal.ofReal_pow (by positivity)]
    refine ENNReal.ofReal_le_ofReal ?_
    have := (Rat.cast_le (K := ℝ)).2 hpow
    push_cast at this ⊢
    exact this
  exact add_le_add hδ le_rfl

/-! ### The uniform certificate with slack -/

open Real in
/-- **The uniform row at `w = 16,448`, `T = 2^20`, with `lam` bits of slack.** With `g = ⌊c·B⌋ + 1`, the row at
`S + 1` holds when `c^k ≤ 1/100`, `S ≤ ((18/19)·16448 + lam)·B`, `B ≤ 2^n` and `k·n + e + 3 ≤ m·B`, where the
per-block margin `m` is at most `c·(16448 − 20 − 1.4427) − (18/19)·16448 − lam − 1`. -/
theorem row_uniform_gen (B n S k e lam : ℕ) (c m : ℚ) (hB2 : 2 ≤ B) (hBn : B ≤ 2 ^ n) (hk : 1 ≤ k)
    (hc0 : 0 ≤ c) (hc1 : c < 1) (hck : c ^ k ≤ 1 / 100)
    (hm : m ≤ c * (16448 - 20 - 14427 / 10000) - 18 / 19 * 16448 - lam - 1)
    (hslack : (k * n + e + 3 : ℚ) ≤ m * B) (hS : (S : ℝ) ≤ (18 / 19 * 16448 + lam) * (B : ℝ)) :
    P2RowE B 16448 (S + 1) (2 ^ 20) k e := by
  obtain ⟨g, hg⟩ : ∃ g, g = ⌊c * B⌋₊ + 1 := ⟨_, rfl⟩
  have hcB0 : 0 ≤ c * B := by positivity
  have hB0 : (0 : ℚ) < B := by exact_mod_cast (by omega : 0 < B)
  have hgB : g ≤ B := by
    have : ⌊c * B⌋₊ < B := (Nat.floor_lt hcB0).2 (by nlinarith)
    omega
  refine ⟨g, by omega, hgB, ?_, ?_⟩
  · have hle : ((g - 1 : ℕ) : ℚ) / B ≤ c := by
      rw [div_le_iff₀ hB0, show g - 1 = ⌊c * B⌋₊ by omega]
      exact Nat.floor_le hcB0
    calc (((g - 1 : ℕ) : ℚ) / B) ^ k ≤ c ^ k := pow_le_pow_left₀ (by positivity) hle k
      _ ≤ 1 / 100 := hck
  · refine (logHistGammaM1_le _ _ _ _ _ _ hB2 hk (by omega) hgB (by norm_num)).trans ?_
    rw [logb_two_pow_nat]
    have hlogB : logb 2 (B : ℝ) ≤ n := by
      have := logb_le_logb_of_le one_lt_two (by exact_mod_cast (by omega : 0 < B))
        (show (B : ℝ) ≤ ((2 ^ n : ℕ) : ℝ) by exact_mod_cast hBn)
      rwa [logb_two_pow_nat] at this
    have hkn : (k : ℝ) * logb 2 B ≤ k * n := mul_le_mul_of_nonneg_left hlogB (Nat.cast_nonneg _)
    have hgc : c * B < ((⌊c * B⌋₊ : ℕ) : ℚ) + 1 := Nat.lt_floor_add_one _
    have hgR : (c : ℝ) * B < g := by
      have := (Rat.cast_lt (K := ℝ)).2 hgc
      push_cast [hg] at this ⊢
      linarith
    have hl := log_two_gt_d9
    have hL : ((g : ℝ) + 1) / log 2 ≤ ((g : ℝ) + 1) * (14427 / 10000) := by
      rw [div_le_iff₀ (by positivity)]
      have h1 : (1 : ℝ) ≤ 14427 / 10000 * log 2 := by norm_num at hl ⊢; linarith
      calc ((g : ℝ) + 1) = ((g : ℝ) + 1) * 1 := (mul_one _).symm
        _ ≤ ((g : ℝ) + 1) * (14427 / 10000 * log 2) := mul_le_mul_of_nonneg_left h1 (by positivity)
        _ = _ := by ring
    have hmR : (m : ℝ) ≤ c * (16448 - 20 - 14427 / 10000) - 18 / 19 * 16448 - lam - 1 := by
      have := (Rat.cast_le (K := ℝ)).2 hm
      push_cast at this
      exact this
    have hslR : (k : ℝ) * n + e + 3 ≤ m * B := by
      have := (Rat.cast_le (K := ℝ)).2 hslack
      push_cast at this
      exact this
    have hmB : (m : ℝ) * B ≤ (c * (16448 - 20 - 14427 / 10000) - 18 / 19 * 16448 - lam - 1) * B :=
      mul_le_mul_of_nonneg_right hmR (Nat.cast_nonneg _)
    have hgR' : (c : ℝ) * B * (16448 - 20 - 14427 / 10000) ≤ g * (16448 - 20 - 14427 / 10000) :=
      mul_le_mul_of_nonneg_right hgR.le (by norm_num)
    push_cast at hL ⊢
    nlinarith

/-- `m1p`'s raised state bound: `⌊(18/19)·B·w⌋ + lam·B ≤ ((18/19)·16448 + lam)·B` at `w = 16,448`. -/
theorem stateBound_slack_le (B p b lam : ℕ) [NeZero B] :
    ((((m1p B 16448 p b).stateBound Params.ρ + lam * B : ℕ)) : ℝ) ≤ (18 / 19 * 16448 + lam) * (B : ℝ) := by
  have h := m1p_stateBound_le B 16448 p b
  push_cast at h ⊢
  nlinarith

/-- **The uniform slack row as a `MeetsWithSlack` statement**, for `lo ≤ B ≤ min(2^n, 2^23)` at P2's 16,448-bit
prime, every payload width in range. -/
theorem slack_uniform (B n lo lam k e : ℕ) [NeZero B] (c m : ℚ) (hlo : lo ≤ B) (hB2 : 2 ≤ lo) (hBn : B ≤ 2 ^ n)
    (hB23 : B ≤ 2 ^ 23) (hk : 1 ≤ k) (he : 128 ≤ e) (hc0 : 0 ≤ c) (hc1 : c < 1) (hck : c ^ k ≤ 1 / 100)
    (hm : m ≤ c * (16448 - 20 - 14427 / 10000) - 18 / 19 * 16448 - lam - 1) (hm0 : 0 ≤ m)
    (hslack : (k * n + e + 3 : ℚ) ≤ m * lo) (b : ℕ) (hsp : 20 * 16448 ≤ 21 * b) (hb : b ≤ 16447) (D : ℕ) :
    MeetsWithSlack (m1p B 16448 (2 ^ 16448 - 21065) b) .sequential lam D (2 ^ 20) k ((2⁻¹ : ℝ≥0∞) ^ e) := by
  obtain ⟨hlo', hhi, hc, hBc⟩ := hyps_16448
  refine slack_of_row B 16448 _ b 21065 (2 ^ 20) k lam e (two_pow_le_of_le hb hlo') hhi hc
    (le_trans (by gcongr) hBc) hsp hk he ?_ D
  have hloB : (lo : ℚ) ≤ B := by exact_mod_cast hlo
  exact row_uniform_gen B n _ k e lam c m (by omega) hBn hk hc0 hc1 hck hm
    (hslack.trans (mul_le_mul_of_nonneg_left hloB hm0)) (stateBound_slack_le B _ b lam)

/-! ### The rows at `B = 2^19`, with the largest `g` the 1% inequality allows -/

/-- A row at `2^19` blocks gives `MeetsWithSlack` for every payload width in range. -/
theorem slack_b19 (lam k e : ℕ) (hk : 1 ≤ k) (he : 128 ≤ e)
    (hrow : P2RowE (2 ^ 19) 16448 (8169621180 + lam * 2 ^ 19 + 1) (2 ^ 20) k e) :
    ∀ b, 20 * 16448 ≤ 21 * b → b ≤ 16447 → ∀ D,
      MeetsWithSlack (m1p (2 ^ 19) 16448 (2 ^ 16448 - 21065) b) .sequential lam D (2 ^ 20) k ((2⁻¹ : ℝ≥0∞) ^ e) := by
  intro b hsp hb D
  obtain ⟨hlo, hhi, hc, hBc⟩ := hyps_16448
  refine slack_of_row (2 ^ 19) 16448 _ b 21065 (2 ^ 20) k lam e (two_pow_le_of_le hb hlo) hhi hc
    (le_trans (by gcongr <;> norm_num) hBc) hsp hk he ?_ D
  rw [stateBound_m1p_b19]
  exact hrow

open Real in
/-- The certificate's log bound at `2^19` blocks, `T = 2^20`, `w = 16,448`, evaluated with `log 2 > 0.6931471803`. -/
theorem log_b19 (Sl k g : ℕ) (hk : 1 ≤ k) (hg : 1 ≤ g) (hgB : g ≤ 2 ^ 19) :
    logHistGammaM1 (2 ^ 19) (8169621180 + Sl + 1) (2 ^ 20) g k (2 ^ 16448) ≤
      19 * k + (8169621180 + Sl + 1 : ℕ) + 2 ^ 19 + (20 * g + (g + 1) / 0.6931471803) - g * 16448 := by
  refine (logHistGammaM1_le _ _ _ _ _ _ (by norm_num) hk hg hgB (by norm_num)).trans ?_
  rw [logb_two_pow_nat, logb_two_pow_nat]
  have hl := log_two_gt_d9
  have h : ((g : ℝ) + 1) / log 2 ≤ (g + 1) / 0.6931471803 :=
    div_le_div_of_nonneg_left (by positivity) (by norm_num) hl.le
  push_cast at h ⊢
  linarith

/-- `λ = 0`, `k = 88`, `g = 497,557`: `log₂(|Hist|·Γ) ≤ −3,001,430`. -/
theorem log_b19_0 : logHistGammaM1 (2 ^ 19) (8169621180 + 0 * 2 ^ 19 + 1) (2 ^ 20) 497557 88 (2 ^ 16448) ≤
    -3001430 :=
  (log_b19 _ 88 497557 (by norm_num) (by norm_num) (by norm_num)).trans (by norm_num)

/-- `λ = 64`, `k = 95`, `g = 499,480`: `log₂(|Hist|·Γ) ≤ −1,035,135`. -/
theorem log_b19_64 : logHistGammaM1 (2 ^ 19) (8169621180 + 64 * 2 ^ 19 + 1) (2 ^ 20) 499480 95 (2 ^ 16448) ≤
    -1035135 :=
  (log_b19 _ 95 499480 (by norm_num) (by norm_num) (by norm_num)).trans (by norm_num)

/-- `λ = 128`, `k = 104`, `g = 501,579`: `log₂(|Hist|·Γ) ≤ −1,959,876`. -/
theorem log_b19_128 : logHistGammaM1 (2 ^ 19) (8169621180 + 128 * 2 ^ 19 + 1) (2 ^ 20) 501579 104 (2 ^ 16448) ≤
    -1959876 :=
  (log_b19 _ 104 501579 (by norm_num) (by norm_num) (by norm_num)).trans (by norm_num)

/-- `λ = 256`, `k = 127`, `g = 505,618`: `log₂(|Hist|·Γ) ≤ −1,197,439`. -/
theorem log_b19_256 : logHistGammaM1 (2 ^ 19) (8169621180 + 256 * 2 ^ 19 + 1) (2 ^ 20) 505618 127 (2 ^ 16448) ≤
    -1197439 :=
  (log_b19 _ 127 505618 (by norm_num) (by norm_num) (by norm_num)).trans (by norm_num)

/-- The four rows at `2^19` blocks and target `2^-129`; each `g` is the largest with `((g − 1)/2^19)^k ≤ 1/100`. -/
theorem row_b19_0 : P2RowE (2 ^ 19) 16448 (8169621180 + 0 * 2 ^ 19 + 1) (2 ^ 20) 88 129 :=
  ⟨497557, by norm_num, by norm_num, by norm_num, log_b19_0.trans (by norm_num)⟩

theorem row_b19_64 : P2RowE (2 ^ 19) 16448 (8169621180 + 64 * 2 ^ 19 + 1) (2 ^ 20) 95 129 :=
  ⟨499480, by norm_num, by norm_num, by norm_num, log_b19_64.trans (by norm_num)⟩

theorem row_b19_128 : P2RowE (2 ^ 19) 16448 (8169621180 + 128 * 2 ^ 19 + 1) (2 ^ 20) 104 129 :=
  ⟨501579, by norm_num, by norm_num, by norm_num, log_b19_128.trans (by norm_num)⟩

theorem row_b19_256 : P2RowE (2 ^ 19) 16448 (8169621180 + 256 * 2 ^ 19 + 1) (2 ^ 20) 127 129 :=
  ⟨505618, by norm_num, by norm_num, by norm_num, log_b19_256.trans (by norm_num)⟩

/-- The largest `g` is exact: `g + 1` fails the 1% inequality at each row. -/
theorem row_b19_g_max : ¬ (((497557 : ℚ)) / 2 ^ 19) ^ 88 ≤ 1 / 100 ∧ ¬ (((499480 : ℚ)) / 2 ^ 19) ^ 95 ≤ 1 / 100 ∧
    ¬ (((501579 : ℚ)) / 2 ^ 19) ^ 104 ≤ 1 / 100 ∧ ¬ (((505618 : ℚ)) / 2 ^ 19) ^ 127 ≤ 1 / 100 := by
  norm_num

/-! ### `B = 2^19`: the slack rows at `ε = 2^-129` and `2^-128` -/

section B19

/-- **Error split, `λ = 0`**: `2^19` blocks meet the requirement at `k = 88` with `ε = 2^-129`. -/
theorem b19_0_129 : ∀ b, 20 * 16448 ≤ 21 * b → b ≤ 16447 → ∀ D,
    MeetsWithSlack (m1p (2 ^ 19) 16448 (2 ^ 16448 - 21065) b) .sequential 0 D (2 ^ 20) 88 ((2⁻¹ : ℝ≥0∞) ^ 129) :=
  slack_b19 0 88 129 (by norm_num) (by norm_num) row_b19_0

theorem b19_64_129 : ∀ b, 20 * 16448 ≤ 21 * b → b ≤ 16447 → ∀ D,
    MeetsWithSlack (m1p (2 ^ 19) 16448 (2 ^ 16448 - 21065) b) .sequential 64 D (2 ^ 20) 95 ((2⁻¹ : ℝ≥0∞) ^ 129) :=
  slack_b19 64 95 129 (by norm_num) (by norm_num) row_b19_64

theorem b19_128_129 : ∀ b, 20 * 16448 ≤ 21 * b → b ≤ 16447 → ∀ D,
    MeetsWithSlack (m1p (2 ^ 19) 16448 (2 ^ 16448 - 21065) b) .sequential 128 D (2 ^ 20) 104 ((2⁻¹ : ℝ≥0∞) ^ 129) :=
  slack_b19 128 104 129 (by norm_num) (by norm_num) row_b19_128

theorem b19_256_129 : ∀ b, 20 * 16448 ≤ 21 * b → b ≤ 16447 → ∀ D,
    MeetsWithSlack (m1p (2 ^ 19) 16448 (2 ^ 16448 - 21065) b) .sequential 256 D (2 ^ 20) 127 ((2⁻¹ : ℝ≥0∞) ^ 129) :=
  slack_b19 256 127 129 (by norm_num) (by norm_num) row_b19_256

/-- `λ = 0` at `2^-129` as a `Meets` statement. -/
theorem b19_meets_129 : ∀ b, 20 * 16448 ≤ 21 * b → b ≤ 16447 → ∀ D,
    Meets (m1p (2 ^ 19) 16448 (2 ^ 16448 - 21065) b) .sequential D (2 ^ 20) 88 ((2⁻¹ : ℝ≥0∞) ^ 129) :=
  fun b hsp hb D => (meetsWithSlack_zero _ _ _ _ _ _).1 (b19_0_129 b hsp hb D)

/-- **The slack rows at `2^19` blocks, `ε = 2^-128`.** -/
theorem b19_64 : ∀ b, 20 * 16448 ≤ 21 * b → b ≤ 16447 → ∀ D,
    MeetsWithSlack (m1p (2 ^ 19) 16448 (2 ^ 16448 - 21065) b) .sequential 64 D (2 ^ 20) 95 Params.εMax :=
  fun b hsp hb D => meetsWithSlack_mono (b19_64_129 b hsp hb D) eps129_le le_rfl

theorem b19_128 : ∀ b, 20 * 16448 ≤ 21 * b → b ≤ 16447 → ∀ D,
    MeetsWithSlack (m1p (2 ^ 19) 16448 (2 ^ 16448 - 21065) b) .sequential 128 D (2 ^ 20) 104 Params.εMax :=
  fun b hsp hb D => meetsWithSlack_mono (b19_128_129 b hsp hb D) eps129_le le_rfl

theorem b19_256 : ∀ b, 20 * 16448 ≤ 21 * b → b ≤ 16447 → ∀ D,
    MeetsWithSlack (m1p (2 ^ 19) 16448 (2 ^ 16448 - 21065) b) .sequential 256 D (2 ^ 20) 127 Params.εMax :=
  fun b hsp hb D => meetsWithSlack_mono (b19_256_129 b hsp hb D) eps129_le le_rfl

end B19

/-! ### The families, at `ε = 2^-129` -/

/-- `λ = 0`, `k = 88`, all of `2^8 ≤ B ≤ 2^23` (`c = 0.94901`, margin 5 bits per block). -/
theorem fam_0_129 : ∀ (B : ℕ) [NeZero B], 2 ^ 8 ≤ B → B ≤ 2 ^ 23 → ∀ b, 20 * 16448 ≤ 21 * b → b ≤ 16447 → ∀ D,
    MeetsWithSlack (m1p B 16448 (2 ^ 16448 - 21065) b) .sequential 0 D (2 ^ 20) 88 ((2⁻¹ : ℝ≥0∞) ^ 129) := by
  intro B _ h0 h1 b hsp hb D
  by_cases h : B ≤ 2 ^ 9
  · exact slack_uniform B 9 (2 ^ 8) 0 88 129 (94901 / 100000) 5 h0 (by norm_num) h h1 (by norm_num) (by norm_num)
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) b hsp hb D
  · exact slack_uniform B 23 (2 ^ 9 + 1) 0 88 129 (94901 / 100000) 5 (by omega) (by norm_num) h1 h1 (by norm_num)
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) b hsp hb D

/-- `λ = 64`, `k = 95`, `2^10 ≤ B ≤ 2^23` (`c = 0.95268`, margin 1.9 bits per block). -/
theorem fam_64_129 : ∀ (B : ℕ) [NeZero B], 2 ^ 10 ≤ B → B ≤ 2 ^ 23 → ∀ b, 20 * 16448 ≤ 21 * b → b ≤ 16447 → ∀ D,
    MeetsWithSlack (m1p B 16448 (2 ^ 16448 - 21065) b) .sequential 64 D (2 ^ 20) 95 ((2⁻¹ : ℝ≥0∞) ^ 129) := by
  intro B _ h0 h1 b hsp hb D
  by_cases h : B ≤ 2 ^ 11
  · exact slack_uniform B 11 (2 ^ 10) 64 95 129 (95268 / 100000) (19 / 10) h0 (by norm_num) h h1 (by norm_num)
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) b hsp hb D
  · exact slack_uniform B 23 (2 ^ 11 + 1) 64 95 129 (95268 / 100000) (19 / 10) (by omega) (by norm_num) h1 h1
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      b hsp hb D

/-- `λ = 64`, `k = 96`, all of `2^8 ≤ B ≤ 2^23` (`c = 0.953161`, margin 9 bits per block). -/
theorem fam_64_96_129 : ∀ (B : ℕ) [NeZero B], 2 ^ 8 ≤ B → B ≤ 2 ^ 23 → ∀ b, 20 * 16448 ≤ 21 * b → b ≤ 16447 → ∀ D,
    MeetsWithSlack (m1p B 16448 (2 ^ 16448 - 21065) b) .sequential 64 D (2 ^ 20) 96 ((2⁻¹ : ℝ≥0∞) ^ 129) := by
  intro B _ h0 h1 b hsp hb D
  by_cases h : B ≤ 2 ^ 9
  · exact slack_uniform B 9 (2 ^ 8) 64 96 129 (953161 / 1000000) 9 h0 (by norm_num) h h1 (by norm_num)
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) b hsp hb D
  · exact slack_uniform B 23 (2 ^ 9 + 1) 64 96 129 (953161 / 1000000) 9 (by omega) (by norm_num) h1 h1
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      b hsp hb D

/-- `λ = 128`, `k = 104`, `2^9 ≤ B ≤ 2^23` (`c = 0.956685`, margin 3.7 bits per block). -/
theorem fam_128_129 : ∀ (B : ℕ) [NeZero B], 2 ^ 9 ≤ B → B ≤ 2 ^ 23 → ∀ b, 20 * 16448 ≤ 21 * b → b ≤ 16447 → ∀ D,
    MeetsWithSlack (m1p B 16448 (2 ^ 16448 - 21065) b) .sequential 128 D (2 ^ 20) 104 ((2⁻¹ : ℝ≥0∞) ^ 129) := by
  intro B _ h0 h1 b hsp hb D
  by_cases h : B ≤ 2 ^ 10
  · exact slack_uniform B 10 (2 ^ 9) 128 104 129 (956685 / 1000000) (37 / 10) h0 (by norm_num) h h1 (by norm_num)
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) b hsp hb D
  · exact slack_uniform B 23 (2 ^ 10 + 1) 128 104 129 (956685 / 1000000) (37 / 10) (by omega) (by norm_num) h1 h1
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      b hsp hb D

/-- `λ = 128`, `k = 105`, all of `2^8 ≤ B ≤ 2^23` (`c = 0.957089`, margin 10 bits per block). -/
theorem fam_128_105_129 : ∀ (B : ℕ) [NeZero B], 2 ^ 8 ≤ B → B ≤ 2 ^ 23 → ∀ b, 20 * 16448 ≤ 21 * b → b ≤ 16447 →
    ∀ D, MeetsWithSlack (m1p B 16448 (2 ^ 16448 - 21065) b) .sequential 128 D (2 ^ 20) 105 ((2⁻¹ : ℝ≥0∞) ^ 129) := by
  intro B _ h0 h1 b hsp hb D
  by_cases h : B ≤ 2 ^ 9
  · exact slack_uniform B 9 (2 ^ 8) 128 105 129 (957089 / 1000000) 10 h0 (by norm_num) h h1 (by norm_num)
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) b hsp hb D
  · exact slack_uniform B 23 (2 ^ 9 + 1) 128 105 129 (957089 / 1000000) 10 (by omega) (by norm_num) h1 h1
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      b hsp hb D

/-- `λ = 256`, `k = 127`, `2^10 ≤ B ≤ 2^23` (`c = 0.964388`, margin 2.2 bits per block). -/
theorem fam_256_129 : ∀ (B : ℕ) [NeZero B], 2 ^ 10 ≤ B → B ≤ 2 ^ 23 → ∀ b, 20 * 16448 ≤ 21 * b → b ≤ 16447 → ∀ D,
    MeetsWithSlack (m1p B 16448 (2 ^ 16448 - 21065) b) .sequential 256 D (2 ^ 20) 127 ((2⁻¹ : ℝ≥0∞) ^ 129) := by
  intro B _ h0 h1 b hsp hb D
  by_cases h : B ≤ 2 ^ 11
  · exact slack_uniform B 11 (2 ^ 10) 256 127 129 (964388 / 1000000) (22 / 10) h0 (by norm_num) h h1 (by norm_num)
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) b hsp hb D
  · exact slack_uniform B 23 (2 ^ 11 + 1) 256 127 129 (964388 / 1000000) (22 / 10) (by omega) (by norm_num) h1 h1
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      b hsp hb D

/-- `λ = 256`, `k = 128`, all of `2^8 ≤ B ≤ 2^23` (`c = 0.964661`, margin 6.5 bits per block). -/
theorem fam_256_128_129 : ∀ (B : ℕ) [NeZero B], 2 ^ 8 ≤ B → B ≤ 2 ^ 23 → ∀ b, 20 * 16448 ≤ 21 * b → b ≤ 16447 →
    ∀ D, MeetsWithSlack (m1p B 16448 (2 ^ 16448 - 21065) b) .sequential 256 D (2 ^ 20) 128 ((2⁻¹ : ℝ≥0∞) ^ 129) := by
  intro B _ h0 h1 b hsp hb D
  by_cases h : B ≤ 2 ^ 9
  · exact slack_uniform B 9 (2 ^ 8) 256 128 129 (964661 / 1000000) (65 / 10) h0 (by norm_num) h h1 (by norm_num)
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) b hsp hb D
  · exact slack_uniform B 23 (2 ^ 9 + 1) 256 128 129 (964661 / 1000000) (65 / 10) (by omega) (by norm_num) h1 h1
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      b hsp hb D

/-! ### Non-vacuity: raw storage of `⌊(S + λ·B)/w⌋` blocks -/

/-- Raw storage of `m` blocks, `m·w ≤ ⌊ρ|C|⌋ + λ·B`, breaks `MeetsWithSlack` at every `k ≤ kr` and every `ε`
once `(m/B)^kr > 1% + 2^-128`. -/
theorem not_slack_raw {M : IdealModel} (sch : Scheme M) (lam m kr : ℕ)
    (hS : m * sch.ℓ ≤ sch.stateBound Params.ρ + lam * sch.B) (hm : m ≤ sch.B)
    (hbig : (1 / 100 : ℝ) + (2⁻¹ : ℝ) ^ 128 < ((m : ℝ) / sch.B) ^ kr) (D Q k : ℕ) (hk : k ≤ kr) (ε : ℝ≥0∞) :
    ¬ MeetsWithSlack sch .sequential lam D Q k ε := by
  rintro ⟨hε, -, -, -, hsec⟩
  have h := (raw_seq_accepted sch hS hm D Q k (fun _ => false)).trans (hsec _ _ _)
  have hB0 : (0 : ℝ) < sch.B := by exact_mod_cast sch.B_pos
  have hmB : (m : ℝ) / sch.B ≤ 1 := by
    rw [div_le_one hB0]; exact_mod_cast hm
  have hL : (((m : ℕ) : ℝ≥0∞) / (sch.B : ℝ≥0∞)) ^ k = ENNReal.ofReal (((m : ℝ) / sch.B) ^ k) := by
    rw [← ENNReal.ofReal_natCast, ← ENNReal.ofReal_natCast sch.B, ← ENNReal.ofReal_div_of_pos hB0,
      ← ENNReal.ofReal_pow (by positivity)]
  have hR : ENNReal.ofReal Params.δ + Params.εMax = ENNReal.ofReal (1 / 100 + (2⁻¹ : ℝ) ^ 128) := by
    rw [ENNReal.ofReal_add (by norm_num) (by positivity), ENNReal.ofReal_pow (by norm_num),
      ENNReal.ofReal_inv_of_pos (by norm_num), ENNReal.ofReal_ofNat]
    simp only [Params.δ, Params.εMax, Params.secParam]
    norm_num
  have h2 := h.trans (add_le_add le_rfl hε)
  rw [hL, hR, ENNReal.ofReal_le_ofReal_iff (by positivity)] at h2
  have h3 : ((m : ℝ) / sch.B) ^ kr ≤ ((m : ℝ) / sch.B) ^ k := pow_le_pow_of_le_one (by positivity) hmB hk
  linarith

/-- At `2^19` blocks: raw storage of `m` blocks with `m·16448 ≤ 8169621180 + λ·2^19`. -/
theorem not_slack_b19 (lam m kr : ℕ) (hS : m * 16448 ≤ 8169621180 + lam * 2 ^ 19) (hm : m ≤ 2 ^ 19)
    (hbig : (1 / 100 : ℝ) + (2⁻¹ : ℝ) ^ 128 < ((m : ℝ) / 2 ^ 19) ^ kr) (p b D Q k : ℕ) (hk : k ≤ kr) (ε : ℝ≥0∞) :
    ¬ MeetsWithSlack (m1p (2 ^ 19) 16448 p b) .sequential lam D Q k ε := by
  refine not_slack_raw _ lam m kr ?_ hm ?_ D Q k hk ε
  · rw [stateBound_m1p_b19]; exact hS
  · have : ((m1p (2 ^ 19) 16448 p b).B : ℝ) = 2 ^ 19 := by norm_num [m1p]
    rw [this]; exact hbig

/-- **`λ = 0`: raw storage (496,693 blocks, fraction 0.94737) breaks every `k ≤ 85`.** -/
theorem raw_b19_0 (p b D Q k : ℕ) (hk : k ≤ 85) (ε : ℝ≥0∞) :
    ¬ MeetsWithSlack (m1p (2 ^ 19) 16448 p b) .sequential 0 D Q k ε :=
  not_slack_b19 0 496693 85 (by norm_num) (by norm_num) (by norm_num) p b D Q k hk ε

/-- **`λ = 64`: raw storage (498,733 blocks, fraction 0.95126) breaks every `k ≤ 92`.** -/
theorem raw_b19_64 (p b D Q k : ℕ) (hk : k ≤ 92) (ε : ℝ≥0∞) :
    ¬ MeetsWithSlack (m1p (2 ^ 19) 16448 p b) .sequential 64 D Q k ε :=
  not_slack_b19 64 498733 92 (by norm_num) (by norm_num) (by norm_num) p b D Q k hk ε

/-- **`λ = 128`: raw storage (500,773 blocks, fraction 0.95515) breaks every `k ≤ 100`.** -/
theorem raw_b19_128 (p b D Q k : ℕ) (hk : k ≤ 100) (ε : ℝ≥0∞) :
    ¬ MeetsWithSlack (m1p (2 ^ 19) 16448 p b) .sequential 128 D Q k ε :=
  not_slack_b19 128 500773 100 (by norm_num) (by norm_num) (by norm_num) p b D Q k hk ε

/-- **`λ = 256`: raw storage (504,854 blocks, fraction 0.96293) breaks every `k ≤ 121`.** -/
theorem raw_b19_256 (p b D Q k : ℕ) (hk : k ≤ 121) (ε : ℝ≥0∞) :
    ¬ MeetsWithSlack (m1p (2 ^ 19) 16448 p b) .sequential 256 D Q k ε :=
  not_slack_b19 256 504854 121 (by norm_num) (by norm_num) (by norm_num) p b D Q k hk ε

/-- Raw storage keeps exactly those blocks: one more block would exceed the raised state bound. -/
theorem raw_counts_max : 496694 * 16448 > 8169621180 + 0 * 2 ^ 19 ∧ 498734 * 16448 > 8169621180 + 64 * 2 ^ 19 ∧
    500774 * 16448 > 8169621180 + 128 * 2 ^ 19 ∧ 504855 * 16448 > 8169621180 + 256 * 2 ^ 19 := by
  norm_num

/-! ### The families at `ε = 2^-128` -/

/-- The `2^-129` audit bound gives the `2^-128` one. -/
theorem audit_eps128 {M : IdealModel} {sch : Scheme M} {lam D Q k : ℕ}
    (h : MeetsWithSlack sch .sequential lam D Q k ((2⁻¹ : ℝ≥0∞) ^ 129)) :
    AuditSecure sch .sequential (sch.stateBound Params.ρ + lam * sch.B) D Q k (ENNReal.ofReal Params.δ)
      Params.εMax :=
  (meetsWithSlack_mono h eps129_le le_rfl).2.2.2.2

theorem fam_64 : ∀ (B : ℕ) [NeZero B], 2 ^ 10 ≤ B → B ≤ 2 ^ 23 → ∀ b, 20 * 16448 ≤ 21 * b → b ≤ 16447 → ∀ D,
    MeetsWithSlack (m1p B 16448 (2 ^ 16448 - 21065) b) .sequential 64 D (2 ^ 20) 95 Params.εMax :=
  fun B _ h0 h1 b hsp hb D => meetsWithSlack_mono (fam_64_129 B h0 h1 b hsp hb D) eps129_le le_rfl

theorem fam_64_96 : ∀ (B : ℕ) [NeZero B], 2 ^ 8 ≤ B → B ≤ 2 ^ 23 → ∀ b, 20 * 16448 ≤ 21 * b → b ≤ 16447 → ∀ D,
    MeetsWithSlack (m1p B 16448 (2 ^ 16448 - 21065) b) .sequential 64 D (2 ^ 20) 96 Params.εMax :=
  fun B _ h0 h1 b hsp hb D => meetsWithSlack_mono (fam_64_96_129 B h0 h1 b hsp hb D) eps129_le le_rfl

theorem fam_128 : ∀ (B : ℕ) [NeZero B], 2 ^ 9 ≤ B → B ≤ 2 ^ 23 → ∀ b, 20 * 16448 ≤ 21 * b → b ≤ 16447 → ∀ D,
    MeetsWithSlack (m1p B 16448 (2 ^ 16448 - 21065) b) .sequential 128 D (2 ^ 20) 104 Params.εMax :=
  fun B _ h0 h1 b hsp hb D => meetsWithSlack_mono (fam_128_129 B h0 h1 b hsp hb D) eps129_le le_rfl

theorem fam_128_105 : ∀ (B : ℕ) [NeZero B], 2 ^ 8 ≤ B → B ≤ 2 ^ 23 → ∀ b, 20 * 16448 ≤ 21 * b → b ≤ 16447 → ∀ D,
    MeetsWithSlack (m1p B 16448 (2 ^ 16448 - 21065) b) .sequential 128 D (2 ^ 20) 105 Params.εMax :=
  fun B _ h0 h1 b hsp hb D => meetsWithSlack_mono (fam_128_105_129 B h0 h1 b hsp hb D) eps129_le le_rfl

theorem fam_256 : ∀ (B : ℕ) [NeZero B], 2 ^ 10 ≤ B → B ≤ 2 ^ 23 → ∀ b, 20 * 16448 ≤ 21 * b → b ≤ 16447 → ∀ D,
    MeetsWithSlack (m1p B 16448 (2 ^ 16448 - 21065) b) .sequential 256 D (2 ^ 20) 127 Params.εMax :=
  fun B _ h0 h1 b hsp hb D => meetsWithSlack_mono (fam_256_129 B h0 h1 b hsp hb D) eps129_le le_rfl

theorem fam_256_128 : ∀ (B : ℕ) [NeZero B], 2 ^ 8 ≤ B → B ≤ 2 ^ 23 → ∀ b, 20 * 16448 ≤ 21 * b → b ≤ 16447 → ∀ D,
    MeetsWithSlack (m1p B 16448 (2 ^ 16448 - 21065) b) .sequential 256 D (2 ^ 20) 128 Params.εMax :=
  fun B _ h0 h1 b hsp hb D => meetsWithSlack_mono (fam_256_128_129 B h0 h1 b hsp hb D) eps129_le le_rfl

/-- `λ = 0`, `k = 88`, `ε = 2^-129` on the whole family, as a `Meets` statement. -/
theorem fam_meets_129 : ∀ (B : ℕ) [NeZero B], 2 ^ 8 ≤ B → B ≤ 2 ^ 23 → ∀ b, 20 * 16448 ≤ 21 * b → b ≤ 16447 →
    ∀ D, Meets (m1p B 16448 (2 ^ 16448 - 21065) b) .sequential D (2 ^ 20) 88 ((2⁻¹ : ℝ≥0∞) ^ 129) :=
  fun B _ h0 h1 b hsp hb D => (meetsWithSlack_zero _ _ _ _ _ _).1 (fam_0_129 B h0 h1 b hsp hb D)

/-- **Satisfiability of the uniform certificates' numeric hypotheses** is discharged inside each `fam_*` by
`norm_num`; the width hypotheses are `PousP2p.widths` (`b = 16447`, `16432`), and `2^19`, `2^23` lie in every
family range. The rows are not vacuous: `raw_b19_*` break each at a smaller `k`. -/
theorem b19_in_ranges : 2 ^ 10 ≤ 2 ^ 19 ∧ 2 ^ 19 ≤ 2 ^ 23 ∧ 2 ^ 10 ≤ 2 ^ 23 := by norm_num

end PousMargin
