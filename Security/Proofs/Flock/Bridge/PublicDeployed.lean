import Proofs.Flock.Bridge.Public

/-!
# `deltaPub` is below 1 at the deployment's parameters

`SampledKSLivePub` bounds a cheating prover's success by `Bridge.deltaPub`, which says something only below 1. At the
parameters ONE-19 deploys, `ρ = 1/2`, `k = ⌈2^24/e⌉` (`kDep`) and `M = R_w = 2^192`, it is:

* `rateM1_kDep`: `SampledKSLivePub`'s rate hypothesis holds, `2^23/(e·kDep) ≤ 1/2`;
* `ksClosedM1_le`: the knowledge term is at most `2^-155` at budgets `q_F, q_S ≤ 2^82`;
* `link_le`: the link factor is at most `2^-150` at `t' ≤ 2^80`;
* `deltaPub_le_deployed`: so `deltaPub` is the draw's miss plus at most `2^-100`, for `W ≤ 2^40` committed wires;
* `miss_le_pow`, `miss_le_exp`: the miss `C(n−K₀, k_d)/C(n, k_d)` is at most `(1 − K₀/n)^{k_d} ≤ e^{−K₀k_d/n}`;
* `Flock.Guarantees.DeltaPubDeployed`: at `K₀ ≤ n ≤ K₀·k_d` (a draw of `k_d` units expects at least one of `K₀` bad
  ones), the miss is at most `1/e`, so `deltaPub < 1` at `SampledKSLivePub`'s budgets `q_F = 2(t' + v)`, `q_S = t' + v`
  for `t', v ≤ 2^80`.
-/

namespace Flock.Bridge

open FlockSoundness.Discharge.ZkLink

/-- **The deployment's `k`**: `⌈2^24/e⌉`, the least at which `rateM1 k ≤ 1/2`. -/
noncomputable def kDep : ℕ := ⌈(2 : ℝ) ^ 24 / Real.exp 1⌉₊

theorem kDep_ge : (2 : ℝ) ^ 24 / Real.exp 1 ≤ kDep := Nat.le_ceil _

theorem kDep_le : (kDep : ℝ) ≤ 2 ^ 23 := by
  have he : (2.7 : ℝ) < Real.exp 1 := by linarith [Real.exp_one_gt_d9]
  have h1 : (2 : ℝ) ^ 24 / Real.exp 1 ≤ 2 ^ 23 - 1 := by
    rw [div_le_iff₀ (Real.exp_pos 1)]
    nlinarith
  have := Nat.ceil_lt_add_one (show (0 : ℝ) ≤ 2 ^ 24 / Real.exp 1 by positivity)
  unfold kDep
  linarith

theorem one_le_kDep : 1 ≤ kDep := by
  have h := kDep_ge
  have he : Real.exp 1 < 2.7182818286 := Real.exp_one_lt_d9
  have : (1 : ℝ) < 2 ^ 24 / Real.exp 1 := by
    rw [lt_div_iff₀ (Real.exp_pos 1)]
    linarith
  exact_mod_cast (this.trans_le h).le

/-- **The rate hypothesis at `kDep`**: `2^23/(e·kDep) ≤ 1/2`. -/
theorem rateM1_kDep : rateM1 kDep ≤ 1 / 2 := by
  unfold rateM1
  have he := Real.exp_pos 1
  have hk := kDep_ge
  rw [div_le_iff₀ he] at hk
  rw [div_le_iff₀ (by nlinarith)]
  nlinarith

set_option exponentiation.threshold 1024 in
theorem sqrt_qsq_le {q : ℝ} (h0 : 0 ≤ q) : Real.sqrt (q ^ 2 / 2 ^ (513 : ℝ)) ≤ q / 2 ^ 256 := by
  rw [Real.sqrt_le_left (by positivity)]
  rw [div_pow, show ((2 : ℝ) ^ 256) ^ 2 = 2 ^ (512 : ℝ) by
    rw [← pow_mul, show (512 : ℝ) = ((512 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]]
  exact div_le_div_of_nonneg_left (sq_nonneg q) (by positivity)
    (Real.rpow_le_rpow_of_exponent_le (by norm_num) (by norm_num))

set_option exponentiation.threshold 1024 in
/-- **The knowledge term at budgets up to `2^82`** is at most `2^-155`, for `k ≤ 2^23`. -/
theorem ksClosedM1_le {k : ℕ} (hk : (k : ℝ) ≤ 2 ^ 23) {qF qS : ℝ} (hF0 : 0 ≤ qF) (hF : qF ≤ 2 ^ 82)
    (hS0 : 0 ≤ qS) (hS : qS ≤ 2 ^ 82) : ksClosedM1 k qF qS ≤ 2 ^ (-(155 : ℝ)) := by
  unfold ksClosedM1
  have hx' : Real.sqrt (qF ^ 2 / 2 ^ (513 : ℝ)) ≤ 2 ^ 82 / 2 ^ 256 := (sqrt_qsq_le hF0).trans (by gcongr)
  have h513 : (2 : ℝ) ^ (513 : ℝ) = 2 ^ 513 := by norm_num
  have hS2 : qS ^ 2 / 2 ^ (513 : ℝ) ≤ (2 ^ 82) ^ 2 / 2 ^ 513 := by
    rw [h513]; gcongr
  have hkx : Real.sqrt (k * (qF ^ 2 / 2 ^ (513 : ℝ))) ≤ 2 ^ 12 * (2 ^ 82 / 2 ^ 256) := by
    rw [Real.sqrt_mul (Nat.cast_nonneg k)]
    have hsk : Real.sqrt k ≤ 2 ^ 12 := by
      rw [Real.sqrt_le_left (by positivity)]
      linarith
    exact mul_le_mul hsk hx' (Real.sqrt_nonneg _) (by positivity)
  have h203 : (2 : ℝ) ^ (-(203 : ℝ)) = 1 / 2 ^ 203 := by
    rw [Real.rpow_neg (by norm_num)]; norm_num
  have h155 : (2 : ℝ) ^ (-(155 : ℝ)) = 1 / 2 ^ 155 := by
    rw [Real.rpow_neg (by norm_num)]; norm_num
  rw [h203, h155]
  have := mul_le_mul_of_nonneg_left hx' (show (0 : ℝ) ≤ 2 * 25638 + 2 ^ 14 + 2 by norm_num)
  refine le_trans (?_ : _ ≤ (1 : ℝ) / 2 ^ 203 + (2 * 25638 + 2 ^ 14 + 2) * (2 ^ 82 / 2 ^ 256) +
      (2 ^ 82) ^ 2 / 2 ^ 513 + 2 ^ 12 * (2 ^ 82 / 2 ^ 256)) ?_
  · linarith
  · norm_num

/-- **The link factor at `t' ≤ 2^80` and `M = R_w = 2^192`** is at most `2^-150`, for `k ≤ 2^23`. -/
theorem link_le {k : ℕ} (hk : (k : ℝ) ≤ 2 ^ 23) {t' : ℝ} (ht : t' ≤ 2 ^ 80) :
    2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * ((2 ^ 192 : ℕ) : ℝ)) +
      k / (Real.exp 1 * ((2 ^ 192 : ℕ) : ℝ)) ≤ 2 ^ (-(150 : ℝ)) := by
  have he : (1 : ℝ) ≤ Real.exp 1 := by linarith [Real.exp_one_gt_d9]
  have h256 : (2 : ℝ) ^ (256 : ℝ) = 2 ^ 256 := by norm_num
  have h150 : (2 : ℝ) ^ (-(150 : ℝ)) = 1 / 2 ^ 150 := by
    rw [Real.rpow_neg (by norm_num)]; norm_num
  rw [h256, h150]
  simp only [Nat.cast_pow, Nat.cast_ofNat]
  have hk0 : (0 : ℝ) ≤ k := Nat.cast_nonneg k
  have hM : (2 : ℝ) ^ 192 ≤ Real.exp 1 * 2 ^ 192 := le_mul_of_one_le_left (by positivity) he
  have h1 : 2 * (t' * (1 + k)) / 2 ^ 256 ≤ 2 * (2 ^ 80 * (1 + 2 ^ 23)) / 2 ^ 256 := by gcongr
  have h2 : 1 / (Real.exp 1 * 2 ^ 192) ≤ 1 / 2 ^ 192 := one_div_le_one_div_of_le (by positivity) hM
  have h3 : (k : ℝ) / (Real.exp 1 * 2 ^ 192) ≤ 2 ^ 23 / 2 ^ 192 :=
    div_le_div₀ (by positivity) hk (by positivity) hM
  refine le_trans (?_ : _ ≤ (2 : ℝ) * (2 ^ 80 * (1 + 2 ^ 23)) / 2 ^ 256 + 1 / 2 ^ 192 + 2 ^ 23 / 2 ^ 192) ?_
  · linarith
  · norm_num

/-- **`deltaPub` at the deployment's parameters is the draw's miss plus at most `2^-100`**: `ρ = 1/2`, `k = kDep`,
`M = R_w = 2^192`, budgets `q_F, q_S ≤ 2^82`, `t' ≤ 2^80` and at most `2^40` committed wires. -/
theorem deltaPub_le_deployed (n kd K₀ W : ℕ) (hW : W ≤ 2 ^ 40) {qF qS t' : ℝ} (hF0 : 0 ≤ qF) (hF : qF ≤ 2 ^ 82)
    (hS0 : 0 ≤ qS) (hS : qS ≤ 2 ^ 82) (ht0 : 0 ≤ t') (ht : t' ≤ 2 ^ 80) :
    deltaPub n kd K₀ kDep qF qS t' (2 ^ 192) (2 ^ 192) W (1 / 2) ≤
      ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ) + 2 ^ (-(100 : ℝ)) := by
  unfold deltaPub
  have hks := ksClosedM1_le kDep_le hF0 hF hS0 hS
  have hl := link_le kDep_le ht
  have hW' : (W : ℝ) ≤ 2 ^ 40 := by exact_mod_cast hW
  set B := 2 * (t' * (1 + (kDep : ℝ))) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * ((2 ^ 192 : ℕ) : ℝ)) +
      (kDep : ℝ) / (Real.exp 1 * ((2 ^ 192 : ℕ) : ℝ)) with hB
  have hl0 : 0 ≤ B := by rw [hB]; positivity
  have hWl := mul_le_mul hW' hl hl0 (by positivity)
  have e : ∀ r : ℝ, (2 : ℝ) ^ (-r) = 1 / (2 : ℝ) ^ r := fun r => by rw [Real.rpow_neg (by norm_num)]; ring
  have e155 : (2 : ℝ) ^ (155 : ℝ) = 2 ^ (155 : ℕ) := by norm_num
  have e150 : (2 : ℝ) ^ (150 : ℝ) = 2 ^ (150 : ℕ) := by norm_num
  have e100 : (2 : ℝ) ^ (100 : ℝ) = 2 ^ (100 : ℕ) := by norm_num
  rw [e, e155] at hks
  rw [e, e150] at hWl
  rw [e, e100]
  have hρ : (1 : ℝ) - 1 / 2 = 1 / 2 := by norm_num
  rw [hρ, div_div_eq_mul_div, div_div_eq_mul_div]
  have h1 : ksClosedM1 kDep qF qS * 2 / 1 ≤ 1 / 2 ^ 155 * 2 := by linarith
  have h2 : (W : ℝ) * B * 2 / 1 ≤ 2 ^ 40 * (1 / 2 ^ 150) * 2 := by linarith
  have h3 : (1 : ℝ) / 2 ^ 155 * 2 + 2 ^ 40 * (1 / 2 ^ 150) * 2 ≤ 1 / 2 ^ 100 := by norm_num
  linarith

theorem descFactorial_mul_pow_le {a n : ℕ} (h : a ≤ n) (k : ℕ) :
    a.descFactorial k * n ^ k ≤ a ^ k * n.descFactorial k := by
  induction k with
  | zero => simp
  | succ k ih =>
    have step : (a - k) * n ≤ a * (n - k) := by
      rcases le_or_gt k a with hk | hk
      · obtain ⟨x, rfl⟩ := Nat.exists_eq_add_of_le hk
        obtain ⟨y, rfl⟩ := Nat.exists_eq_add_of_le h
        rw [show k + x - k = x by omega, show k + x + y - k = x + y by omega]
        nlinarith
      · rw [show a - k = 0 by omega, zero_mul]
        exact Nat.zero_le _
    rw [Nat.descFactorial_succ, Nat.descFactorial_succ, pow_succ, pow_succ]
    calc (a - k) * a.descFactorial k * (n ^ k * n) = ((a - k) * n) * (a.descFactorial k * n ^ k) := by ring
      _ ≤ (a * (n - k)) * (a ^ k * n.descFactorial k) := Nat.mul_le_mul step ih
      _ = a ^ k * a * ((n - k) * n.descFactorial k) := by ring

/-- **The draw's miss is at most `(1 − K/n)^{k_d}`**: `C(n − K, k_d)/C(n, k_d)`, the chance that `k_d` units drawn
without replacement from `n` miss all of `K`. -/
theorem miss_le_pow {n K : ℕ} (hK : K ≤ n) (kd : ℕ) :
    ((n - K).choose kd : ℝ) / (n.choose kd : ℝ) ≤ (1 - (K : ℝ) / n) ^ kd := by
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · obtain rfl : K = 0 := by omega
    simp only [Nat.sub_zero, Nat.cast_zero, div_zero, sub_zero, one_pow]
    exact div_self_le_one _
  rcases Nat.eq_zero_or_pos (n.choose kd) with h0 | hc
  · rw [h0, Nat.cast_zero, div_zero]
    have : (K : ℝ) / n ≤ 1 := div_le_one_of_le₀ (by exact_mod_cast hK) (Nat.cast_nonneg n)
    exact pow_nonneg (by linarith) kd
  have hnR : (0 : ℝ) < n := by exact_mod_cast hn
  have hrw : 1 - (K : ℝ) / n = ((n - K : ℕ) : ℝ) / n := by
    rw [Nat.cast_sub hK]; field_simp
  rw [hrw, div_pow, div_le_div_iff₀ (by exact_mod_cast hc) (by positivity)]
  have hd := descFactorial_mul_pow_le (Nat.sub_le n K) kd
  rw [Nat.descFactorial_eq_factorial_mul_choose, Nat.descFactorial_eq_factorial_mul_choose] at hd
  have hf : (0 : ℝ) < kd.factorial := by exact_mod_cast Nat.factorial_pos kd
  have hd' : ((kd.factorial * (n - K).choose kd * n ^ kd : ℕ) : ℝ) ≤
      (((n - K) ^ kd * (kd.factorial * n.choose kd) : ℕ) : ℝ) := by exact_mod_cast hd
  push_cast at hd' ⊢
  nlinarith

/-- **The draw's miss is at most `e^{−K k_d/n}`.** -/
theorem miss_le_exp {n K : ℕ} (hK : K ≤ n) (kd : ℕ) :
    ((n - K).choose kd : ℝ) / (n.choose kd : ℝ) ≤ Real.exp (-((K : ℝ) * kd / n)) := by
  refine (miss_le_pow hK kd).trans ?_
  have h1 : 1 - (K : ℝ) / n ≤ Real.exp (-((K : ℝ) / n)) := by
    linarith [Real.add_one_le_exp (-((K : ℝ) / n))]
  have h0 : 0 ≤ 1 - (K : ℝ) / n := by
    have : (K : ℝ) / n ≤ 1 := div_le_one_of_le₀ (by exact_mod_cast hK) (Nat.cast_nonneg n)
    linarith
  calc (1 - (K : ℝ) / n) ^ kd ≤ Real.exp (-((K : ℝ) / n)) ^ kd := pow_le_pow_left₀ h0 h1 kd
    _ = Real.exp (-((K : ℝ) * kd / n)) := by
      rw [← Real.exp_nat_mul]; congr 1; ring

/-- **The miss at `0 < n ≤ K·k_d`** is at most `1/2`: `e^{−K k_d/n} ≤ e^{−1} < 1/2`. -/
theorem miss_le_half {n K kd : ℕ} (hn0 : 0 < n) (hK : K ≤ n) (hn : n ≤ K * kd) :
    ((n - K).choose kd : ℝ) / (n.choose kd : ℝ) ≤ 1 / 2 := by
  refine (miss_le_exp hK kd).trans ?_
  have hnR : (0 : ℝ) < n := by exact_mod_cast hn0
  have h1 : (1 : ℝ) ≤ (K : ℝ) * kd / n := by
    rw [le_div_iff₀ hnR, one_mul]; exact_mod_cast hn
  have he : Real.exp (-((K : ℝ) * kd / n)) ≤ Real.exp (-1) := Real.exp_le_exp.2 (by linarith)
  have h2 : Real.exp (-1) < 1 / 2 := by
    rw [Real.exp_neg, one_div]
    exact inv_strictAnti₀ (by norm_num) (by linarith [Real.exp_one_gt_d9])
  linarith

end Flock.Bridge

namespace Flock.Guarantees

/-- **`SampledKSLivePub`'s error is below 1 at the deployment's parameters.** At `k = kDep = ⌈2^24/e⌉`, `ρ = 1/2` and
`M = R_w = 2^192`, its rate hypothesis holds (`1 ≤ k`, `rateM1 k ≤ ρ`), and `deltaPub < 1` at its budgets
`q_F = 2(t' + v)` and `q_S = t' + v` for a prover's per-run time `t' ≤ 2^80` and one opening's verification `v ≤ 2^80`, at
most `2^40` committed wires (`W`, `Bridge.ioSum`) and a draw of `k_d` of `n` units with `K₀ ≤ n ≤ K₀·k_d`. -/
def DeltaPubDeployed : Prop :=
  1 ≤ Bridge.kDep ∧ FlockSoundness.Discharge.ZkLink.rateM1 Bridge.kDep ≤ 1 / 2 ∧
    ∀ (n kd K₀ W : ℕ) (t' v : ℝ), 0 < n → K₀ ≤ n → n ≤ K₀ * kd → W ≤ 2 ^ 40 → 0 ≤ t' → t' ≤ 2 ^ 80 →
      0 ≤ v → v ≤ 2 ^ 80 →
      Bridge.deltaPub n kd K₀ Bridge.kDep (2 * (t' + v)) (t' + v) t' (2 ^ 192) (2 ^ 192) W (1 / 2) < 1

end Flock.Guarantees

/-- **`SampledKSLivePub`'s error is below 1 at the deployment's parameters** (`Flock.Guarantees.DeltaPubDeployed`). -/
theorem Flock.SecurityProofs.DeltaPubDeployed : Flock.Guarantees.DeltaPubDeployed := by
  refine ⟨Flock.Bridge.one_le_kDep, Flock.Bridge.rateM1_kDep, ?_⟩
  intro n kd K₀ W t' v hn0 hK hn hW ht0 ht hv0 hv
  have hF : 2 * (t' + v) ≤ 2 ^ 82 := by linarith
  have hS : t' + v ≤ 2 ^ 82 := by linarith
  refine (Flock.Bridge.deltaPub_le_deployed n kd K₀ W hW (by linarith) hF (by linarith) hS ht0 ht).trans_lt ?_
  have h := Flock.Bridge.miss_le_half hn0 hK hn
  have : (2 : ℝ) ^ (-(100 : ℝ)) < 1 / 2 := by
    rw [Real.rpow_neg (by norm_num), show (100 : ℝ) = ((100 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
    norm_num
  linarith
