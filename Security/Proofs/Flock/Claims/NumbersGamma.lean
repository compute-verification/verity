import Proofs.Flock.Claims.Numbers
import Proofs.Flock.Claims.NumbersPads
import Proofs.Flock.Soundness.Accounting.PaddedNumbers

/-!
# Flock's `γ` at the deployed parameters

`γ = 16·(√(2·k(x)·C) + k(x))²` (`Claims.Flock.γ`), with `k(x)` the verifier's rounds and `C` the weight. Here:

* `N₀ = 2^{d₀}`, level 0's codeword length, has `10 ≤ d₀ ≤ 23` at every `fast100` schedule (`fast100_logLen₀`), and
  `--zk` keeps it (`m1Schedule_logLen₀`), so `k = ⌈4N₀/e⌉` is `kVal d₀` (`kRep_eq`) and `Numbers`' bounds on `κ` apply;
* the weight at a schedule and pads' domain `N` is `C = 2·Σ_ℓ 2^{d_ℓ}·Q_ℓ + 2N² + 2 + 4N₀/e` (`weightOf`; `N = 0`
  without `--zk`, `N = padDomain`, `2^12` or `2^13`, with it), at most `2^33` at every `fast100` schedule and every
  `--zk` schedule (`weightOf_le_fast100`, `weightOf_le_m1`);
* so `γ ≤ 2^48` at up to `768` rounds (`γ_le_48`), and `γ ≤ 2^105` at up to `2^20` rounds and any weight up to `2^64`
  (`γ_le_105`), where `λ`'s hash side, `255.5 − ½ log₂ γ ≥ 203`, stays above its table side: `λ ≥ 201.98` there
  (`lam_ge_of_rounds`).

At `201` rounds `C` runs from `2^{18.98}` (`m = 22`) to `2^{31.97}` (`m = 35`) without `--zk`, and from `2^{25.20}`
(`m = 25`, `N = 2^12`) to `2^{32.02}` (`m = 35`, `N = 2^13`) with it, so `γ` from `2^{31.67}` to `2^{44.67}`; the spec's
`2^{38.5}` is `--zk` at `m = 26, 27` with `N = 2^12` (`2^{38.00}`, `2^{38.25}`). `λ` doesn't move: `½ log₂ γ ≤ 23.4`.
-/

namespace Claims.Flock.Numbers

open FlockSoundness.Accounting

/-! ## `N₀` -/

/-- Level 0's codeword length at every `fast100` schedule is `2^10` to `2^23`. -/
theorem fast100_logLen₀ {m : ℕ} {sch : Schedule} (h : fast100 m = some sch) :
    10 ≤ (sch.levels.headD default).logLen ∧ (sch.levels.headD default).logLen ≤ 23 := by
  have hm : 22 ≤ m ∧ m ≤ 35 := by
    by_contra hc
    have : fast100 m = none := by unfold fast100; split <;> simp_all
    simp [this] at h
  obtain ⟨h1, h2⟩ := hm
  interval_cases m <;> (simp only [fast100, Option.some.injEq] at h; subst h; decide)

/-- `--zk` keeps level 0's codeword length. -/
theorem m1Schedule_logLen₀ {m : ℕ} (h1 : 25 ≤ m) (h2 : m ≤ 35) :
    10 ≤ ((m1Schedule m).levels.headD default).logLen ∧ ((m1Schedule m).levels.headD default).logLen ≤ 23 := by
  interval_cases m <;> decide

/-- `k = ⌈4N₀/e⌉` at a `fast100` schedule is `kVal d₀`. -/
theorem k_fast100 {m : ℕ} {sch : Schedule} (h : fast100 m = some sch) :
    kRep (sch.levels.headD default).logLen = kVal (sch.levels.headD default).logLen := by
  have hm : 22 ≤ m ∧ m ≤ 35 := by
    by_contra hc
    have : fast100 m = none := by unfold fast100; split <;> simp_all
    simp [this] at h
  obtain ⟨h1, h2⟩ := hm
  apply kRep_eq
  interval_cases m <;> (simp only [fast100, Option.some.injEq] at h; subst h; decide)

/-! ## The weight -/

/-- The weight's natural part: `2·Σ_ℓ 2^{d_ℓ}·Q_ℓ + 2N² + 2`. -/
def weightNat (sch : Schedule) (N : ℕ) : ℕ :=
  2 * (sch.levels.map fun l => 2 ^ l.logLen * l.queries).sum + 2 * N ^ 2 + 2

/-- **The weight `C` of `γ`** at a schedule and pads' domain `N`: `2·Σ_ℓ 2^{d_ℓ}·Q_ℓ + 2N² + 2 + 4N₀/e`. -/
noncomputable def weightOf (sch : Schedule) (N : ℕ) : ℝ :=
  weightNat sch N + 4 * 2 ^ (sch.levels.headD default).logLen / Real.exp 1

theorem weightOf_nonneg (sch : Schedule) (N : ℕ) : 0 ≤ weightOf sch N := by
  unfold weightOf
  have := Real.exp_pos 1
  positivity

theorem weightNat_mono (sch : Schedule) {N N' : ℕ} (h : N ≤ N') : weightNat sch N ≤ weightNat sch N' := by
  unfold weightNat
  have := Nat.pow_le_pow_left h 2
  omega

/-- `4N₀/e ≤ 2^25` at `N₀ ≤ 2^23`. -/
theorem kTerm_le {d : ℕ} (hd : d ≤ 23) : 4 * (2 : ℝ) ^ d / Real.exp 1 ≤ 2 ^ 25 := by
  have he0 : 0 < Real.exp 1 := Real.exp_pos 1
  have hlo := Real.exp_one_gt_d9
  have h2 : (2 : ℝ) ^ d ≤ 2 ^ 23 := pow_le_pow_right₀ (by norm_num) hd
  rw [div_le_iff₀ he0]
  nlinarith

theorem weightOf_le_of {sch : Schedule} {N : ℕ} (hd : (sch.levels.headD default).logLen ≤ 23)
    (hN : weightNat sch N ≤ 2 ^ 33 - 2 ^ 25) : weightOf sch N ≤ 2 ^ 33 := by
  unfold weightOf
  have h1 : (weightNat sch N : ℝ) ≤ 2 ^ 33 - 2 ^ 25 := by
    have : ((2 ^ 33 - 2 ^ 25 : ℕ) : ℝ) = 2 ^ 33 - 2 ^ 25 := by norm_num
    rw [← this]
    exact_mod_cast hN
  have h2 := kTerm_le hd
  linarith

/-- **`C ≤ 2^33` without `--zk`**, and with pads' domains up to `2^13`, at every `fast100` schedule. -/
theorem weightOf_le_fast100 {m : ℕ} {sch : Schedule} (h : fast100 m = some sch) {N : ℕ} (hN : N ≤ 2 ^ 13) :
    weightOf sch N ≤ 2 ^ 33 := by
  refine weightOf_le_of (fast100_logLen₀ h).2 ((weightNat_mono sch hN).trans ?_)
  have hm : 22 ≤ m ∧ m ≤ 35 := by
    by_contra hc
    have : fast100 m = none := by unfold fast100; split <;> simp_all
    simp [this] at h
  obtain ⟨h1, h2⟩ := hm
  interval_cases m <;> (simp only [fast100, Option.some.injEq] at h; subst h; decide)

/-- **`C ≤ 2^33` with `--zk`**, at every `--zk` schedule and pads' domain up to `2^13`. -/
theorem weightOf_le_m1 {m : ℕ} (h1 : 25 ≤ m) (h2 : m ≤ 35) {N : ℕ} (hN : N ≤ 2 ^ 13) :
    weightOf (m1Schedule m) N ≤ 2 ^ 33 := by
  refine weightOf_le_of (m1Schedule_logLen₀ h1 h2).2 ((weightNat_mono _ hN).trans ?_)
  interval_cases m <;> decide

/-- `C ≤ 2^33` at a `--zk` statement's own pads' domain, in scope. -/
theorem weightOf_le_m1_pads {m kLog k0 : ℕ} (h1 : 25 ≤ m) (h2 : m ≤ 35) (hk : kLog ≤ 27) (h0 : k0 ≤ 6) :
    weightOf (m1Schedule m) (padDomain (_root_.Flock.Zk.PadLayout.new m kLog k0)) ≤ 2 ^ 33 :=
  weightOf_le_m1 h1 h2 (padDomain_le h2 hk h0)

/-! ## `γ` -/

section Deploy

variable {Coin Col T : Type}

/-- `γ` is monotone in the rounds and the weight. -/
theorem γ_le_of (D : Deploy Coin Col T) (x : Msg) {r C : ℝ} (hr : (D.V.k x : ℝ) ≤ r) (hC0 : 0 ≤ D.weight x)
    (hC : D.weight x ≤ C) : γ D x ≤ 16 * (Real.sqrt (2 * r * C) + r) ^ 2 := by
  unfold γ
  have hk0 : (0 : ℝ) ≤ D.V.k x := Nat.cast_nonneg _
  have hs : Real.sqrt (2 * D.V.k x * D.weight x) ≤ Real.sqrt (2 * r * C) := by
    apply Real.sqrt_le_sqrt
    have := mul_le_mul hr hC hC0 (hk0.trans hr)
    nlinarith
  have h0 : 0 ≤ Real.sqrt (2 * D.V.k x * D.weight x) + D.V.k x := by positivity
  nlinarith

/-- **`γ ≤ 2^48`** at up to `768` rounds and a weight up to `2^33`. -/
theorem γ_le_48 (D : Deploy Coin Col T) (x : Msg) (hr : D.V.k x ≤ 768) (hC0 : 0 ≤ D.weight x)
    (hC : D.weight x ≤ 2 ^ 33) : γ D x ≤ 2 ^ (48 : ℝ) := by
  refine (γ_le_of D x (r := 768) (by exact_mod_cast hr) hC0 hC).trans ?_
  have hs : Real.sqrt (2 * 768 * 2 ^ 33) ≤ 3700000 := by
    rw [Real.sqrt_le_left (by norm_num)]
    norm_num
  rw [show (48 : ℝ) = ((48 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
  have h0 : 0 ≤ Real.sqrt (2 * 768 * 2 ^ 33) := Real.sqrt_nonneg _
  nlinarith

/-- **`γ ≤ 2^105`** at up to `2^20` rounds and a weight up to `2^64`. -/
theorem γ_le_105 (D : Deploy Coin Col T) (x : Msg) (hr : D.V.k x ≤ 2 ^ 20) (hC0 : 0 ≤ D.weight x)
    (hC : D.weight x ≤ 2 ^ 64) : γ D x ≤ 2 ^ (105 : ℝ) := by
  refine (γ_le_of D x (r := 2 ^ 20) (by exact_mod_cast hr) hC0 hC).trans ?_
  have hs : Real.sqrt (2 * 2 ^ 20 * 2 ^ 64) ≤ 2 ^ 43 := by
    rw [Real.sqrt_le_left (by norm_num)]
    norm_num
  rw [show (105 : ℝ) = ((105 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
  have h0 : 0 ≤ Real.sqrt (2 * 2 ^ 20 * 2 ^ 64) := Real.sqrt_nonneg _
  nlinarith

/-- **`λ ≥ 201.98`** at the deployed numbers, up to `2^20` rounds and a weight up to `2^64`. -/
theorem lam_ge_of_rounds (D : Deploy Coin Col T) (x : Msg) (hε0 : 0 ≤ D.εtable x)
    (hε : D.εtable x ≤ 2 ^ (-(205 : ℝ))) (hw : 210 ≤ D.waits x) (hK : D.cap x = cap) (hN : 2 ^ 10 ≤ D.N₀ x)
    (hN' : D.N₀ x ≤ 2 ^ 23) (hr : D.V.k x ≤ 2 ^ 20) (hC0 : 0 ≤ D.weight x) (hC : D.weight x ≤ 2 ^ 64) :
    201.98 ≤ lamOf (κ D x) (γ D x) :=
  lam_ge D x hε0 hε hw hK hN hN' (γ_le_105 D x hr hC0 hC)

end Deploy

end Claims.Flock.Numbers
