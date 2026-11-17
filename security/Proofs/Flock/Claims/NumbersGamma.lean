module

public import Proofs.Flock.Claims.Numbers
public import Proofs.Flock.Claims.NumbersPads
public import Proofs.Flock.Soundness.Accounting.PaddedNumbers

/-! Flock's `γ` at the deployed parameters. -/

@[expose] public section

namespace Claims.Flock.Numbers

open FlockSoundness.Accounting

variable {wideLeaf : Bool}

theorem fast100_logLen₀ {m : ℕ} {sch : Schedule} (h : fast100 m = some sch) :
    10 ≤ (sch.levels.headD default).logLen ∧ (sch.levels.headD default).logLen ≤ 23 := by
  have hm : 22 ≤ m ∧ m ≤ 35 := by
    by_contra hc
    have : fast100 m = none := by unfold fast100; split <;> simp_all
    simp [this] at h
  obtain ⟨h1, h2⟩ := hm
  interval_cases m <;> (simp only [fast100, Option.some.injEq] at h; subst h; decide)

theorem m1Schedule_logLen₀ {m : ℕ} (h1 : 25 ≤ m) (h2 : m ≤ 35) :
    10 ≤ ((m1Schedule wideLeaf m).levels.headD default).logLen ∧ ((m1Schedule wideLeaf m).levels.headD default).logLen ≤ 23 := by
  cases wideLeaf <;> interval_cases m <;> decide

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

theorem weightOf_le_fast100 {m : ℕ} {sch : Schedule} (h : fast100 m = some sch) {N : ℕ} (hN : N ≤ 2 ^ 13) :
    weightOf sch N ≤ 2 ^ 33 := by
  refine weightOf_le_of (fast100_logLen₀ h).2 ((weightNat_mono sch hN).trans ?_)
  have hm : 22 ≤ m ∧ m ≤ 35 := by
    by_contra hc
    have : fast100 m = none := by unfold fast100; split <;> simp_all
    simp [this] at h
  obtain ⟨h1, h2⟩ := hm
  interval_cases m <;> (simp only [fast100, Option.some.injEq] at h; subst h; decide)

theorem weightOf_le_m1 {m : ℕ} (h1 : 25 ≤ m) (h2 : m ≤ 35) {N : ℕ} (hN : N ≤ 2 ^ 13) :
    weightOf (m1Schedule wideLeaf m) N ≤ 2 ^ 33 := by
  refine weightOf_le_of (m1Schedule_logLen₀ h1 h2).2 ((weightNat_mono _ hN).trans ?_)
  cases wideLeaf <;> interval_cases m <;> decide

theorem weightOf_le_m1_pads {m kLog k0 : ℕ} (h1 : 25 ≤ m) (h2 : m ≤ 35) (hk : kLog ≤ 27) (h0 : k0 ≤ 10) :
    weightOf (m1Schedule wideLeaf m) (padDomain (_root_.Flock.Zk.PadLayout.new m kLog k0)) ≤ 2 ^ 33 :=
  weightOf_le_m1 h1 h2 (padDomain_le h2 hk h0)

section Deploy

variable {Coin Col T : Type}

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

theorem γ_le_48 (D : Deploy Coin Col T) (x : Msg) (hr : D.V.k x ≤ 768) (hC0 : 0 ≤ D.weight x)
    (hC : D.weight x ≤ 2 ^ 33) : γ D x ≤ 2 ^ (48 : ℝ) := by
  refine (γ_le_of D x (r := 768) (by exact_mod_cast hr) hC0 hC).trans ?_
  have hs : Real.sqrt (2 * 768 * 2 ^ 33) ≤ 3700000 := by
    rw [Real.sqrt_le_left (by norm_num)]
    norm_num
  rw [show (48 : ℝ) = ((48 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
  have h0 : 0 ≤ Real.sqrt (2 * 768 * 2 ^ 33) := Real.sqrt_nonneg _
  nlinarith

theorem γ_le_105 (D : Deploy Coin Col T) (x : Msg) (hr : D.V.k x ≤ 2 ^ 20) (hC0 : 0 ≤ D.weight x)
    (hC : D.weight x ≤ 2 ^ 64) : γ D x ≤ 2 ^ (105 : ℝ) := by
  refine (γ_le_of D x (r := 2 ^ 20) (by exact_mod_cast hr) hC0 hC).trans ?_
  have hs : Real.sqrt (2 * 2 ^ 20 * 2 ^ 64) ≤ 2 ^ 43 := by
    rw [Real.sqrt_le_left (by norm_num)]
    norm_num
  rw [show (105 : ℝ) = ((105 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
  have h0 : 0 ≤ Real.sqrt (2 * 2 ^ 20 * 2 ^ 64) := Real.sqrt_nonneg _
  nlinarith

theorem lam_ge_of_rounds (D : Deploy Coin Col T) (x : Msg) (hε0 : 0 ≤ D.εtable x)
    (hε : D.εtable x ≤ 2 ^ (-(205 : ℝ))) (hw : 210 ≤ D.waits x) (hK : D.cap x = cap) (hN : 2 ^ 10 ≤ D.N₀ x)
    (hN' : D.N₀ x ≤ 2 ^ 23) (hr : D.V.k x ≤ 2 ^ 20) (hC0 : 0 ≤ D.weight x) (hC : D.weight x ≤ 2 ^ 64) :
    201.98 ≤ lamOf (κ D x) (γ D x) :=
  lam_ge D x hε0 hε hw hK hN hN' (γ_le_105 D x hr hC0 hC)

end Deploy

end Claims.Flock.Numbers
