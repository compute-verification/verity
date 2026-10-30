import Proofs.Flock.Claims.Bits
import Mathlib.Analysis.Complex.ExponentialBounds

/-!
# Flock's numbers: `m′`, `K`, `k`, `κ` and `λ`

The numbers Flock's claims read (`Claims.Flock.κ`, `γ`, `lamOf`) at the deployed parameters, each a theorem:

* `E^rep`'s waits `m′ = 210` (`waits`) and its truncation `K = 2^256` (`cap`);
* `k = ⌈4N₀/e⌉` exactly, at every level-0 length `N₀ = 2^d` a `fast100` schedule has, `d ∈ {10, …, 15, 18, …, 23}`
  (`kRep_eq`): from `1507` at `d = 10` to `12343986` at `d = 23`;
* `κ ≤ 2^{−203} + 2^{−210} + 2^{−231} + 2^{−250} ≤ 2^{−202.98}` once `ε_table ≤ 2^{−205}`, `m′ ≥ 210`, `K = 2^256` and
  `2^10 ≤ N₀ ≤ 2^23` (`κ_le_sum`, `κ_le`);
* `λ = lamOf κ γ ≥ 201.98` whenever moreover `γ ≤ 2^105` (`lam_ge`): the hash side, `255.5 − ½ log₂ γ`, is then at
  least `203`, so the table error binds and `γ`'s exact value doesn't move the bits. `γ` at the deployed weight is at
  most `2^48` at every schedule and up to `768` rounds (`NumbersGamma`).

Each is stated over a `Deploy`'s fields, so it applies to the deployed record once its fields are pinned to these values.
-/

namespace Claims.Flock.Numbers

/-- `E^rep`'s waits `m′`. -/
def waits : ℕ := 210

/-- `E^rep`'s truncation `K`: each wait gives up after `2^256` reruns. -/
def cap : ℕ := 2 ^ 256

theorem waits_ge : 210 ≤ waits := le_rfl

/-! ## `k = ⌈4N₀/e⌉` -/

/-- `E^rep`'s accepted reruns per wait at a level-0 codeword of `2^d` symbols: `⌈4·2^d/e⌉`. -/
noncomputable def kRep (d : ℕ) : ℕ := ⌈4 * (2 : ℝ) ^ d / Real.exp 1⌉₊

/-- `⌈a/e⌉ = n` from `e`'s nine-digit bounds. -/
theorem ceil_div_exp_one_eq {a : ℝ} {n : ℕ} (hn : n ≠ 0) (h1 : ((n : ℝ) - 1) * 2.7182818286 < a)
    (h2 : a ≤ n * 2.7182818283) : ⌈a / Real.exp 1⌉₊ = n := by
  have he0 : 0 < Real.exp 1 := Real.exp_pos 1
  have hlo := Real.exp_one_gt_d9
  have hhi := Real.exp_one_lt_d9
  have hn1 : (1 : ℝ) ≤ n := by exact_mod_cast Nat.one_le_iff_ne_zero.2 hn
  rw [Nat.ceil_eq_iff hn, Nat.cast_sub (Nat.one_le_iff_ne_zero.2 hn), Nat.cast_one, lt_div_iff₀ he0,
    div_le_iff₀ he0]
  constructor
  · calc ((n : ℝ) - 1) * Real.exp 1 ≤ ((n : ℝ) - 1) * 2.7182818286 :=
          mul_le_mul_of_nonneg_left hhi.le (by linarith)
      _ < a := h1
  · calc a ≤ n * 2.7182818283 := h2
      _ ≤ n * Real.exp 1 := mul_le_mul_of_nonneg_left hlo.le (by linarith)

/-- `k` at each level-0 length a `fast100` schedule has (`2^10` at `m = 22` to `2^23` at `m = 35`). -/
def kVal : ℕ → ℕ
  | 10 => 1507
  | 11 => 3014
  | 12 => 6028
  | 13 => 12055
  | 14 => 24110
  | 15 => 48219
  | 18 => 385750
  | 19 => 771500
  | 20 => 1542999
  | 21 => 3085997
  | 22 => 6171993
  | 23 => 12343986
  | _ => 0

/-- **`k = ⌈4N₀/e⌉`, exactly**, at every level-0 length of a `fast100` schedule. -/
theorem kRep_eq {d : ℕ} (hd : d ∈ [10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23]) : kRep d = kVal d := by
  unfold kRep
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hd
  rcases hd with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    exact ceil_div_exp_one_eq (by simp [kVal]) (by norm_num [kVal]) (by norm_num [kVal])

/-- `k` at `N₀ ≥ 2^10` is at least `1507`. -/
theorem k_ge {N : ℝ} (hN : 2 ^ 10 ≤ N) : 1507 ≤ ⌈4 * N / Real.exp 1⌉₊ := by
  have he0 : 0 < Real.exp 1 := Real.exp_pos 1
  have hhi := Real.exp_one_lt_d9
  have h : ((1506 : ℕ) : ℝ) < 4 * N / Real.exp 1 := by
    rw [lt_div_iff₀ he0]
    push_cast
    linarith
  exact Nat.lt_ceil.2 h

/-- `k` at `N₀ ≤ 2^23` is at most `2^24`. -/
theorem k_le {N : ℝ} (hN : N ≤ 2 ^ 23) : ⌈4 * N / Real.exp 1⌉₊ ≤ 2 ^ 24 := by
  have he0 : 0 < Real.exp 1 := Real.exp_pos 1
  have hlo := Real.exp_one_gt_d9
  rw [Nat.ceil_le, div_le_iff₀ he0]
  push_cast
  linarith

/-! ## `κ` -/

section Deploy

variable {Coin Col T : Type}

/-- `e^{−k/8} ≤ 2^{−250}` at `k ≥ 1507`. -/
theorem exp_neg_k_le {k : ℕ} (hk : 1507 ≤ k) : Real.exp (-(k : ℝ) / 8) ≤ 2 ^ (-(250 : ℝ)) := by
  have h188 : Real.exp (-(k : ℝ) / 8) ≤ Real.exp (-188) := by
    rw [Real.exp_le_exp]
    have : (1507 : ℝ) ≤ k := by exact_mod_cast hk
    linarith
  refine h188.trans ?_
  have he : (27 / 10 : ℝ) ^ 188 ≤ Real.exp 188 := by
    rw [show (188 : ℝ) = ((188 : ℕ) : ℝ) by norm_num, ← Real.exp_one_pow]
    exact pow_le_pow_left₀ (by norm_num) (by linarith [Real.exp_one_gt_d9]) _
  rw [Real.exp_neg, Real.rpow_neg (by norm_num), show (250 : ℝ) = ((250 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
  refine inv_anti₀ (by positivity) (le_trans ?_ he)
  norm_num

/-- `1 + 2^{−7} + 2^{−28} + 2^{−47} ≤ 2^{1/50}`. -/
theorem slack_le : (1 + 2⁻¹ ^ 7 + 2⁻¹ ^ 28 + 2⁻¹ ^ 47 : ℝ) ≤ 2 ^ ((1 : ℝ) / 50) := by
  have h50 : (1 + 2⁻¹ ^ 7 + 2⁻¹ ^ 28 + 2⁻¹ ^ 47 : ℝ) ^ 50 ≤ 2 := by norm_num
  have h0 : (0 : ℝ) ≤ 1 + 2⁻¹ ^ 7 + 2⁻¹ ^ 28 + 2⁻¹ ^ 47 := by positivity
  have h2 : ((2 : ℝ) ^ ((1 : ℝ) / 50)) ^ (50 : ℕ) = 2 := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul (by norm_num)]
    norm_num
  rw [← pow_le_pow_iff_left₀ h0 (by positivity) (by norm_num : (50 : ℕ) ≠ 0), h2]
  exact h50

/-- **`κ` at the deployed numbers, term by term**: `ε_table ≤ 2^{−205}`, `m′ ≥ 210`, `K = 2^256` and
`2^10 ≤ N₀ ≤ 2^23` give `κ ≤ 2^{−203} + 2^{−210} + 2^{−231} + 2^{−250}`. -/
theorem κ_le_sum (D : Deploy Coin Col T) (x : Msg) (hε : D.εtable x ≤ 2 ^ (-(205 : ℝ))) (hw : 210 ≤ D.waits x)
    (hK : D.cap x = cap) (hN : 2 ^ 10 ≤ D.N₀ x) (hN' : D.N₀ x ≤ 2 ^ 23) :
    κ D x ≤ 2 ^ (-(203 : ℝ)) + 2 ^ (-(210 : ℝ)) + 2 ^ (-(231 : ℝ)) + 2 ^ (-(250 : ℝ)) := by
  unfold κ
  have h1 : 4 * D.εtable x ≤ 2 ^ (-(203 : ℝ)) := by
    have : (2 : ℝ) ^ (-(203 : ℝ)) = 4 * 2 ^ (-(205 : ℝ)) := by
      rw [show (-(203 : ℝ)) = 2 + -(205 : ℝ) by norm_num, Real.rpow_add (by norm_num)]; norm_num
    rw [this]; linarith
  have h2 : (2 : ℝ) ^ (-(D.waits x : ℝ)) ≤ 2 ^ (-(210 : ℝ)) :=
    Real.rpow_le_rpow_of_exponent_le (by norm_num) (neg_le_neg (by exact_mod_cast hw))
  have hk1 : 1507 ≤ D.k x := k_ge hN
  have hk2 : D.k x ≤ 2 ^ 24 := k_le hN'
  have h3 : 2 * (D.k x : ℝ) / D.cap x ≤ 2 ^ (-(231 : ℝ)) := by
    have hk : (D.k x : ℝ) ≤ 2 ^ 24 := by exact_mod_cast hk2
    have hc : (D.cap x : ℝ) = 2 ^ 256 := by rw [hK, cap]; push_cast; ring
    rw [hc, Real.rpow_neg (by norm_num), show (231 : ℝ) = ((231 : ℕ) : ℝ) by norm_num, Real.rpow_natCast,
      div_le_iff₀ (by positivity)]
    calc 2 * (D.k x : ℝ) ≤ 2 * 2 ^ 24 := by linarith
      _ = (2 ^ 231)⁻¹ * 2 ^ 256 := by norm_num
  have h4 : Real.exp (-(D.k x : ℝ) / 8) ≤ 2 ^ (-(250 : ℝ)) := exp_neg_k_le hk1
  exact add_le_add (add_le_add (add_le_add h1 h2) h3) h4

/-- **`κ ≤ 2^{−202.98}`** at the deployed numbers. -/
theorem κ_le (D : Deploy Coin Col T) (x : Msg) (hε : D.εtable x ≤ 2 ^ (-(205 : ℝ))) (hw : 210 ≤ D.waits x)
    (hK : D.cap x = cap) (hN : 2 ^ 10 ≤ D.N₀ x) (hN' : D.N₀ x ≤ 2 ^ 23) : κ D x ≤ 2 ^ (-(202.98 : ℝ)) := by
  have e : ∀ n : ℕ, (2 : ℝ) ^ (-((203 + n : ℕ) : ℝ)) = 2 ^ (-(203 : ℝ)) * 2⁻¹ ^ n := by
    intro n
    rw [Nat.cast_add, neg_add, Real.rpow_add (by norm_num), Real.rpow_neg (by norm_num) (n : ℝ),
      Real.rpow_natCast, inv_pow]
    norm_num
  have e7 : (2 : ℝ) ^ (-(210 : ℝ)) = 2 ^ (-(203 : ℝ)) * 2⁻¹ ^ 7 := by rw [← e 7]; norm_num
  have e28 : (2 : ℝ) ^ (-(231 : ℝ)) = 2 ^ (-(203 : ℝ)) * 2⁻¹ ^ 28 := by rw [← e 28]; norm_num
  have e47 : (2 : ℝ) ^ (-(250 : ℝ)) = 2 ^ (-(203 : ℝ)) * 2⁻¹ ^ 47 := by rw [← e 47]; norm_num
  calc κ D x ≤ 2 ^ (-(203 : ℝ)) + 2 ^ (-(210 : ℝ)) + 2 ^ (-(231 : ℝ)) + 2 ^ (-(250 : ℝ)) :=
        κ_le_sum D x hε hw hK hN hN'
    _ = 2 ^ (-(203 : ℝ)) * (1 + 2⁻¹ ^ 7 + 2⁻¹ ^ 28 + 2⁻¹ ^ 47) := by rw [e7, e28, e47]; ring
    _ ≤ 2 ^ (-(203 : ℝ)) * 2 ^ ((1 : ℝ) / 50) := mul_le_mul_of_nonneg_left slack_le (by positivity)
    _ = 2 ^ (-(202.98 : ℝ)) := by
        rw [← Real.rpow_add (by norm_num)]; norm_num

/-! ## `λ` -/

/-- **`λ ≥ 201.98`**: at the deployed numbers, with `γ ≤ 2^105`, `lamOf κ γ ≥ 201.98`. The hash side is then at least
`255.5 − 105/2 = 203`, so the table error binds. -/
theorem lam_ge (D : Deploy Coin Col T) (x : Msg) (hε0 : 0 ≤ D.εtable x) (hε : D.εtable x ≤ 2 ^ (-(205 : ℝ)))
    (hw : 210 ≤ D.waits x) (hK : D.cap x = cap) (hN : 2 ^ 10 ≤ D.N₀ x) (hN' : D.N₀ x ≤ 2 ^ 23)
    (hγ : γ D x ≤ 2 ^ (105 : ℝ)) : 201.98 ≤ lamOf (κ D x) (γ D x) := by
  have hκ0 := κ_pos D x hε0
  have hκ := κ_le D x hε hw hK hN hN'
  rcases (γ_nonneg D x).lt_or_eq with hγ0 | hγ0
  · have h := lamOf_ge hκ0 hκ hγ0 hγ
    rw [min_eq_left (by norm_num)] at h
    linarith
  · unfold lamOf
    rw [← hγ0, Real.logb_zero, zero_div, sub_zero]
    have h1 : (202.98 : ℝ) ≤ Real.logb 2 (1 / κ D x) := by
      rw [one_div, Real.logb_inv, le_neg, ← Real.rpow_le_rpow_left_iff (by norm_num : (1 : ℝ) < 2),
        Real.rpow_logb (by norm_num) (by norm_num) hκ0]
      exact hκ
    have h2 : (202.98 : ℝ) ≤ min (Real.logb 2 (1 / κ D x)) 255.5 := le_min h1 (by norm_num)
    linarith

end Deploy

end Claims.Flock.Numbers
