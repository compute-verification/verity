import Properties.Onsite
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.GCongr

/-!
# Onsite's pooled figures, proved (`Properties.Onsite`)

`ln`: `atanh2 a b` sums the first `TERMS` terms of `2·atanh(a/b) = log(1 + a/b) − log(1 − a/b)`
(`Real.hasSum_log_sub_log_of_abs_lt_one`), floored for the lower bound (`atanh2_lo`, the terms are positive) and
rounded up, plus `9/8` of the next term, for the upper (`atanh2_hi`: past `TERMS` each term is at most the next one's
times `(a/b)²^j`, and `1/(1 − (a/b)²) ≤ 9/8` at `a/b ≤ 1/3`). `ln x` splits `x = 2^k·(n/m)` with `m ≤ n ≤ 2m`, so
`(n − m)/(n + m) ≤ 1/3` and `log(n/m) = 2·atanh((n − m)/(n + m))` (`split_eq`), and `log 2 = 2·atanh(1/3)`.

The figures: `pooledCount` rounds `ln(1/δ)`'s upper bound up over `q` (`pooledCount_ge`); `least` returns its upper
end or a value that passes its test (`least_spec`), `choose` is `Nat.choose` (`choose_eq`), and `shortOk`'s test,
divided through, is `pooledShort_log`'s hypothesis at its `z` (`shortOk_spec`); `clog2 x` is at least `log2 x`
(`clog2_spec`).
-/

namespace Properties.Onsite

open Core Core.Onsite Core.Onsite.Pooled

theorem cdiv_ge (a b : ℕ) (hb : 0 < b) : a ≤ cdiv a b * b := by
  unfold cdiv
  have h1 := Nat.div_add_mod (a + b - 1) b
  have h2 := Nat.mod_lt (a + b - 1) hb
  have h3 : (a + b - 1) / b * b = b * ((a + b - 1) / b) := Nat.mul_comm _ _
  omega

theorem cdiv_ge_real (a b : ℕ) (hb : 0 < b) : (a : ℝ) / b ≤ cdiv a b := by
  rw [div_le_iff₀ (by exact_mod_cast hb)]
  exact_mod_cast cdiv_ge a b hb

theorem sumBelow_eq (f : ℕ → ℕ) (n : ℕ) : sumBelow f n = ∑ k ∈ Finset.range n, f k := by
  induction n with
  | zero => rfl
  | succ n ih => rw [sumBelow, ih, Finset.sum_range_succ]

theorem termDen_pos (b k : ℕ) (hb : 0 < b) : 0 < termDen b k :=
  Nat.mul_pos (Nat.succ_pos _) (Nat.pow_pos hb)

/-- The `k`-th term over `2^PREC`, as the series writes it. -/
theorem term_eq (a b k : ℕ) (hb : 0 < b) :
    ((termNum a k : ℕ) : ℝ) / (termDen b k : ℕ) =
      2 ^ PREC * ((2 : ℝ) * (1 / (2 * k + 1)) * ((a : ℝ) / b) ^ (2 * k + 1)) := by
  have hb' : (b : ℝ) ≠ 0 := by exact_mod_cast hb.ne'
  have hk : (2 * (k : ℝ) + 1) ≠ 0 := by positivity
  simp only [termNum, termDen]
  push_cast
  rw [div_pow]
  field_simp <;> ring

theorem series (a b : ℕ) (hb : 0 < b) (h3 : 3 * a ≤ b) :
    HasSum (fun k : ℕ => (2 : ℝ) * (1 / (2 * k + 1)) * ((a : ℝ) / b) ^ (2 * k + 1))
      (Real.log (1 + (a : ℝ) / b) - Real.log (1 - (a : ℝ) / b)) := by
  apply Real.hasSum_log_sub_log_of_abs_lt_one
  rw [abs_of_nonneg (by positivity), div_lt_one (by exact_mod_cast hb)]
  exact_mod_cast (show a < b by omega)

theorem atanh2_lo (a b : ℕ) (hb : 0 < b) (h3 : 3 * a ≤ b) :
    ((atanh2 a b).1 : ℝ) ≤ 2 ^ PREC * (Real.log (1 + (a : ℝ) / b) - Real.log (1 - (a : ℝ) / b)) := by
  have hle := sum_le_hasSum (Finset.range TERMS) (fun i _ => by positivity) (series a b hb h3)
  simp only [atanh2, sumBelow_eq, Nat.cast_sum]
  calc ∑ k ∈ Finset.range TERMS, ((termNum a k / termDen b k : ℕ) : ℝ)
      ≤ ∑ k ∈ Finset.range TERMS, ((termNum a k : ℕ) : ℝ) / (termDen b k : ℕ) :=
        Finset.sum_le_sum fun k _ => Nat.cast_div_le
    _ = 2 ^ PREC * ∑ k ∈ Finset.range TERMS, (2 : ℝ) * (1 / (2 * k + 1)) * ((a : ℝ) / b) ^ (2 * k + 1) := by
        rw [Finset.mul_sum]
        exact Finset.sum_congr rfl fun k _ => term_eq a b k hb
    _ ≤ _ := mul_le_mul_of_nonneg_left hle (by positivity)

theorem atanh2_hi (a b : ℕ) (hb : 0 < b) (h3 : 3 * a ≤ b) :
    2 ^ PREC * (Real.log (1 + (a : ℝ) / b) - Real.log (1 - (a : ℝ) / b)) ≤ ((atanh2 a b).2 : ℝ) := by
  have hS := series a b hb h3
  have hT := (hasSum_nat_add_iff' TERMS).mpr hS
  have hx0 : (0 : ℝ) ≤ (a : ℝ) / b := by positivity
  have hx3 : (a : ℝ) / b ≤ 1 / 3 := by
    rw [div_le_iff₀ (by exact_mod_cast hb : (0 : ℝ) < b)]
    have : (3 * a : ℝ) ≤ b := by exact_mod_cast h3
    linarith
  have hx2 : ((a : ℝ) / b) ^ 2 < 1 := by nlinarith
  have hg := (hasSum_geometric_of_lt_one (sq_nonneg ((a : ℝ) / b)) hx2).mul_left
    ((2 : ℝ) * (1 / (2 * (TERMS : ℕ) + 1)) * ((a : ℝ) / b) ^ (2 * TERMS + 1))
  have htail : ∀ n : ℕ, (2 : ℝ) * (1 / (2 * ((n + TERMS : ℕ) : ℝ) + 1)) * ((a : ℝ) / b) ^ (2 * (n + TERMS) + 1) ≤
      (2 : ℝ) * (1 / (2 * (TERMS : ℕ) + 1)) * ((a : ℝ) / b) ^ (2 * TERMS + 1) * (((a : ℝ) / b) ^ 2) ^ n := by
    intro n
    have hd : 1 / (2 * ((n + TERMS : ℕ) : ℝ) + 1) ≤ 1 / (2 * ((TERMS : ℕ) : ℝ) + 1) := by
      apply one_div_le_one_div_of_le (by positivity)
      push_cast
      linarith [(Nat.cast_nonneg n : (0 : ℝ) ≤ n)]
    have hp : ((a : ℝ) / b) ^ (2 * (n + TERMS) + 1) = ((a : ℝ) / b) ^ (2 * TERMS + 1) * (((a : ℝ) / b) ^ 2) ^ n := by
      rw [← pow_mul, ← pow_add]
      congr 1
      ring
    have h0 : 0 ≤ ((a : ℝ) / b) ^ (2 * TERMS + 1) * (((a : ℝ) / b) ^ 2) ^ n := by positivity
    rw [hp, mul_assoc _ (((a : ℝ) / b) ^ (2 * TERMS + 1))]
    exact mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hd (by norm_num)) h0
  have hle := hasSum_le htail hT hg
  have hinv : (1 - ((a : ℝ) / b) ^ 2)⁻¹ ≤ 9 / 8 := by
    rw [← one_div, div_le_iff₀ (by nlinarith)]
    nlinarith
  have htT : 0 ≤ (2 : ℝ) * (1 / (2 * (TERMS : ℕ) + 1)) * ((a : ℝ) / b) ^ (2 * TERMS + 1) := by positivity
  have hsum : Real.log (1 + (a : ℝ) / b) - Real.log (1 - (a : ℝ) / b) ≤
      ∑ i ∈ Finset.range TERMS, (2 : ℝ) * (1 / (2 * i + 1)) * ((a : ℝ) / b) ^ (2 * i + 1) +
        9 / 8 * ((2 : ℝ) * (1 / (2 * (TERMS : ℕ) + 1)) * ((a : ℝ) / b) ^ (2 * TERMS + 1)) := by
    have := mul_le_mul_of_nonneg_left hinv htT
    linarith
  have hb1 : ∀ k ∈ Finset.range TERMS,
      2 ^ PREC * ((2 : ℝ) * (1 / (2 * k + 1)) * ((a : ℝ) / b) ^ (2 * k + 1)) ≤
        ((cdiv (termNum a k) (termDen b k) : ℕ) : ℝ) := fun k _ =>
    (term_eq a b k hb).symm.le.trans (cdiv_ge_real _ _ (termDen_pos b k hb))
  have hb2 : 9 / 8 * (2 ^ PREC * ((2 : ℝ) * (1 / (2 * (TERMS : ℕ) + 1)) * ((a : ℝ) / b) ^ (2 * TERMS + 1))) ≤
      ((cdiv (9 * termNum a TERMS) (8 * termDen b TERMS) : ℕ) : ℝ) := by
    refine le_trans (le_of_eq ?_) (cdiv_ge_real _ _ (by have := termDen_pos b TERMS hb; omega))
    push_cast
    rw [mul_div_mul_comm, term_eq a b TERMS hb]
  have hsumc := Finset.sum_le_sum hb1
  rw [← Finset.mul_sum] at hsumc
  have h2 := mul_le_mul_of_nonneg_left hsum (by positivity : (0 : ℝ) ≤ 2 ^ PREC)
  simp only [atanh2, sumBelow_eq]
  push_cast
  linarith

theorem log_two_eq :
    Real.log 2 = Real.log (1 + ((1 : ℕ) : ℝ) / ((3 : ℕ) : ℝ)) - Real.log (1 - ((1 : ℕ) : ℝ) / ((3 : ℕ) : ℝ)) := by
  rw [← Real.log_div (by norm_num) (by norm_num)]
  norm_num

/-- `log(n/m) = 2·atanh((n − m)/(n + m))`. -/
theorem split_eq (n m : ℕ) (hm : 0 < m) (hmn : m ≤ n) :
    Real.log (1 + ((n - m : ℕ) : ℝ) / ((n + m : ℕ) : ℝ)) - Real.log (1 - ((n - m : ℕ) : ℝ) / ((n + m : ℕ) : ℝ)) =
      Real.log ((n : ℝ) / (m : ℕ)) := by
  rw [Nat.cast_sub hmn, Nat.cast_add]
  have hM : (0 : ℝ) < m := by exact_mod_cast hm
  have hN : (0 : ℝ) < n := by exact_mod_cast lt_of_lt_of_le hm hmn
  have hS : (0 : ℝ) < n + m := by linarith
  have hS' : (n : ℝ) + m ≠ 0 := hS.ne'
  have hM' : (m : ℝ) ≠ 0 := hM.ne'
  have e1 : 1 + ((n : ℝ) - m) / (n + m) = 2 * n / (n + m) := by
    field_simp <;> ring
  have e2 : 1 - ((n : ℝ) - m) / (n + m) = 2 * m / (n + m) := by
    field_simp <;> ring
  rw [e1, e2, ← Real.log_div (div_pos (by linarith) hS).ne' (div_pos (by linarith) hS).ne']
  congr 1
  field_simp <;> ring

theorem lnBrackets : LnBrackets := by
  intro x hd hdn
  rcases x with ⟨n, d⟩
  simp only at hd hdn
  have hq0 : n / d ≠ 0 := (Nat.div_pos hdn hd).ne'
  have hk := (Nat.log2_eq_iff hq0).mp rfl
  simp only [ln]
  generalize Nat.log2 (n / d) = k at hk ⊢
  obtain ⟨hk1, hk2⟩ := hk
  have hmn : d * 2 ^ k ≤ n := le_trans (Nat.mul_le_mul_left d hk1) (Nat.mul_div_le n d)
  have hn2m : n ≤ 2 * (d * 2 ^ k) := by
    have h1 := Nat.div_add_mod n d
    have h2 := Nat.mod_lt n hd
    have h3 : d * (n / d) + d ≤ 2 * (d * 2 ^ k) :=
      calc d * (n / d) + d = d * (n / d + 1) := by ring
        _ ≤ d * 2 ^ (k + 1) := Nat.mul_le_mul_left d (Nat.succ_le_of_lt hk2)
        _ = 2 * (d * 2 ^ k) := by ring
    omega
  have hm0 : 0 < d * 2 ^ k := Nat.mul_pos hd (Nat.two_pow_pos k)
  have h3 : 3 * (n - d * 2 ^ k) ≤ n + d * 2 ^ k := by omega
  have hb : 0 < n + d * 2 ^ k := by omega
  have l2lo := atanh2_lo 1 3 (by norm_num) (by norm_num)
  have l2hi := atanh2_hi 1 3 (by norm_num) (by norm_num)
  rw [← log_two_eq] at l2lo l2hi
  have rlo := atanh2_lo _ _ hb h3
  have rhi := atanh2_hi _ _ hb h3
  rw [split_eq n (d * 2 ^ k) hm0 hmn] at rlo rhi
  have hsplit : Real.log ((n : ℝ) / d) = k * Real.log 2 + Real.log ((n : ℝ) / ((d * 2 ^ k : ℕ) : ℝ)) := by
    have hN : (0 : ℝ) < n := by exact_mod_cast (show 0 < n by omega)
    have hM : (0 : ℝ) < ((d * 2 ^ k : ℕ) : ℝ) := by exact_mod_cast hm0
    have hD : (d : ℝ) ≠ 0 := by exact_mod_cast hd.ne'
    have e : (n : ℝ) / d = (n : ℝ) / ((d * 2 ^ k : ℕ) : ℝ) * 2 ^ k := by
      push_cast
      field_simp <;> ring
    rw [e, Real.log_mul (div_pos hN hM).ne' (by positivity), Real.log_pow]
    ring
  have hk0 : (0 : ℝ) ≤ k := Nat.cast_nonneg k
  have hP : (0 : ℝ) < 2 ^ PREC := by positivity
  push_cast
  rw [hsplit]
  constructor
  · rw [div_le_iff₀ hP]
    have := mul_le_mul_of_nonneg_left l2lo hk0
    linarith
  · rw [le_div_iff₀ hP]
    have := mul_le_mul_of_nonneg_left l2hi hk0
    linarith

/-- `pooledCount q δ` is at least `ln(1/δ)/q`. -/
theorem pooledCount_ge (q δ : Ratio) (hq : 0 < q.num) (hqd : 0 < q.den) (hδ : 0 < δ.num) (hδd : δ.num ≤ δ.den) :
    Real.log (1 / ((δ.num : ℝ) / δ.den)) / ((q.num : ℝ) / q.den) ≤ pooledCount q δ := by
  have hb := (lnBrackets ⟨δ.den, δ.num⟩ hδ hδd).2
  simp only at hb
  have hqn : (0 : ℝ) < q.num := by exact_mod_cast hq
  have hqd' : (0 : ℝ) < q.den := by exact_mod_cast hqd
  have hP : (0 : ℝ) < 2 ^ PREC := by positivity
  rw [one_div_div]
  unfold pooledCount
  refine le_trans ?_ (cdiv_ge_real _ _ (Nat.mul_pos hq (Nat.two_pow_pos _)))
  push_cast
  calc Real.log ((δ.den : ℝ) / δ.num) / ((q.num : ℝ) / q.den)
      = Real.log ((δ.den : ℝ) / δ.num) * ((q.den : ℝ) / q.num) := by ring
    _ ≤ ((Pooled.ln ⟨δ.den, δ.num⟩).2 : ℝ) / 2 ^ PREC * ((q.den : ℝ) / q.num) :=
        mul_le_mul_of_nonneg_right hb (by positivity)
    _ = _ := by ring

theorem integrityHorizonCovers : IntegrityHorizonCovers := by
  intro q δ hq hqd hδ hδd w hw
  have h := lt_of_lt_of_le (not_le.mp hw) (pooledCount_ge q δ hq hqd hδ hδd)
  have : w < pooledCount q δ := by exact_mod_cast h
  unfold faultyBlocks
  omega

/-- `least` returns its upper end, or a value that passes its test. -/
theorem least_spec (ok : ℕ → Bool) : ∀ fuel lo hi, least ok fuel lo hi = hi ∨ ok (least ok fuel lo hi) = true
  | 0, _, _ => Or.inl rfl
  | fuel + 1, lo, hi => by
    simp only [least]
    split_ifs with h1 h2
    · exact Or.inl rfl
    · rcases least_spec ok fuel lo ((lo + hi) / 2) with h | h
      · exact Or.inr (h ▸ h2)
      · exact Or.inr h
    · exact least_spec ok fuel _ hi

theorem foldl_choose (n : ℕ) :
    ∀ k, (List.range k).foldl (fun c i => c * (n - i) / (i + 1)) 1 = n.choose k
  | 0 => rfl
  | k + 1 => by
    rw [List.range_succ, List.foldl_append, foldl_choose n k]
    simp only [List.foldl_cons, List.foldl_nil]
    exact Nat.div_eq_of_eq_mul_left (Nat.succ_pos k) (Nat.choose_succ_right_eq n k).symm

theorem choose_eq (n k : ℕ) : Pooled.choose n k = n.choose k := by
  unfold Pooled.choose
  split_ifs with h
  · exact (Nat.choose_eq_zero_of_lt h).symm
  · exact foldl_choose n k

theorem integrityPeriodCovers : IntegrityPeriodCovers := by
  intro n kd e he hK
  have hdef : threshold n kd e = least (fun k => decide (Pooled.choose (n - k) kd * e.den ≤
      e.num * Pooled.choose n kd)) (fuel (n + 1)) 0 (n + 1) := rfl
  rw [hdef] at hK ⊢
  rcases least_spec (fun k => decide (Pooled.choose (n - k) kd * e.den ≤ e.num * Pooled.choose n kd))
    (fuel (n + 1)) 0 (n + 1) with h | h
  · rw [h] at hK
    omega
  · simp only [decide_eq_true_eq, choose_eq] at h
    rcases Nat.eq_zero_or_pos (n.choose kd) with h0 | h0
    · rw [h0, Nat.cast_zero, div_zero]
      positivity
    · rw [div_le_div_iff₀ (by exact_mod_cast h0) (by exact_mod_cast he)]
      exact_mod_cast h

/-- `shortOk`'s test, divided through, is `pooledShort_log`'s hypothesis at `z = z.num/z.den`. -/
theorem shortOk_spec (U : ℕ) (p δ z : Ratio) (M : ℕ) (hp : 0 < p.den) (hδ : 0 < δ.num) (hδd : δ.num ≤ δ.den)
    (h : shortOk U p δ z M = true) :
    1 < (z.num : ℝ) / z.den ∧
      (((z.num : ℝ) / z.den - 1) * U * ((p.num : ℝ) / p.den) + Real.log (1 / ((δ.num : ℝ) / δ.den))) /
        Real.log ((z.num : ℝ) / z.den) ≤ M := by
  simp only [shortOk, Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨⟨hz0, hz⟩, hN⟩ := h
  have hH := (lnBrackets ⟨δ.den, δ.num⟩ hδ hδd).2
  have hL := (lnBrackets z hz0 hz.le).1
  simp only at hH
  generalize (Pooled.ln ⟨δ.den, δ.num⟩).2 = H at hH hN
  generalize (Pooled.ln z).1 = L at hL hN
  have hzd : (0 : ℝ) < z.den := by exact_mod_cast hz0
  have hpd : (0 : ℝ) < p.den := by exact_mod_cast hp
  have hP : (0 : ℝ) < 2 ^ PREC := by positivity
  have hzd' : (z.den : ℝ) ≠ 0 := hzd.ne'
  have hpd' : (p.den : ℝ) ≠ 0 := hpd.ne'
  have hzr : 1 < (z.num : ℝ) / z.den := by
    rw [one_lt_div hzd]
    exact_mod_cast hz
  refine ⟨hzr, ?_⟩
  have hlog : 0 < Real.log ((z.num : ℝ) / z.den) := Real.log_pos hzr
  rw [div_le_iff₀ hlog, one_div_div]
  have hN' : ((z.num - z.den : ℕ) : ℝ) * U * p.num * 2 ^ PREC + (H : ℝ) * z.den * p.den ≤
      (M : ℝ) * L * z.den * p.den := by exact_mod_cast hN
  rw [Nat.cast_sub hz.le] at hN'
  have a1 : 2 ^ PREC * Real.log ((δ.den : ℝ) / δ.num) ≤ H := by
    have := (le_div_iff₀ hP).mp hH
    linarith
  have a2 : (L : ℝ) ≤ 2 ^ PREC * Real.log ((z.num : ℝ) / z.den) := by
    have := (div_le_iff₀ hP).mp hL
    linarith
  have b1 := mul_le_mul_of_nonneg_right a1 (by positivity : (0 : ℝ) ≤ z.den * p.den)
  have b2 := mul_le_mul_of_nonneg_right a2 (by positivity : (0 : ℝ) ≤ M * (z.den * p.den))
  have key : ((z.num : ℝ) - z.den) * U * p.num * 2 ^ PREC +
      2 ^ PREC * Real.log ((δ.den : ℝ) / δ.num) * (z.den * p.den) ≤
        2 ^ PREC * Real.log ((z.num : ℝ) / z.den) * (M * (z.den * p.den)) := by
    linarith
  have hc : (0 : ℝ) < 2 ^ PREC * (z.den * p.den) := by positivity
  refine le_of_mul_le_mul_left ?_ hc
  have e1 : 2 ^ PREC * (z.den * p.den) * (((z.num : ℝ) / z.den - 1) * U * ((p.num : ℝ) / p.den) +
      Real.log ((δ.den : ℝ) / δ.num)) = ((z.num : ℝ) - z.den) * U * p.num * 2 ^ PREC +
        2 ^ PREC * Real.log ((δ.den : ℝ) / δ.num) * (z.den * p.den) := by
    field_simp <;> ring
  rw [e1]
  linarith

theorem pooledShortCovers : PooledShortCovers := by
  intro U p δ hp hδ hδd hM
  unfold shortCount at hM ⊢
  split_ifs at hM ⊢ with h0 h1
  · obtain ⟨hz, hle⟩ := shortOk_spec U p δ _ 1 hp hδ hδd h1
    exact ⟨_, hz, by exact_mod_cast hle⟩
  · omega
  · rcases least_spec (fun M => shortOk U p δ ⟨M * p.den, U * p.num⟩ M) (fuel (U + 1))
      (U * p.num / p.den + 1) (U + 1) with h | h
    · rw [h] at hM
      omega
    · obtain ⟨hz, hle⟩ := shortOk_spec U p δ _ _ hp hδ hδd h
      exact ⟨_, hz, hle⟩

theorem clog2_spec (x : ℕ) : x ≤ 2 ^ clog2 x := by
  unfold clog2
  split_ifs with h
  · rw [pow_zero]
    exact h
  · have h1 : x - 1 ≠ 0 := by omega
    have := (Nat.log2_lt h1).mp (Nat.lt_add_one _)
    omega

theorem horizonLeakCovers : HorizonLeakCovers := by
  intro B every units q δ hq hqd hδ hδd
  refine ⟨?_, fun N hN => ?_, rfl⟩
  · have h := pooledCount_ge q δ hq hqd hδ hδd
    have hq' : (0 : ℝ) < (q.num : ℝ) / q.den := by
      have : (0 : ℝ) < q.num := by exact_mod_cast hq
      have : (0 : ℝ) < q.den := by exact_mod_cast hqd
      positivity
    show _ ≤ _ * (pooledCount q δ : ℝ)
    rw [div_le_iff₀ hq'] at h
    linarith
  · show (N + 1) ^ pooledCount q δ * 2 ^ (BLOCK_BITS * pooledCount q δ) ≤
      2 ^ (pooledCount q δ * (BLOCK_BITS + clog2 (B * units + 1)))
    have hc := clog2_spec (B * units + 1)
    calc (N + 1) ^ pooledCount q δ * 2 ^ (BLOCK_BITS * pooledCount q δ)
        ≤ (2 ^ clog2 (B * units + 1)) ^ pooledCount q δ * 2 ^ (BLOCK_BITS * pooledCount q δ) := by
          gcongr
          exact le_trans (Nat.add_le_add_right hN 1) hc
      _ = 2 ^ (pooledCount q δ * (BLOCK_BITS + clog2 (B * units + 1))) := by
          rw [← pow_mul, ← pow_add]
          congr 1
          ring

end Properties.Onsite
