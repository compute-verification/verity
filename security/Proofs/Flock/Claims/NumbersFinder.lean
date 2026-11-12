module

public import Proofs.Flock.Claims.Numbers

/-!
# The first-hit finder's terms, as numbers

The by-position first-hit finder bounds the probability that an accepted run opens a value off the committed trace by
its expected calls of SHA-512 over `2^256`, four times. With every run of at most `T` calls, the prover's and the
verifier's, sessions extracted in at most `e` expected reruns each, a sample of `N_s` sessions in the mean that needs
`K_s` gates in the mean, `K_m` counted session by session, and `c` calls of the run's own opening checks, that term
is

`4·(T·(1 + e·N_s + K_s + e·K_m) + (1 + K_s)·c)/2^256`.

A call is one of SHA-512 on a whole message, whatever its length, as a finder's cost counts them: none of these
numbers counts compressions.

This file holds that expression over the rationals (`firstHitBound`), where an instance at numerals is decided by
arithmetic, and its value at sessions of one table:

* `firstHitBound_table1`: at most `2^-100` at `T = 2^80` calls, `e = 2592237061 = 1 + 210·12343986` (a session of one
  table, at most `2^23` level-0 symbols), `N_s = 2^30`, `K_s = K_m = 2^42` and `c = 2^80`;
* `real_le_firstHitBound`, `real_le_two_rpow_neg_100`, `real_table1_le`: the same expression over the reals, at any
  reals within a design's numbers, with `2^256` and `2^-100` real powers as the finder's statements write them;
* `firstHitBound_table1_513`, `real_table1_513_le`: the same at `513·2^80` in place of `2^80`, every call charged as
  the `513` compressions of SHA-512 on a row of `2^16` bytes, at `N_s = 2^30` and `K_s = K_m = 2^33`;
* `sqrt_term_le_strict`: the rewinding term `√(k·γ·p)` beside it is at most `2^-128` at `k ≤ 4 + 4402·2^30` rounds,
  `γ ≤ 2^51` and `p ≤ q²/2^513` at `q ≤ 2·2^80`, the strict birthday bound at a finder of two runs.
-/

@[expose] public section

namespace Claims.Flock.Numbers

/-! ## The bound over the rationals -/

/-- The first-hit finders' expected calls, as an expression: `T·(1 + e·N_s + K_s + e·K_m) + (1 + K_s)·c`. -/
def firstHitCalls (T e Ns Ks Km c : ℚ) : ℚ :=
  T * (1 + e * Ns + Ks + e * Km) + (1 + Ks) * c

/-- **The by-position first-hit finder's term**: `4·(T·(1 + e·N_s + K_s + e·K_m) + (1 + K_s)·c)/2^256`. -/
def firstHitBound (T e Ns Ks Km c : ℚ) : ℚ :=
  4 * firstHitCalls T e Ns Ks Km c / 2 ^ 256

/-- **At most `2^-100` at sessions of one table**: runs of `2^80` calls, `e = 1 + 210·12343986`, `2^30` sessions, a
sample needing `2^42` gates in the mean, counted either way, and `2^80` calls of the run's own opening checks. -/
theorem firstHitBound_table1 :
    firstHitBound (2 ^ 80) 2592237061 (2 ^ 30) (2 ^ 42) (2 ^ 42) (2 ^ 80) ≤ 1 / 2 ^ 100 := by
  norm_num [firstHitBound, firstHitCalls]

/-! ## Over the reals -/

/-- **The bound over the reals**, `2^256` a real power. -/
theorem firstHitBound_cast (T e Ns Ks Km c : ℚ) :
    ((firstHitBound T e Ns Ks Km c : ℚ) : ℝ) =
      4 * ((T : ℝ) * (1 + (e : ℝ) * (Ns : ℝ) + (Ks : ℝ) + (e : ℝ) * (Km : ℝ)) + (1 + (Ks : ℝ)) * (c : ℝ)) /
        2 ^ (256 : ℝ) := by
  rw [show (256 : ℝ) = ((256 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
  simp only [firstHitBound, firstHitCalls]
  push_cast
  all_goals ring

/-- **The expression at reals within a design's numbers is at most the bound at them**: runs of `T ≤ T'` calls,
sessions extracted in `e ≤ e'` expected reruns, `N_s ≤ N_s'` sessions, `K_s ≤ K_s'` and `K_m ≤ K_m'` gates, and
`c ≤ c'` calls of the run's own. -/
theorem real_le_firstHitBound {T e Ns Ks Km c : ℝ} {T' e' Ns' Ks' Km' c' : ℚ} (hT0 : 0 ≤ T) (he0 : 0 ≤ e)
    (hNs0 : 0 ≤ Ns) (hKs0 : 0 ≤ Ks) (hKm0 : 0 ≤ Km) (hc0 : 0 ≤ c) (hT : T ≤ T') (he : e ≤ e') (hNs : Ns ≤ Ns')
    (hKs : Ks ≤ Ks') (hKm : Km ≤ Km') (hc : c ≤ c') :
    4 * (T * (1 + e * Ns + Ks + e * Km) + (1 + Ks) * c) / 2 ^ (256 : ℝ) ≤
      ((firstHitBound T' e' Ns' Ks' Km' c' : ℚ) : ℝ) := by
  have hT' : (0 : ℝ) ≤ T' := hT0.trans hT
  have he' : (0 : ℝ) ≤ e' := he0.trans he
  have hKs' : (0 : ℝ) ≤ Ks' := hKs0.trans hKs
  have h1 : e * Ns ≤ (e' : ℝ) * (Ns' : ℝ) := mul_le_mul he hNs hNs0 he'
  have h2 : e * Km ≤ (e' : ℝ) * (Km' : ℝ) := mul_le_mul he hKm hKm0 he'
  have h3 : 0 ≤ e * Ns := mul_nonneg he0 hNs0
  have h4 : 0 ≤ e * Km := mul_nonneg he0 hKm0
  have h5 : T * (1 + e * Ns + Ks + e * Km) ≤
      (T' : ℝ) * (1 + (e' : ℝ) * (Ns' : ℝ) + (Ks' : ℝ) + (e' : ℝ) * (Km' : ℝ)) :=
    mul_le_mul hT (by linarith) (by linarith) hT'
  have h6 : (1 + Ks) * c ≤ (1 + (Ks' : ℝ)) * (c' : ℝ) := mul_le_mul (by linarith) hc hc0 (by linarith)
  have h7 : 4 * (T * (1 + e * Ns + Ks + e * Km) + (1 + Ks) * c) ≤
      4 * ((T' : ℝ) * (1 + (e' : ℝ) * (Ns' : ℝ) + (Ks' : ℝ) + (e' : ℝ) * (Km' : ℝ)) + (1 + (Ks' : ℝ)) * (c' : ℝ)) := by
    linarith
  rw [firstHitBound_cast]
  exact div_le_div_of_nonneg_right h7 (by positivity)

/-- `1/2^100` over the rationals is `2^-100`, a real power. -/
theorem cast_two_pow_neg_100 : ((1 / 2 ^ 100 : ℚ) : ℝ) = 2 ^ (-(100 : ℝ)) := by
  rw [Real.rpow_neg (by norm_num), show (100 : ℝ) = ((100 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
  push_cast
  all_goals ring

/-- **The expression over the reals is at most `2^-100`** wherever the rational bound at numbers above it is. -/
theorem real_le_two_rpow_neg_100 {T e Ns Ks Km c : ℝ} {T' e' Ns' Ks' Km' c' : ℚ} (hT0 : 0 ≤ T) (he0 : 0 ≤ e)
    (hNs0 : 0 ≤ Ns) (hKs0 : 0 ≤ Ks) (hKm0 : 0 ≤ Km) (hc0 : 0 ≤ c) (hT : T ≤ T') (he : e ≤ e') (hNs : Ns ≤ Ns')
    (hKs : Ks ≤ Ks') (hKm : Km ≤ Km') (hc : c ≤ c') (h : firstHitBound T' e' Ns' Ks' Km' c' ≤ 1 / 2 ^ 100) :
    4 * (T * (1 + e * Ns + Ks + e * Km) + (1 + Ks) * c) / 2 ^ (256 : ℝ) ≤ 2 ^ (-(100 : ℝ)) := by
  rw [← cast_two_pow_neg_100]
  exact (real_le_firstHitBound hT0 he0 hNs0 hKs0 hKm0 hc0 hT he hNs hKs hKm hc).trans (Rat.cast_le.2 h)

/-- **The first-hit term is at most `2^-100` at sessions of one table**: at runs of `T ≤ 2^80` calls of SHA-512,
sessions extracted in `e ≤ 2592237061 = 1 + 210·12343986` expected reruns, `N_s ≤ 2^30` sessions, `K_s ≤ 2^42` and
`K_m ≤ 2^42` gates in the mean, and `c ≤ 2^80` calls of the run's own opening checks. -/
theorem real_table1_le {T e Ns Ks Km c : ℝ} (hT0 : 0 ≤ T) (he0 : 0 ≤ e) (hNs0 : 0 ≤ Ns) (hKs0 : 0 ≤ Ks)
    (hKm0 : 0 ≤ Km) (hc0 : 0 ≤ c) (hT : T ≤ 2 ^ 80) (he : e ≤ 2592237061) (hNs : Ns ≤ 2 ^ 30) (hKs : Ks ≤ 2 ^ 42)
    (hKm : Km ≤ 2 ^ 42) (hc : c ≤ 2 ^ 80) :
    4 * (T * (1 + e * Ns + Ks + e * Km) + (1 + Ks) * c) / 2 ^ (256 : ℝ) ≤ 2 ^ (-(100 : ℝ)) := by
  refine real_le_two_rpow_neg_100 (T' := 2 ^ 80) (e' := 2592237061) (Ns' := 2 ^ 30) (Ks' := 2 ^ 42)
    (Km' := 2 ^ 42) (c' := 2 ^ 80) hT0 he0 hNs0 hKs0 hKm0 hc0 ?_ ?_ ?_ ?_ ?_ ?_ firstHitBound_table1
  · rw [Rat.cast_pow, Rat.cast_ofNat]
    exact hT
  · rw [Rat.cast_ofNat]
    exact he
  · rw [Rat.cast_pow, Rat.cast_ofNat]
    exact hNs
  · rw [Rat.cast_pow, Rat.cast_ofNat]
    exact hKs
  · rw [Rat.cast_pow, Rat.cast_ofNat]
    exact hKm
  · rw [Rat.cast_pow, Rat.cast_ofNat]
    exact hc

/-! ## With every call charged at its longest

A finder's cost counts calls. Charged instead by compressions, a call of SHA-512 on at most `2^16` bytes is at most
`513` of them, and the expression then reads `513·T` where it read `T` and `513·c` where it read `c`. No premise here
counts compressions: these are the expression's values at those numbers. -/

/-- **At most `2^-100` with every call charged as `513`**: the expression at `513·2^80` in place of a run's `2^80`
calls and of its own opening checks', at `e = 1 + 210·12343986`, `2^30` sessions and `2^33` in place of the gates,
counted either way. -/
theorem firstHitBound_table1_513 :
    firstHitBound (513 * 2 ^ 80) 2592237061 (2 ^ 30) (2 ^ 33) (2 ^ 33) (513 * 2 ^ 80) ≤ 1 / 2 ^ 100 := by
  norm_num [firstHitBound, firstHitCalls]

/-- **The first-hit term is at most `2^-100` at sessions of one table with every call charged as `513`**: at
`T ≤ 513·2^80`, sessions extracted in `e ≤ 2592237061` expected reruns, `N_s ≤ 2^30` sessions, `K_s ≤ 2^33` and
`K_m ≤ 2^33`, and `c ≤ 513·2^80`. -/
theorem real_table1_513_le {T e Ns Ks Km c : ℝ} (hT0 : 0 ≤ T) (he0 : 0 ≤ e) (hNs0 : 0 ≤ Ns) (hKs0 : 0 ≤ Ks)
    (hKm0 : 0 ≤ Km) (hc0 : 0 ≤ c) (hT : T ≤ 513 * 2 ^ 80) (he : e ≤ 2592237061) (hNs : Ns ≤ 2 ^ 30)
    (hKs : Ks ≤ 2 ^ 33) (hKm : Km ≤ 2 ^ 33) (hc : c ≤ 513 * 2 ^ 80) :
    4 * (T * (1 + e * Ns + Ks + e * Km) + (1 + Ks) * c) / 2 ^ (256 : ℝ) ≤ 2 ^ (-(100 : ℝ)) := by
  refine real_le_two_rpow_neg_100 (T' := 513 * 2 ^ 80) (e' := 2592237061) (Ns' := 2 ^ 30) (Ks' := 2 ^ 33)
    (Km' := 2 ^ 33) (c' := 513 * 2 ^ 80) hT0 he0 hNs0 hKs0 hKm0 hc0 ?_ ?_ ?_ ?_ ?_ ?_ firstHitBound_table1_513
  · exact hT.trans (le_of_eq (by norm_num))
  · rw [Rat.cast_ofNat]
    exact he
  · rw [Rat.cast_pow, Rat.cast_ofNat]
    exact hNs
  · rw [Rat.cast_pow, Rat.cast_ofNat]
    exact hKs
  · rw [Rat.cast_pow, Rat.cast_ofNat]
    exact hKm
  · exact hc.trans (le_of_eq (by norm_num))

/-! ## The rewinding term beside it -/

set_option exponentiation.threshold 600 in
/-- **The rewinding term `√(k·γ·p)` is at most `2^-128` under the strict birthday bound**: at `k ≤ 4 + 4402·2^30`
rounds, a session's `γ ≤ 2^51`, and `p ≤ q²/2^513` at a finder of `q ≤ 2·2^80` calls on every coin, two runs of
`2^80`. -/
theorem sqrt_term_le_strict {k γ p q : ℝ} (hγ0 : 0 ≤ γ) (hp0 : 0 ≤ p) (hq0 : 0 ≤ q) (hk : k ≤ 4 + 4402 * 2 ^ 30)
    (hγ : γ ≤ 2 ^ (51 : ℝ)) (hq : q ≤ 2 * 2 ^ 80) (hp : p ≤ q ^ 2 / 2 ^ (513 : ℝ)) :
    Real.sqrt (k * γ * p) ≤ 2 ^ (-(128 : ℝ)) := by
  have h51 : (2 : ℝ) ^ (51 : ℝ) = 2 ^ 51 := by
    rw [show (51 : ℝ) = ((51 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
  have h513 : (2 : ℝ) ^ (513 : ℝ) = 2 ^ 513 := by
    rw [show (513 : ℝ) = ((513 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
  have h128 : (2 : ℝ) ^ (-(128 : ℝ)) = 1 / 2 ^ 128 := by
    rw [Real.rpow_neg (by norm_num), show (128 : ℝ) = ((128 : ℕ) : ℝ) by norm_num, Real.rpow_natCast, one_div]
  rw [h51] at hγ
  rw [h513] at hp
  have hq2 : q ^ 2 ≤ (2 * 2 ^ 80) ^ 2 := pow_le_pow_left₀ hq0 hq 2
  have hp' : p ≤ (2 * 2 ^ 80) ^ 2 / 2 ^ 513 := hp.trans (div_le_div_of_nonneg_right hq2 (by positivity))
  have hkγ : k * γ ≤ (4 + 4402 * 2 ^ 30) * 2 ^ 51 := mul_le_mul hk hγ hγ0 (by positivity)
  have h : k * γ * p ≤ (4 + 4402 * 2 ^ 30) * 2 ^ 51 * ((2 * 2 ^ 80) ^ 2 / 2 ^ 513) :=
    mul_le_mul hkγ hp' hp0 (by positivity)
  rw [h128, Real.sqrt_le_left (by positivity)]
  refine h.trans ?_
  norm_num

end Claims.Flock.Numbers
