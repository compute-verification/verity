module

public import Proofs.Flock.Claims.Numbers

/-!
# The first-hit finder's binding term, as a number

The by-position first-hit finder bounds the probability that an accepted run opens a value off the committed trace by
its expected SHA-512 compressions over `2^256`, four times. At a run of `T` compressions with at most `n` sessions,
each extracted in at most `e` expected reruns, and a sample whose proof blocks need `K̄` gates in the mean, that is

`4·T·(1 + n·e + (1 + e)·K̄)/2^256`, with `e = 1 + m′·k`

at a session of `k` accepted reruns a wait and `m′ = 210` waits (`waits`): the first run with every session
extracted, and for each needed gate a run with the sessions holding it extracted.

This file holds that expression over the rationals (`firstHitBound`), where an instance at numerals is decided by
arithmetic, and its value at two session sizes:

* `k_le_of_one_table`, `k_le_of_tables_le_32`: a session whose level-0 lengths sum to at most `2^23`, one table, has
  `k = ⌈4N₀/e⌉ ≤ 12343986`, and one of at most `32·2^23`, `32` tables, has `k ≤ 395007543`;
* `firstHitBound_le`: the bound rises with `T`, `n`, `k` and `K̄`, so its value at a design's numbers bounds it at
  every statement within them;
* `firstHitBound_table1`: at most `2^-100` at `T = 2^80`, `n = 2^24` sessions of one table and `K̄ = 2^42` gates;
* `firstHitBound_tables32`: at most `2^-100` at `T = 2^80`, `n = 5523986` sessions of `32` tables and
  `K̄ = 227711235758` gates, the most that keeps it (`firstHitBound_tables32_tight`);
* `real_le_firstHitBound`, `real_le_two_rpow_neg_100`, `real_table1_le`, `real_tables32_le`: the same expression over
  the reals, at any reals within a design's numbers, with `2^256` and `2^-100` real powers as the finder's
  statements write them.
-/

@[expose] public section

namespace Claims.Flock.Numbers

/-! ## A session's `k` at one table and at 32 -/

/-- **`k ≤ 12343986` at a session of one table**: `⌈4N₀/e⌉` at `N₀ ≤ 2^23`. -/
theorem k_le_of_one_table {N : ℝ} (hN : N ≤ 2 ^ 23) : ⌈4 * N / Real.exp 1⌉₊ ≤ 12343986 := by
  have he0 : 0 < Real.exp 1 := Real.exp_pos 1
  have hlo := Real.exp_one_gt_d9
  rw [Nat.ceil_le, div_le_iff₀ he0]
  push_cast
  linarith

/-- **`k ≤ 395007543` at a session of at most `32` tables**: `⌈4N₀/e⌉` at `N₀ ≤ 32·2^23`. -/
theorem k_le_of_tables_le_32 {N : ℝ} (hN : N ≤ 32 * 2 ^ 23) : ⌈4 * N / Real.exp 1⌉₊ ≤ 395007543 := by
  have he0 : 0 < Real.exp 1 := Real.exp_pos 1
  have hlo := Real.exp_one_gt_d9
  rw [Nat.ceil_le, div_le_iff₀ he0]
  push_cast
  linarith

/-! ## The bound over the rationals -/

/-- `e = 1 + m′·k`: the expected reruns of one session's extraction, at `k` accepted reruns a wait. -/
def sessionReruns (k : ℕ) : ℚ := 1 + (waits : ℚ) * (k : ℚ)

/-- **The by-position first-hit finder's binding term**: `4·T·(1 + n·e + (1 + e)·K̄)/2^256` at `e = 1 + m′·k`. -/
def firstHitBound (T n k : ℕ) (Kmean : ℚ) : ℚ :=
  4 * (T : ℚ) * (1 + (n : ℚ) * sessionReruns k + (1 + sessionReruns k) * Kmean) / 2 ^ 256

theorem sessionReruns_nonneg (k : ℕ) : 0 ≤ sessionReruns k := by
  unfold sessionReruns
  positivity

theorem sessionReruns_le {k k' : ℕ} (hk : k ≤ k') : sessionReruns k ≤ sessionReruns k' := by
  have hk' : (k : ℚ) ≤ k' := by exact_mod_cast hk
  have hw : (0 : ℚ) ≤ (waits : ℚ) := by positivity
  unfold sessionReruns
  linarith [mul_le_mul_of_nonneg_left hk' hw]

/-- **The bound is monotone**: it rises with `T`, `n`, `k` and `K̄`. -/
theorem firstHitBound_le {T T' n n' k k' : ℕ} {K K' : ℚ} (hT : T ≤ T') (hn : n ≤ n') (hk : k ≤ k') (hK0 : 0 ≤ K)
    (hK : K ≤ K') : firstHitBound T n k K ≤ firstHitBound T' n' k' K' := by
  have hT' : (T : ℚ) ≤ T' := by exact_mod_cast hT
  have hn' : (n : ℚ) ≤ n' := by exact_mod_cast hn
  have hn0 : (0 : ℚ) ≤ n' := by positivity
  have he0 := sessionReruns_nonneg k
  have he := sessionReruns_le hk
  have h1 : (n : ℚ) * sessionReruns k ≤ (n' : ℚ) * sessionReruns k' := mul_le_mul hn' he he0 hn0
  have h2 : (1 + sessionReruns k) * K ≤ (1 + sessionReruns k') * K' :=
    mul_le_mul (by linarith) hK hK0 (by linarith)
  have h3 : 0 ≤ (n : ℚ) * sessionReruns k := mul_nonneg (by positivity) he0
  have h4 : 0 ≤ (1 + sessionReruns k) * K := mul_nonneg (by linarith) hK0
  have h5 : 4 * (T : ℚ) * (1 + (n : ℚ) * sessionReruns k + (1 + sessionReruns k) * K) ≤
      4 * (T' : ℚ) * (1 + (n' : ℚ) * sessionReruns k' + (1 + sessionReruns k') * K') :=
    mul_le_mul (by linarith) (by linarith) (by linarith) (by linarith)
  unfold firstHitBound
  exact div_le_div_of_nonneg_right h5 (by positivity)

/-! ## Sessions of one table -/

/-- **At most `2^-100` at sessions of one table**: a prover of `2^80` compressions, `2^24` sessions, `k = 12343986`
and a sample needing `2^42` gates in the mean. -/
theorem firstHitBound_table1 : firstHitBound (2 ^ 80) (2 ^ 24) 12343986 (2 ^ 42) ≤ 1 / 2 ^ 100 := by
  norm_num [firstHitBound, sessionReruns, waits]

/-- **At most `2^-100` at every statement within those numbers.** -/
theorem firstHitBound_le_of_table1 {T n k : ℕ} {K : ℚ} (hT : T ≤ 2 ^ 80) (hn : n ≤ 2 ^ 24) (hk : k ≤ 12343986)
    (hK0 : 0 ≤ K) (hK : K ≤ 2 ^ 42) : firstHitBound T n k K ≤ 1 / 2 ^ 100 :=
  (firstHitBound_le hT hn hk hK0 hK).trans firstHitBound_table1

/-! ## Sessions of at most 32 tables -/

/-- **At most `2^-100` at sessions of `32` tables**: a prover of `2^80` compressions, `5523986` sessions,
`k = 395007543` and a sample needing `227711235758` gates in the mean. -/
theorem firstHitBound_tables32 : firstHitBound (2 ^ 80) 5523986 395007543 227711235758 ≤ 1 / 2 ^ 100 := by
  norm_num [firstHitBound, sessionReruns, waits]

/-- One gate more in the mean and the bound is above `2^-100`. -/
theorem firstHitBound_tables32_tight : 1 / 2 ^ 100 < firstHitBound (2 ^ 80) 5523986 395007543 227711235759 := by
  norm_num [firstHitBound, sessionReruns, waits]

/-- **At most `2^-100` at every statement within those numbers.** -/
theorem firstHitBound_le_of_tables32 {T n k : ℕ} {K : ℚ} (hT : T ≤ 2 ^ 80) (hn : n ≤ 5523986) (hk : k ≤ 395007543)
    (hK0 : 0 ≤ K) (hK : K ≤ 227711235758) : firstHitBound T n k K ≤ 1 / 2 ^ 100 :=
  (firstHitBound_le hT hn hk hK0 hK).trans firstHitBound_tables32

/-! ## Over the reals -/

/-- **The bound over the reals**, `2^256` a real power. -/
theorem firstHitBound_cast (T n k : ℕ) (K : ℚ) :
    ((firstHitBound T n k K : ℚ) : ℝ) =
      4 * (T : ℝ) * (1 + (n : ℝ) * (1 + 210 * (k : ℝ)) + (1 + (1 + 210 * (k : ℝ))) * (K : ℝ)) / 2 ^ (256 : ℝ) := by
  rw [show (256 : ℝ) = ((256 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
  simp only [firstHitBound, sessionReruns, waits]
  push_cast
  ring

/-- **The expression at reals within a design's numbers is at most the bound at them**: a run of `T ≤ T'`
compressions, `n ≤ n'` sessions, each extracted in `e ≤ 1 + 210·k'` expected reruns, and a mean of `K ≤ K'` gates. -/
theorem real_le_firstHitBound {T n e K : ℝ} {T' n' k' : ℕ} {K' : ℚ} (hn0 : 0 ≤ n) (he0 : 0 ≤ e) (hK0 : 0 ≤ K)
    (hT : T ≤ T') (hn : n ≤ n') (he : e ≤ 1 + 210 * (k' : ℝ)) (hK : K ≤ K') :
    4 * T * (1 + n * e + (1 + e) * K) / 2 ^ (256 : ℝ) ≤ ((firstHitBound T' n' k' K' : ℚ) : ℝ) := by
  have hn'0 : (0 : ℝ) ≤ n' := by positivity
  have h1 : n * e ≤ (n' : ℝ) * (1 + 210 * (k' : ℝ)) := mul_le_mul hn he he0 hn'0
  have h2 : (1 + e) * K ≤ (1 + (1 + 210 * (k' : ℝ))) * (K' : ℝ) := mul_le_mul (by linarith) hK hK0 (by linarith)
  have h3 : 0 ≤ n * e := mul_nonneg hn0 he0
  have h4 : 0 ≤ (1 + e) * K := mul_nonneg (by linarith) hK0
  have h5 : 4 * T * (1 + n * e + (1 + e) * K) ≤
      4 * (T' : ℝ) * (1 + (n' : ℝ) * (1 + 210 * (k' : ℝ)) + (1 + (1 + 210 * (k' : ℝ))) * (K' : ℝ)) :=
    mul_le_mul (by linarith) (by linarith) (by linarith) (by linarith)
  rw [firstHitBound_cast]
  exact div_le_div_of_nonneg_right h5 (by positivity)

/-- `1/2^100` over the rationals is `2^-100`, a real power. -/
theorem cast_two_pow_neg_100 : ((1 / 2 ^ 100 : ℚ) : ℝ) = 2 ^ (-(100 : ℝ)) := by
  rw [Real.rpow_neg (by norm_num), show (100 : ℝ) = ((100 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
  push_cast
  ring

/-- **The expression over the reals is at most `2^-100`** wherever the rational bound at numbers above it is. -/
theorem real_le_two_rpow_neg_100 {T n e K : ℝ} {T' n' k' : ℕ} {K' : ℚ} (hn0 : 0 ≤ n) (he0 : 0 ≤ e) (hK0 : 0 ≤ K)
    (hT : T ≤ T') (hn : n ≤ n') (he : e ≤ 1 + 210 * (k' : ℝ)) (hK : K ≤ K')
    (h : firstHitBound T' n' k' K' ≤ 1 / 2 ^ 100) :
    4 * T * (1 + n * e + (1 + e) * K) / 2 ^ (256 : ℝ) ≤ 2 ^ (-(100 : ℝ)) := by
  rw [← cast_two_pow_neg_100]
  exact (real_le_firstHitBound hn0 he0 hK0 hT hn he hK).trans (Rat.cast_le.2 h)

/-- **The binding term is at most `2^-100` at sessions of one table**: at a run of `T ≤ 2^80` compressions,
`n ≤ 2^24` sessions, each extracted in `e ≤ 1 + 210·12343986` expected reruns, and a sample needing `K ≤ 2^42` gates
in the mean. -/
theorem real_table1_le {T n e K : ℝ} (hn0 : 0 ≤ n) (he0 : 0 ≤ e) (hK0 : 0 ≤ K) (hT : T ≤ 2 ^ 80) (hn : n ≤ 2 ^ 24)
    (he : e ≤ 1 + 210 * 12343986) (hK : K ≤ 2 ^ 42) :
    4 * T * (1 + n * e + (1 + e) * K) / 2 ^ (256 : ℝ) ≤ 2 ^ (-(100 : ℝ)) := by
  refine real_le_two_rpow_neg_100 (T' := 2 ^ 80) (n' := 2 ^ 24) (k' := 12343986) (K' := 2 ^ 42) hn0 he0 hK0
    ?_ ?_ ?_ ?_ firstHitBound_table1
  · rw [Nat.cast_pow, Nat.cast_ofNat]
    exact hT
  · rw [Nat.cast_pow, Nat.cast_ofNat]
    exact hn
  · push_cast
    exact he
  · rw [Rat.cast_pow, Rat.cast_ofNat]
    exact hK

/-- **The binding term is at most `2^-100` at sessions of at most `32` tables**: at a run of `T ≤ 2^80`
compressions, `n ≤ 5523986` sessions, each extracted in `e ≤ 1 + 210·395007543` expected reruns, and a sample
needing `K ≤ 227711235758` gates in the mean. -/
theorem real_tables32_le {T n e K : ℝ} (hn0 : 0 ≤ n) (he0 : 0 ≤ e) (hK0 : 0 ≤ K) (hT : T ≤ 2 ^ 80)
    (hn : n ≤ 5523986) (he : e ≤ 1 + 210 * 395007543) (hK : K ≤ 227711235758) :
    4 * T * (1 + n * e + (1 + e) * K) / 2 ^ (256 : ℝ) ≤ 2 ^ (-(100 : ℝ)) := by
  refine real_le_two_rpow_neg_100 (T' := 2 ^ 80) (n' := 5523986) (k' := 395007543) (K' := 227711235758) hn0 he0 hK0
    ?_ ?_ ?_ ?_ firstHitBound_tables32
  · rw [Nat.cast_pow, Nat.cast_ofNat]
    exact hT
  · push_cast
    exact hn
  · push_cast
    exact he
  · push_cast
    exact hK

end Claims.Flock.Numbers
