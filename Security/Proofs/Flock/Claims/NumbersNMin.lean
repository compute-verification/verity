import Proofs.Flock.Claims.DecodeFlock
import Proofs.Flock.Claims.NumbersGamma

/-!
# `nMin` at the deployed parameters

The list-decoding spec once asked a decoder to list a trace only from at least `nMin` opened columns at distinct
positions (the spec's `nMin` form, since replaced by its radius form, `Flock.ListWithin`, for what follows), the
Johnson count `johnsonCount k agree = ⌊(k+1)/agree²⌋ + 1` of the trace's code (`Claims.johnsonCount`,
`listComplete_flockLevelZero`). Flock's first committed codeword is level 0 of its `fast100` schedule, of dimension
`k = 2^c` (`c = level₀.logCols`) and length `N₀ = 2^{c+1}` (rate `1/2`). At the analysis' threshold
`agree = 1 − δ = √(1/2) + 1/200` (`agree₀`; `δ = 1 − √ρ − η`, `Level.radius`):

* `nMin = johnsonCount (2^c) agree₀` exactly, at every `c` a `fast100` schedule has (`johnsonCount_agree₀`): from
  `1012` at `c = 9` (`m = 22`) to `8271224` at `c = 22` (`m = 35`) (`nMinVal`, `nMin_fast100`);
* that is `98.6%` to `98.8%` of the codeword: `nMin < N₀` (`nMin_lt_len`), so `nMin` distinct positions exist, but a
  run opens `218` level-0 columns (`277` at most under `--zk`), far fewer (`queries_lt_nMin`; `fast100_level₀_queries_lt`
  at any `agree ≤ 1`);
* `E^rep` gathers its `k = ⌈4N₀/e⌉` accepted reruns' columns, `218·k > N₀` of them (`len_lt_k_mul_queries`), so they
  never sit at distinct positions of the codeword, which `ListComplete` asks of its input;
* a set of the `N₀ − nMin + 1` positions that `nMin` distinct positions must meet is missed by `q ≤ 277` uniform
  queries with probability `((nMin − 1)/N₀)^q ≥ 1/64` (`miss_ge`), against `κ ≤ 2^{−202.98}`;
* under `--zk` the masked level-0 code has dimension `2^c + t` (`t = 2q₀`) and the same length, so at `agree₀` its
  Johnson count is past the length at `m = 25, 26, 27` (`zk_nMin_gt_len`): the `nMin` form's agreement was one
  number, while the masked rate `ρ' = (2^c + t)/2^{c+1}` runs from `0.568` (`m = 25`) to `0.500` (`m = 35`), which is
  why the radius form reads its radius per trace (`Deploy.θ`).
-/

namespace Claims.Flock.Numbers

open FlockSoundness.Accounting

/-- Level 0's decoding threshold: `1 − δ = √ρ + η` at rate `1/2` and `η = 1/200`. -/
noncomputable def agree₀ : ℝ := Real.sqrt (1 / 2) + 1 / 200

theorem sqrt_half_bounds : (0.7071067811 : ℝ) < Real.sqrt (1 / 2) ∧ Real.sqrt (1 / 2) < 0.7071067812 := by
  constructor
  · rw [Real.lt_sqrt (by norm_num)]
    norm_num
  · rw [Real.sqrt_lt' (by norm_num)]
    norm_num

theorem agree₀_pos : 0 < agree₀ := by
  unfold agree₀
  positivity

theorem agree₀_le_one : agree₀ ≤ 1 := by
  unfold agree₀
  linarith [sqrt_half_bounds.2]

theorem agree₀_sq_bounds : (0.507096067811 : ℝ) ≤ agree₀ ^ 2 ∧ agree₀ ^ 2 ≤ 0.507096067812 := by
  obtain ⟨h1, h2⟩ := sqrt_half_bounds
  have hs : Real.sqrt (1 / 2) ^ 2 = 1 / 2 := Real.sq_sqrt (by norm_num)
  have e : agree₀ ^ 2 = 1 / 2 + Real.sqrt (1 / 2) / 100 + 1 / 40000 := by
    unfold agree₀
    nlinarith
  rw [e]
  constructor <;> linarith

/-- `johnsonCount k a = J` from two-sided bounds on `a²`. -/
theorem johnsonCount_eq_of {k J : ℕ} {a lo hi : ℝ} (hJ : 1 ≤ J) (hlo0 : 0 < lo) (hlo : lo ≤ a ^ 2)
    (hhi : a ^ 2 ≤ hi) (h1 : ((J : ℝ) - 1) * hi ≤ k + 1) (h2 : (k : ℝ) + 1 < J * lo) : johnsonCount k a = J := by
  unfold johnsonCount
  have ha2 : 0 < a ^ 2 := lt_of_lt_of_le hlo0 hlo
  have hJ1 : (1 : ℝ) ≤ J := by exact_mod_cast hJ
  have hfl : ⌊((k : ℝ) + 1) / a ^ 2⌋₊ = J - 1 := by
    rw [Nat.floor_eq_iff (by positivity), Nat.cast_sub hJ, Nat.cast_one, le_div_iff₀ ha2, div_lt_iff₀ ha2]
    constructor
    · nlinarith
    · simp only [sub_add_cancel]
      nlinarith
  omega

/-- `nMin` at level 0 of `2^c` message columns, at every `c` a `fast100` schedule has. -/
def nMinVal : ℕ → ℕ
  | 9 => 1012
  | 10 => 2022
  | 11 => 4041
  | 12 => 8080
  | 13 => 16157
  | 14 => 32312
  | 17 => 258478
  | 18 => 516954
  | 19 => 1033905
  | 20 => 2067808
  | 21 => 4135613
  | 22 => 8271224
  | _ => 0

/-- **`nMin`, exactly**: `johnsonCount (2^c) agree₀ = nMinVal c` at every level-0 width of a `fast100` schedule. -/
theorem johnsonCount_agree₀ {c : ℕ} (hc : c ∈ [9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22]) :
    johnsonCount (2 ^ c) agree₀ = nMinVal c := by
  obtain ⟨hlo, hhi⟩ := agree₀_sq_bounds
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hc
  rcases hc with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    exact johnsonCount_eq_of (by simp [nMinVal]) (by norm_num) hlo hhi (by norm_num [nMinVal])
      (by norm_num [nMinVal])

theorem fast100_logCols₀ {m : ℕ} {sch : Schedule} (h : fast100 m = some sch) :
    (sch.levels.headD default).logCols ∈ [9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22] ∧
      (sch.levels.headD default).logLen = (sch.levels.headD default).logCols + 1 := by
  have hm : 22 ≤ m ∧ m ≤ 35 := by
    by_contra hc
    have : fast100 m = none := by unfold fast100; split <;> simp_all
    simp [this] at h
  obtain ⟨h1, h2⟩ := hm
  interval_cases m <;> (simp only [fast100, Option.some.injEq] at h; subst h; decide)

/-- **`nMin` at every `fast100` schedule**, at level 0's own width. -/
theorem nMin_fast100 {m : ℕ} {sch : Schedule} (h : fast100 m = some sch) :
    johnsonCount (2 ^ (sch.levels.headD default).logCols) agree₀ = nMinVal (sch.levels.headD default).logCols :=
  johnsonCount_agree₀ (fast100_logCols₀ h).1

/-- `nMin` is below the codeword's length `2^{c+1}`: the threshold leaves `N₀ − nMin + 1` positions out. -/
theorem nMin_lt_len {c : ℕ} (hc : c ∈ [9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22]) :
    nMinVal c < 2 ^ (c + 1) ∧ 2 ^ (c + 1) - nMinVal c + 1 ≥ 2 ^ (c + 1) / 80 := by
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hc
  rcases hc with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> decide

/-- A run's level-0 queries, `277` at most (`--zk` at `m = 25`), are fewer than `nMin`. -/
theorem queries_lt_nMin {c q : ℕ} (hc : c ∈ [9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22]) (hq : q ≤ 277) :
    q < nMinVal c := by
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hc
  rcases hc with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> simp [nMinVal] <;> omega

/-- `E^rep`'s gathered level-0 columns outnumber the codeword's positions: `N₀ < 218·k`, with `k = ⌈4N₀/e⌉`. -/
theorem len_lt_k_mul_queries {d : ℕ} (hd : d ∈ [10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23]) :
    2 ^ d < 218 * kRep d := by
  rw [kRep_eq hd]
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hd
  rcases hd with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> decide

/-- **The positions `nMin` must meet are missed with probability at least `1/64`**: `q ≤ 277` uniform level-0 queries
avoid a set of `N₀ − nMin + 1` positions with probability `((nMin − 1)/N₀)^q ≥ 1/64`. -/
theorem miss_ge {c q : ℕ} (hc : c ∈ [9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22]) (hq : q ≤ 277) :
    (1 / 64 : ℚ) ≤ (((nMinVal c - 1 : ℕ) : ℚ) / 2 ^ (c + 1)) ^ q := by
  have hle : ∀ c, c ∈ [9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22] →
      (1 / 64 : ℚ) ≤ (((nMinVal c - 1 : ℕ) : ℚ) / 2 ^ (c + 1)) ^ 277 := by
    intro c hc
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hc
    rcases hc with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> decide +kernel
  refine (hle c hc).trans (pow_le_pow_of_le_one (by positivity) ?_ hq)
  rw [div_le_one (by positivity)]
  have := (nMin_lt_len hc).1
  exact_mod_cast (Nat.sub_le _ _).trans this.le

/-- **Under `--zk`, `agree₀` puts `nMin` past the codeword at `m = 25, 26, 27`**: the masked level-0 code, of
dimension `2^c + t` (`t = 2q₀`, `m1Pad`) and length `2^{c+1}`, has its Johnson count at `agree₀` above its length
(`9172 > 2^13`, `17112 > 2^14`, `33215 > 2^15`), so no columns at distinct positions reach it. -/
theorem zk_nMin_gt_len {m : ℕ} (hm : m ∈ [25, 26, 27]) :
    2 ^ (((m1Schedule m).levels.headD default).logCols + 1) <
      johnsonCount (2 ^ ((m1Schedule m).levels.headD default).logCols + m1Pad m) agree₀ := by
  obtain ⟨hlo, hhi⟩ := agree₀_sq_bounds
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hm
  rcases hm with rfl | rfl | rfl
  · rw [show ((m1Schedule 25).levels.headD default).logCols = 12 by decide, show m1Pad 25 = 554 by decide,
      johnsonCount_eq_of (J := 9172) (by norm_num) (by norm_num) hlo hhi (by norm_num) (by norm_num)]
    norm_num
  · rw [show ((m1Schedule 26).levels.headD default).logCols = 13 by decide, show m1Pad 26 = 484 by decide,
      johnsonCount_eq_of (J := 17112) (by norm_num) (by norm_num) hlo hhi (by norm_num) (by norm_num)]
    norm_num
  · rw [show ((m1Schedule 27).levels.headD default).logCols = 14 by decide, show m1Pad 27 = 458 by decide,
      johnsonCount_eq_of (J := 33215) (by norm_num) (by norm_num) hlo hhi (by norm_num) (by norm_num)]
    norm_num

end Claims.Flock.Numbers
