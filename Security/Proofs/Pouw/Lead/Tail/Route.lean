import Mathlib.Analysis.SpecialFunctions.Stirling
import Proofs.Pouw.Lead.Tail.Blocks

/-!
# The coin-branch tail's route at the three points and decode

A **route** for a call (`Route`) is four counts with `ρ (a − 1) + (ρ − 1)(w − 1)` below the tail's least repeats
(`tailCount`) and `(k − 1)(x − 1)` below `w`, whose three terms `λ_X^a/a! + λ_G^k/k! + colBlocks · λ_b^x/x!`, each
`cut` at its count's size, sum to at most `2^−42`. `Tail.coinTail_le` turns a route into the tail's bound.
`route_le`: each of the three points and decode, for `n, K ≥ 4,096` and at most `2^30` tiles, has one. Write
`R = tailCount` and `s = S · tiles` (a column block's slots); `a = 1 + ⌊R/32⌋` and the largest `w` the first
constraint allows serve throughout.

* **Decode** (`route_decode`): one row block, so `ρ = 1`, `pairs = 0` and only `λ_X^R / R!` is left.
* **Many slots** (`route_wide`, `s ≥ 2^16`): `k = colBlocks + 1`, where `cut` drops the column blocks' term, and
  `x = ⌈w / colBlocks⌉`, at least `8.7 λ_b`.
* **Fewer slots** (`route_mid`, `2^9 ≤ s < 2^16`; `route_corner`, `s < 2^9`, only at T 512 × 64): `x` the least
  count at which `colBlocks · (e λ_b / x)^x ≤ 2^−44` at the top of the dyadic bands holding `s` and `colBlocks`
  (above `2^9`, of `s`'s band alone), and `k = 1 + ⌊(w − 1)/(x − 1)⌋`, where `e λ_G / k` is at most a constant
  times `x`, so the column blocks' term falls with `s · colBlocks`.

`pow_div_factorial_le` (`λ^r / r! ≤ (e λ / r)^r`, from `Stirling.le_factorial_stirling`) and `lt_tailCount` are the
pieces each regime reads.
-/

namespace Pouw.SecurityProofs.Lead.Tail

open Pouw.Lead

/-- **A route for a call**: counts `a`, `k`, `w`, `x` that `Tail.coinTail_le`'s split admits, whose terms sum to at
most `2^−42`. -/
def Route (L : Pouw.Lead.Layout) : Prop :=
  ∃ a k w x : ℕ, rho L * (a - 1) + (rho L - 1) * (w - 1) < tailCount L ∧ (k - 1) * (x - 1) < w ∧
    cut (blkCount L) (lamX L) a + cut L.colBlocks (lamG L) k + L.colBlocks * cut (L.S * L.tiles) (lamB L) x ≤
      1 / 2 ^ 42

/-- **A Poisson term against its Stirling bound**: `λ^r / r! ≤ (e λ / r)^r`. -/
theorem pow_div_factorial_le {l : ℝ} (hl : 0 ≤ l) {r : ℕ} (hr : 0 < r) :
    l ^ r / r.factorial ≤ (Real.exp 1 * l / r) ^ r := by
  have hr' : (0 : ℝ) < r := by exact_mod_cast hr
  have hs : 1 ≤ √(2 * Real.pi * r) := by
    rw [Real.one_le_sqrt]
    nlinarith [Real.two_le_pi, (Nat.one_le_cast (α := ℝ)).2 hr]
  have hf : (r / Real.exp 1) ^ r ≤ (r.factorial : ℝ) :=
    (le_mul_of_one_le_left (by positivity) hs).trans (Stirling.le_factorial_stirling r)
  calc l ^ r / r.factorial ≤ l ^ r / (r / Real.exp 1) ^ r :=
        div_le_div_of_nonneg_left (by positivity) (by positivity) hf
    _ = (Real.exp 1 * l / r) ^ r := by
        rw [← div_pow]
        congr 1
        field_simp

/-- **The tail's least repeats exceed the mean and slack in tiles.** -/
theorem lt_tailCount (L : Pouw.Lead.Layout) : (coinBranchMean L + coinSlack L) / tileWords < tailCount L := by
  unfold tailCount
  push_cast
  exact Nat.lt_floor_add_one _

/-- **Decode has a route.** -/
theorem route_decode (T n K : ℕ) (hn : 4096 ≤ n) (hK : 4096 ≤ K)
    (htiles : (lead T 1 n K).rowBlocks * (lead T 1 n K).colBlocks * (lead T 1 n K).S ≤ 2 ^ 30) :
    Route (lead T 1 n K) := by
  sorry

/-- **The three points have a route at `2^16` slots a column block or more.** -/
theorem route_wide (T m n K : ℕ) (hm : m = 64 ∨ m = 512 ∨ m = 8192) (hn : 4096 ≤ n) (hK : 4096 ≤ K)
    (htiles : (lead T m n K).rowBlocks * (lead T m n K).colBlocks * (lead T m n K).S ≤ 2 ^ 30)
    (hs : 2 ^ 16 ≤ (lead T m n K).S * (lead T m n K).tiles) :
    Route (lead T m n K) := by
  sorry

/-- **The three points have a route from `2^9` to `2^16` slots a column block.** -/
theorem route_mid (T m n K : ℕ) (hm : m = 64 ∨ m = 512 ∨ m = 8192) (hn : 4096 ≤ n) (hK : 4096 ≤ K)
    (htiles : (lead T m n K).rowBlocks * (lead T m n K).colBlocks * (lead T m n K).S ≤ 2 ^ 30)
    (hs : 2 ^ 9 ≤ (lead T m n K).S * (lead T m n K).tiles) (hs' : (lead T m n K).S * (lead T m n K).tiles < 2 ^ 16) :
    Route (lead T m n K) := by
  sorry

/-- **T 512 × 64 has a route below `2^9` slots a column block**: its smallest calls, where the route is tightest. -/
theorem route_corner (T n K : ℕ) (hn : 4096 ≤ n) (hK : 4096 ≤ K)
    (htiles : (lead T 64 n K).rowBlocks * (lead T 64 n K).colBlocks * (lead T 64 n K).S ≤ 2 ^ 30)
    (hs : (lead T 64 n K).S * (lead T 64 n K).tiles < 2 ^ 9) :
    Route (lead T 64 n K) := by
  sorry

/-- **The three points and decode have a route**, for `n, K ≥ 4,096` and at most `2^30` tiles. -/
theorem route_le (T n K : ℕ) (hn : 4096 ≤ n) (hK : 4096 ≤ K) :
    ∀ L ∈ [pt512x64 n K, pt512x512 n K, pt8192 n K, lead T 1 n K],
      L.rowBlocks * L.colBlocks * L.S ≤ 2 ^ 30 → Route L := by
  intro L hL htiles
  have hS : 147 ≤ stepsFor K := by unfold stepsFor; omega
  have hpt : ∀ T' m, (m = 64 ∨ m = 512 ∨ m = 8192) →
      (lead T' m n K).rowBlocks * (lead T' m n K).colBlocks * (lead T' m n K).S ≤ 2 ^ 30 → Route (lead T' m n K) := by
    intro T' m hm ht
    by_cases hw : 2 ^ 16 ≤ (lead T' m n K).S * (lead T' m n K).tiles
    · exact route_wide T' m n K hm hn hK ht hw
    by_cases hmid : 2 ^ 9 ≤ (lead T' m n K).S * (lead T' m n K).tiles
    · exact route_mid T' m n K hm hn hK ht hmid (not_le.1 hw)
    obtain rfl : m = 64 := by
      rcases hm with rfl | rfl | rfl
      · rfl
      all_goals
        refine absurd ?_ hmid
        simp only [lead, Layout.tiles, padRows]
        omega
    exact route_corner T' n K hn hK ht (not_le.1 hmid)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hL
  rcases hL with rfl | rfl | rfl | rfl
  · exact hpt 512 64 (Or.inl rfl) htiles
  · exact hpt 512 512 (Or.inr (Or.inl rfl)) htiles
  · exact hpt 8192 8192 (Or.inr (Or.inr rfl)) htiles
  · exact route_decode T n K hn hK htiles

end Pouw.SecurityProofs.Lead.Tail
