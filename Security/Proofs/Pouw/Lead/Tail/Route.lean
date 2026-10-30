import Mathlib.Analysis.SpecialFunctions.Stirling
import Mathlib.Analysis.Complex.ExponentialBounds
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

theorem cut_le_div {size r : ℕ} {l : ℝ} (hl : 0 ≤ l) : cut size l r ≤ l ^ r / r.factorial := by
  unfold cut
  split_ifs
  · positivity
  · exact le_rfl

/-- **A Poisson term against a geometric bound**: `λ^r / r! ≤ q^{r₀}` once `e λ ≤ q r`, `q ≤ 1` and `r ≥ r₀`. -/
theorem cut_le_pow {size r r₀ : ℕ} {l q : ℝ} (hl : 0 ≤ l) (hq : 0 ≤ q) (hq1 : q ≤ 1) (hr₀ : 0 < r₀)
    (hr : r₀ ≤ r) (h : 2.7182818286 * l ≤ q * r) : cut size l r ≤ q ^ r₀ := by
  have hr0 : 0 < r := lt_of_lt_of_le hr₀ hr
  have hr' : (0 : ℝ) < r := by exact_mod_cast hr0
  have he : Real.exp 1 * l / r ≤ q := by
    rw [div_le_iff₀ hr']
    nlinarith [mul_le_mul_of_nonneg_right Real.exp_one_lt_d9.le hl]
  calc cut size l r ≤ l ^ r / r.factorial := cut_le_div hl
    _ ≤ (Real.exp 1 * l / r) ^ r := pow_div_factorial_le hl hr0
    _ ≤ q ^ r := pow_le_pow_left₀ (by positivity) he r
    _ ≤ q ^ r₀ := pow_le_pow_of_le_one hq hq1 hr

/-- **The three points' shape**: four row blocks a coin tile, whole coin tiles, at least 512 column blocks and 147
steps; and the tail's least repeats exceed `4082 / 1,470,000` of the blocks (the slack alone: `tiles · colBlocks ·
K / 10^4`, with `147 K ≥ 4082 S`). -/
theorem three_facts (T m n K : ℕ) (hm : m = 64 ∨ m = 512 ∨ m = 8192) (hn : 4096 ≤ n) (hK : 4096 ≤ K) :
    rho (lead T m n K) = 4 ∧ pairs (lead T m n K) = 6 ∧
      (lead T m n K).rowBlocks = 4 * (lead T m n K).tiles ∧ 512 ≤ (lead T m n K).colBlocks ∧
      147 ≤ (lead T m n K).S ∧ 1 ≤ (lead T m n K).tiles ∧
      4082 * ((lead T m n K).colBlocks * ((lead T m n K).S * (lead T m n K).tiles)) <
        1470000 * tailCount (lead T m n K) := by
  have hshape : (lead T m n K).m = 64 * (lead T m n K).tiles ∧
      (lead T m n K).rowBlocks = 4 * (lead T m n K).tiles ∧ 1 ≤ (lead T m n K).tiles := by
    rcases hm with rfl | rfl | rfl <;> simp only [lead, padRows, Layout.tiles, Layout.rowBlocks] <;> norm_num
  obtain ⟨hmt, hrb, ht⟩ := hshape
  have hρ : rho (lead T m n K) = 4 := by unfold rho; rw [hrb]; omega
  have hLn : (lead T m n K).n = 8 * (lead T m n K).colBlocks := by
    simp only [Layout.colBlocks, lead, padCols]; omega
  have hC : 512 ≤ (lead T m n K).colBlocks := by simp only [Layout.colBlocks, lead, padCols]; omega
  have hS : 147 ≤ (lead T m n K).S := by simp only [lead, stepsFor]; omega
  have hKS : 4082 * (lead T m n K).S ≤ 147 * (lead T m n K).K := by simp only [lead, stepsFor]; omega
  refine ⟨hρ, by unfold pairs; rw [hρ], hrb, hC, hS, ht, ?_⟩
  set L := lead T m n K
  have h1 := lt_tailCount L
  have hmean : 0 ≤ coinBranchMean L := by unfold coinBranchMean tileWords; positivity
  have hslack : coinSlack L = ((L.tiles * L.colBlocks * L.K : ℕ) : ℚ) * 128 / 10000 := by
    unfold coinSlack stockRegs
    rw [hmt, hLn]
    push_cast
    ring
  have hq : ((4082 * (L.colBlocks * (L.S * L.tiles)) : ℕ) : ℚ) < ((1470000 * tailCount L : ℕ) : ℚ) := by
    have hKq : ((4082 * L.S : ℕ) : ℚ) ≤ ((147 * L.K : ℕ) : ℚ) := by exact_mod_cast hKS
    have hCt : (0 : ℚ) ≤ L.colBlocks * L.tiles := by positivity
    have h2 := mul_le_mul_of_nonneg_left hKq hCt
    rw [hslack, div_lt_iff₀ (by unfold tileWords; norm_num)] at h1
    unfold tileWords at h1
    push_cast at h1 h2 ⊢
    nlinarith
  exact_mod_cast hq

/-- **Decode's shape**: one row block and coin tile, at least 512 column blocks and 147 steps; and the tail's least
repeats exceed `4082 / 5,880,000` of the blocks. -/
theorem decode_facts (T n K : ℕ) (hn : 4096 ≤ n) (hK : 4096 ≤ K) :
    rho (lead T 1 n K) = 1 ∧ pairs (lead T 1 n K) = 0 ∧ (lead T 1 n K).rowBlocks = 1 ∧ (lead T 1 n K).tiles = 1 ∧
      512 ≤ (lead T 1 n K).colBlocks ∧ 147 ≤ (lead T 1 n K).S ∧
      4082 * ((lead T 1 n K).colBlocks * (lead T 1 n K).S) < 5880000 * tailCount (lead T 1 n K) := by
  have hm : (lead T 1 n K).m = 16 := by simp only [lead, padRows]
  have hrb : (lead T 1 n K).rowBlocks = 1 := by simp only [Layout.rowBlocks, hm]
  have ht : (lead T 1 n K).tiles = 1 := by simp only [Layout.tiles, hm]; simp only [lead]
  have hρ : rho (lead T 1 n K) = 1 := by unfold rho; rw [hrb]; rfl
  have hLn : (lead T 1 n K).n = 8 * (lead T 1 n K).colBlocks := by
    simp only [Layout.colBlocks, lead, padCols]; omega
  have hC : 512 ≤ (lead T 1 n K).colBlocks := by simp only [Layout.colBlocks, lead, padCols]; omega
  have hS : 147 ≤ (lead T 1 n K).S := by simp only [lead, stepsFor]; omega
  have hKS : 4082 * (lead T 1 n K).S ≤ 147 * (lead T 1 n K).K := by simp only [lead, stepsFor]; omega
  refine ⟨hρ, by unfold pairs; rw [hρ], hrb, ht, hC, hS, ?_⟩
  set L := lead T 1 n K
  have h1 := lt_tailCount L
  have hmean : 0 ≤ coinBranchMean L := by unfold coinBranchMean tileWords; positivity
  have hslack : coinSlack L = ((L.colBlocks * L.K : ℕ) : ℚ) * 32 / 10000 := by
    unfold coinSlack stockRegs
    rw [hm, hLn]
    push_cast
    ring
  have hq : ((4082 * (L.colBlocks * L.S) : ℕ) : ℚ) < ((5880000 * tailCount L : ℕ) : ℚ) := by
    have hKq : ((4082 * L.S : ℕ) : ℚ) ≤ ((147 * L.K : ℕ) : ℚ) := by exact_mod_cast hKS
    have hC0 : (0 : ℚ) ≤ L.colBlocks := by positivity
    have h2 := mul_le_mul_of_nonneg_left hKq hC0
    rw [hslack, div_lt_iff₀ (by unfold tileWords; norm_num)] at h1
    unfold tileWords at h1
    push_cast at h1 h2 ⊢
    nlinarith
  exact_mod_cast hq

/-- **Decode has a route**: `a = R`, and the other counts 1, where `pairs = 0` leaves the column blocks' and slots'
terms 0. -/
theorem route_decode (T n K : ℕ) (hn : 4096 ≤ n) (hK : 4096 ≤ K)
    (htiles : (lead T 1 n K).rowBlocks * (lead T 1 n K).colBlocks * (lead T 1 n K).S ≤ 2 ^ 30) :
    Route (lead T 1 n K) := by
  obtain ⟨hρ, hp, hrb, ht, hC, hS, hR⟩ := decode_facts T n K hn hK
  set L := lead T 1 n K
  rw [hrb, one_mul] at htiles
  have hU : 512 * 147 ≤ L.colBlocks * L.S := Nat.mul_le_mul hC hS
  have hR5 : 5 ≤ tailCount L := by omega
  refine ⟨tailCount L, 1, 1, 1, by rw [hρ]; omega, by norm_num, ?_⟩
  have hG : lamG L = 0 := by simp [lamG, hp]
  have hB : lamB L = 0 := by simp [lamB, hp]
  have h0 : ∀ size, cut size 0 1 = 0 := fun size => by unfold cut; split_ifs <;> simp
  rw [hG, hB, h0, h0, mul_zero, add_zero, add_zero]
  have hX : cut (blkCount L) (lamX L) (tailCount L) ≤ (1 / 1024) ^ 5 := by
    refine cut_le_pow (by unfold lamX; positivity) (by norm_num) (by norm_num) (by norm_num) hR5 ?_
    have hB : blkCount L = L.colBlocks * L.S := by unfold blkCount; rw [ht]; ring
    have hU2 : ((L.colBlocks * L.S : ℕ) : ℝ) ^ 2 ≤ 2 ^ 30 * ((L.colBlocks * L.S : ℕ) : ℝ) := by
      rw [sq]
      exact mul_le_mul_of_nonneg_right (by exact_mod_cast htiles) (Nat.cast_nonneg _)
    have hRr : (4082 : ℝ) * ((L.colBlocks * L.S : ℕ) : ℝ) ≤ 5880000 * (tailCount L : ℝ) := by
      exact_mod_cast hR.le
    unfold lamX
    rw [hρ, hB]
    push_cast at hU2 hRr ⊢
    nlinarith
  calc cut (blkCount L) (lamX L) (tailCount L) ≤ (1 / 1024) ^ 5 := hX
    _ ≤ 1 / 2 ^ 42 := by norm_num

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
