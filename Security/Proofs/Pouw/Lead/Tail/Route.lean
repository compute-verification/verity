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
  `x = 1 + ⌊(w − 1)/colBlocks⌋`, where `e λ_b / x ≤ 1/3`.
* **Fewer slots** (`route_mid`, `2^9 ≤ s < 2^16`; `route_corner`, `s < 2^9`, only at T 512 × 64): `route_band`,
  `x = y + 1` and `k = 1 + ⌊(w − 1)/y⌋` with `e λ_G / k ≤ q`, on bands of `s` (`route_mid`: `2^12`, `2^16`) and of
  `P = colBlocks · s` (`route_corner`: `2^17`, `2^18`, `2^19`, `2^28`), where the column blocks' term is at most
  `P · 6/2^16 · (6 s/2^16)^y / x!` at the band's top and `q^k` at its bottom.

`across_term_le` bounds the first term by `2^−70` throughout; `cut_le_pow` (`λ^r / r! ≤ q^{r₀}` once `e λ ≤ q r`, from
`pow_div_factorial_le` and `Stirling.le_factorial_stirling`) and `lt_tailCount` are the pieces each regime reads.
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

/-- **The three points' first term**: at `a = 1 + ⌊R/32⌋`, `λ_X^a / a! ≤ 2^−70`, since `e λ_X / a ≤ 1/1024` at
`2^30` tiles or fewer and `a ≥ 7`. -/
theorem across_term_le (L : Pouw.Lead.Layout) (hρ : rho L = 4) (hC : 512 ≤ L.colBlocks) (hs : 147 ≤ L.S * L.tiles)
    (hU : 4 * (L.colBlocks * (L.S * L.tiles)) ≤ 2 ^ 30)
    (hR : 4082 * (L.colBlocks * (L.S * L.tiles)) < 1470000 * tailCount L) :
    cut (blkCount L) (lamX L) (1 + tailCount L / 32) ≤ (1 / 1024) ^ 7 := by
  have hP : 512 * 147 ≤ L.colBlocks * (L.S * L.tiles) := Nat.mul_le_mul hC hs
  have ha : 7 ≤ 1 + tailCount L / 32 := by omega
  have hPa : 4082 * (L.colBlocks * (L.S * L.tiles)) < 47040000 * (1 + tailCount L / 32) := by omega
  refine cut_le_pow (by unfold lamX; positivity) (by norm_num) (by norm_num) (by norm_num) ha ?_
  have hB : blkCount L = L.colBlocks * (L.S * L.tiles) := by unfold blkCount; ring
  have hU2 : ((L.colBlocks * (L.S * L.tiles) : ℕ) : ℝ) ^ 2 ≤ 2 ^ 28 * ((L.colBlocks * (L.S * L.tiles) : ℕ) : ℝ) := by
    rw [sq]
    exact mul_le_mul_of_nonneg_right (by exact_mod_cast (show L.colBlocks * (L.S * L.tiles) ≤ 2 ^ 28 by omega))
      (Nat.cast_nonneg _)
  have hPa' : (4082 : ℝ) * ((L.colBlocks * (L.S * L.tiles) : ℕ) : ℝ) ≤
      47040000 * ((1 + tailCount L / 32 : ℕ) : ℝ) := by exact_mod_cast hPa.le
  unfold lamX
  rw [hρ, hB]
  push_cast at hU2 hPa' ⊢
  nlinarith

/-- **A route at the three points within one band**: a column block's slots `s ≤ s₁` and the blocks
`P₀ ≤ colBlocks · s ≤ P₁`, with `x = y + 1` and `k = 1 + ⌊(w − 1)/y⌋`. `q` bounds `e λ_G / k` (`hkq`), `k₀` is the
least `k` (`hk`), and the three terms are at most `2^−70`, `q^{k₀}` and `P₁ · 6/2^16 · (6 s₁/2^16)^y / x!`. -/
theorem route_band (L : Pouw.Lead.Layout) (hρ : rho L = 4) (hp : pairs L = 6) (hC : 512 ≤ L.colBlocks)
    (hs : 147 ≤ L.S * L.tiles) (hU : 4 * (L.colBlocks * (L.S * L.tiles)) ≤ 2 ^ 30)
    (hR : 4082 * (L.colBlocks * (L.S * L.tiles)) < 1470000 * tailCount L)
    (y k₀ P₀ P₁ s₁ : ℕ) (q : ℝ) (hy : 0 < y) (hk₀ : 0 < k₀) (hq : 0 ≤ q) (hq1 : q ≤ 1)
    (hP₀ : P₀ ≤ L.colBlocks * (L.S * L.tiles)) (hP₁ : L.colBlocks * (L.S * L.tiles) ≤ P₁)
    (hs₁ : L.S * L.tiles ≤ s₁) (hkq : 2.7182818286 * 6 * 35280000 * y ≤ q * 28574 * 7 ^ 8)
    (hk : 35280000 * y * (k₀ - 1) ≤ 28574 * P₀)
    (hsum : (1 / 1024 : ℝ) ^ 7 + q ^ k₀ + P₁ * (6 / 2 ^ 16) * (6 * s₁ / 2 ^ 16) ^ y / (y + 1).factorial ≤
      1 / 2 ^ 42) :
    Route L := by
  have hX := across_term_le L hρ hC hs hU hR
  have hlamG : lamG L = 6 * ((L.colBlocks * (L.S * L.tiles) : ℕ) : ℝ) / 7 ^ 8 := by
    unfold lamG; rw [hp]; push_cast; ring
  have hlamB : lamB L = 6 * ((L.S * L.tiles : ℕ) : ℝ) / 2 ^ 16 := by unfold lamB; rw [hp]; push_cast; ring
  set C := L.colBlocks
  set s := L.S * L.tiles
  set R := tailCount L
  have hP : 512 * 147 ≤ C * s := Nat.mul_le_mul hC hs
  have hR2 : 200 ≤ R := by omega
  set w := 1 + (R - 1 - 4 * (R / 32)) / 3 with hw
  have hw7 : 7 * R ≤ 24 * w := by omega
  set k := 1 + (w - 1) / y with hk_def
  have hyk : w ≤ y * k := by
    have h1 := Nat.div_add_mod (w - 1) y
    have h2 := Nat.mod_lt (w - 1) hy
    have h3 : y * k = y + y * ((w - 1) / y) := by rw [hk_def, mul_add, mul_one]
    omega
  have hkP : 28574 * (C * s) < 35280000 * (y * k) := by omega
  have hk₀k : k₀ ≤ k := by
    have : 35280000 * y * (k₀ - 1) < 35280000 * y * k :=
      calc 35280000 * y * (k₀ - 1) ≤ 28574 * P₀ := hk
        _ ≤ 28574 * (C * s) := Nat.mul_le_mul_left _ hP₀
        _ < 35280000 * (y * k) := hkP
        _ = 35280000 * y * k := by ring
    have := Nat.lt_of_mul_lt_mul_left this
    omega
  refine ⟨1 + R / 32, k, w, y + 1, ?_, ?_, ?_⟩
  · rw [hρ]; omega
  · rw [Nat.add_sub_cancel]
    calc (k - 1) * y = (w - 1) / y * y := by rw [hk_def, Nat.add_sub_cancel_left]
      _ ≤ w - 1 := Nat.div_mul_le_self _ _
      _ < w := by omega
  · have hG : cut C (lamG L) k ≤ q ^ k₀ := by
      refine cut_le_pow (by unfold lamG; positivity) hq hq1 hk₀ hk₀k ?_
      have hkP' : (28574 : ℝ) * ((C * s : ℕ) : ℝ) < 35280000 * ((y : ℝ) * k) := by exact_mod_cast hkP
      rw [hlamG]
      have A := mul_le_mul_of_nonneg_right hkq (Nat.cast_nonneg (C * s))
      have B := mul_le_mul_of_nonneg_left hkP'.le (show (0 : ℝ) ≤ q * 7 ^ 8 by positivity)
      have key : 35280000 * (y : ℝ) * (2.7182818286 * (6 * ((C * s : ℕ) : ℝ) / 7 ^ 8)) ≤
          35280000 * (y : ℝ) * (q * k) := by linarith only [A, B]
      exact le_of_mul_le_mul_left key (by positivity)
    have hB : (C : ℝ) * cut s (lamB L) (y + 1) ≤
        P₁ * (6 / 2 ^ 16) * (6 * s₁ / 2 ^ 16) ^ y / (y + 1).factorial := by
      calc (C : ℝ) * cut s (lamB L) (y + 1) ≤ C * (lamB L ^ (y + 1) / (y + 1).factorial) :=
            mul_le_mul_of_nonneg_left (cut_le_div (by rw [hlamB]; positivity)) (Nat.cast_nonneg _)
        _ = ((C * s : ℕ) : ℝ) * (6 / 2 ^ 16) * (6 * s / 2 ^ 16) ^ y / (y + 1).factorial := by
            rw [hlamB, pow_succ]; push_cast; ring
        _ ≤ P₁ * (6 / 2 ^ 16) * (6 * s₁ / 2 ^ 16) ^ y / (y + 1).factorial := by
            gcongr
    linarith

/-- **A route at the three points from `2^16` slots a column block**: `k = colBlocks + 1`, where `cut` drops the
column blocks' term, and `x = 1 + ⌊(w − 1)/colBlocks⌋`, with `e λ_b / x ≤ 1/3`, `x ≥ 54` and `colBlocks ≤ 2^12`. -/
theorem route_wide_of (L : Pouw.Lead.Layout) (hρ : rho L = 4) (hp : pairs L = 6) (hC : 512 ≤ L.colBlocks)
    (hs : 2 ^ 16 ≤ L.S * L.tiles) (hU : 4 * (L.colBlocks * (L.S * L.tiles)) ≤ 2 ^ 30)
    (hR : 4082 * (L.colBlocks * (L.S * L.tiles)) < 1470000 * tailCount L) : Route L := by
  have hX := across_term_le L hρ hC (le_trans (by norm_num) hs) hU hR
  have hlamB : lamB L = 6 * ((L.S * L.tiles : ℕ) : ℝ) / 2 ^ 16 := by unfold lamB; rw [hp]; push_cast; ring
  set C := L.colBlocks
  set s := L.S * L.tiles
  set R := tailCount L
  have hP : 512 * 147 ≤ C * s := Nat.mul_le_mul hC (le_trans (by norm_num) hs)
  have hR2 : 200 ≤ R := by omega
  set w := 1 + (R - 1 - 4 * (R / 32)) / 3 with hw
  have hw7 : 7 * R ≤ 24 * w := by omega
  have hC0 : 0 < C := by omega
  set x := 1 + (w - 1) / C with hx_def
  have hCx : w ≤ C * x := by
    have h1 := Nat.div_add_mod (w - 1) C
    have h2 := Nat.mod_lt (w - 1) hC0
    have h3 : C * x = C + C * ((w - 1) / C) := by rw [hx_def, mul_add, mul_one]
    omega
  have hxs : 28574 * s < 35280000 * x := by
    have h : 28574 * (C * s) < 35280000 * (C * x) := by omega
    refine Nat.lt_of_mul_lt_mul_left (a := C) ?_
    calc C * (28574 * s) = 28574 * (C * s) := by ring
      _ < 35280000 * (C * x) := h
      _ = C * (35280000 * x) := by ring
  have hx54 : 54 ≤ x := by omega
  have hC12 : C ≤ 2 ^ 12 := by
    have : C * 2 ^ 16 ≤ C * s := Nat.mul_le_mul_left _ hs
    omega
  refine ⟨1 + R / 32, C + 1, w, x, ?_, ?_, ?_⟩
  · rw [hρ]; omega
  · rw [Nat.add_sub_cancel, hx_def, Nat.add_sub_cancel_left]
    calc C * ((w - 1) / C) ≤ w - 1 := Nat.mul_div_le _ _
      _ < w := by omega
  · have hG : cut C (lamG L) (C + 1) = 0 := by unfold cut; rw [ite_eq_left (by omega)]
    have hB : cut s (lamB L) x ≤ (1 / 3) ^ 54 := by
      refine cut_le_pow (by unfold lamB; positivity) (by norm_num) (by norm_num) (by norm_num) hx54 ?_
      have hxs' : (28574 : ℝ) * s ≤ 35280000 * x := by exact_mod_cast hxs.le
      rw [hlamB]
      linarith only [hxs']
    have hCr : (C : ℝ) ≤ 2 ^ 12 := by exact_mod_cast hC12
    have hB' : (C : ℝ) * cut s (lamB L) x ≤ 2 ^ 12 * (1 / 3) ^ 54 :=
      calc (C : ℝ) * cut s (lamB L) x ≤ C * (1 / 3) ^ 54 := mul_le_mul_of_nonneg_left hB (Nat.cast_nonneg _)
        _ ≤ 2 ^ 12 * (1 / 3) ^ 54 := mul_le_mul_of_nonneg_right hCr (by positivity)
    rw [hG]
    have : (1 / 1024 : ℝ) ^ 7 + 0 + 2 ^ 12 * (1 / 3) ^ 54 ≤ 1 / 2 ^ 42 := by norm_num
    linarith

theorem three_tiles_le (T m n K : ℕ) (hrb : (lead T m n K).rowBlocks = 4 * (lead T m n K).tiles)
    (htiles : (lead T m n K).rowBlocks * (lead T m n K).colBlocks * (lead T m n K).S ≤ 2 ^ 30) :
    4 * ((lead T m n K).colBlocks * ((lead T m n K).S * (lead T m n K).tiles)) ≤ 2 ^ 30 := by
  rw [hrb] at htiles
  calc 4 * ((lead T m n K).colBlocks * ((lead T m n K).S * (lead T m n K).tiles)) =
        4 * (lead T m n K).tiles * (lead T m n K).colBlocks * (lead T m n K).S := by ring
    _ ≤ 2 ^ 30 := htiles

/-- **The three points have a route at `2^16` slots a column block or more.** -/
theorem route_wide (T m n K : ℕ) (hm : m = 64 ∨ m = 512 ∨ m = 8192) (hn : 4096 ≤ n) (hK : 4096 ≤ K)
    (htiles : (lead T m n K).rowBlocks * (lead T m n K).colBlocks * (lead T m n K).S ≤ 2 ^ 30)
    (hs : 2 ^ 16 ≤ (lead T m n K).S * (lead T m n K).tiles) :
    Route (lead T m n K) := by
  obtain ⟨hρ, hp, hrb, hC, -, -, hR⟩ := three_facts T m n K hm hn hK
  exact route_wide_of _ hρ hp hC hs (three_tiles_le T m n K hrb htiles) hR

/-- **The three points have a route from `2^9` to `2^16` slots a column block.** -/
theorem route_mid (T m n K : ℕ) (hm : m = 64 ∨ m = 512 ∨ m = 8192) (hn : 4096 ≤ n) (hK : 4096 ≤ K)
    (htiles : (lead T m n K).rowBlocks * (lead T m n K).colBlocks * (lead T m n K).S ≤ 2 ^ 30)
    (hs : 2 ^ 9 ≤ (lead T m n K).S * (lead T m n K).tiles) (hs' : (lead T m n K).S * (lead T m n K).tiles < 2 ^ 16) :
    Route (lead T m n K) := by
  obtain ⟨hρ, hp, hrb, hC, -, -, hR⟩ := three_facts T m n K hm hn hK
  have hU := three_tiles_le T m n K hrb htiles
  set L := lead T m n K
  by_cases h12 : L.S * L.tiles < 2 ^ 12
  · have hP₀ : 2 ^ 18 ≤ L.colBlocks * (L.S * L.tiles) :=
      calc 2 ^ 18 = 512 * 2 ^ 9 := by norm_num
        _ ≤ L.colBlocks * (L.S * L.tiles) := Nat.mul_le_mul hC hs
    exact route_band L hρ hp hC (by omega) hU hR 16 14 (2 ^ 18) (2 ^ 28) (2 ^ 12) (56 / 1000) (by norm_num)
      (by norm_num) (by norm_num) (by norm_num) hP₀ (by omega) h12.le (by norm_num) (by norm_num) (by norm_num)
  · have hP₀ : 2 ^ 21 ≤ L.colBlocks * (L.S * L.tiles) :=
      calc 2 ^ 21 = 512 * 2 ^ 12 := by norm_num
        _ ≤ L.colBlocks * (L.S * L.tiles) := Nat.mul_le_mul hC (not_lt.1 h12)
    exact route_band L hρ hp hC (by omega) hU hR 52 33 (2 ^ 21) (2 ^ 28) (2 ^ 16) (19 / 100) (by norm_num)
      (by norm_num) (by norm_num) (by norm_num) hP₀ (by omega) hs'.le (by norm_num) (by norm_num) (by norm_num)

/-- **T 512 × 64 has a route below `2^9` slots a column block**: its smallest calls, where the route is tightest. -/
theorem route_corner (T n K : ℕ) (hn : 4096 ≤ n) (hK : 4096 ≤ K)
    (htiles : (lead T 64 n K).rowBlocks * (lead T 64 n K).colBlocks * (lead T 64 n K).S ≤ 2 ^ 30)
    (hs : (lead T 64 n K).S * (lead T 64 n K).tiles < 2 ^ 9) :
    Route (lead T 64 n K) := by
  obtain ⟨hρ, hp, hrb, hC, hS, ht, hR⟩ := three_facts T 64 n K (Or.inl rfl) hn hK
  have hU := three_tiles_le T 64 n K hrb htiles
  set L := lead T 64 n K
  have hs147 : 147 ≤ L.S * L.tiles := le_trans hS (Nat.le_mul_of_pos_right _ ht)
  have hP : 512 * 147 ≤ L.colBlocks * (L.S * L.tiles) := Nat.mul_le_mul hC hs147
  rcases lt_or_ge (L.colBlocks * (L.S * L.tiles)) (2 ^ 17) with h17 | h17
  · have hs255 : L.S * L.tiles ≤ 255 := by
      by_contra h
      have : 512 * 256 ≤ L.colBlocks * (L.S * L.tiles) := Nat.mul_le_mul hC (by omega)
      omega
    exact route_band L hρ hp hC hs147 hU hR 7 9 (512 * 147) (2 ^ 17) 255 (245 / 10000) (by norm_num)
      (by norm_num) (by norm_num) (by norm_num) hP h17.le hs255 (by norm_num) (by norm_num) (by norm_num)
  rcases lt_or_ge (L.colBlocks * (L.S * L.tiles)) (2 ^ 18) with h18 | h18
  · exact route_band L hρ hp hC hs147 hU hR 9 12 (2 ^ 17) (2 ^ 18) 511 (315 / 10000) (by norm_num)
      (by norm_num) (by norm_num) (by norm_num) h17 h18.le (by omega) (by norm_num) (by norm_num) (by norm_num)
  rcases lt_or_ge (L.colBlocks * (L.S * L.tiles)) (2 ^ 19) with h19 | h19
  · exact route_band L hρ hp hC hs147 hU hR 12 18 (2 ^ 18) (2 ^ 19) 511 (42 / 1000) (by norm_num)
      (by norm_num) (by norm_num) (by norm_num) h18 h19.le (by omega) (by norm_num) (by norm_num) (by norm_num)
  · exact route_band L hρ hp hC hs147 hU hR 18 24 (2 ^ 19) (2 ^ 28) 511 (63 / 1000) (by norm_num)
      (by norm_num) (by norm_num) (by norm_num) h19 (by omega) (by omega) (by norm_num) (by norm_num) (by norm_num)

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
