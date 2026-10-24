import Definitions.Pouw.H1Zy.Stream
import Proofs.Pouw.H1Zy.Lane
import Mathlib.Data.Int.ModEq

/-!
# `PouwH1Tile_v3`'s z is the EXACT z

`Pouw.H1Zy.streamZ`, a cell's z on words as `PouwH1Tile_v3` computes it (its lane's O' and D words recombined, its wrap
correction and its dither peel), is `GemmAccumulatorS8_v2`'s word on the cell's row's and column's codes, at a `K` the
tile binds at (`32 ∣ K`, `K < 2^17`) and for every noise (`streamZ_eq_gemmAccS8`). So whatever holds of
`PouwH1Tile_v1`'s z (its EXACT, `gemmAccS8`) holds of v3's, the claim `h1tile.stream_cell`'s docstring makes:

* mod `2^32`, every fold is its accumulator plus its step's sum (`foldl_range_modEq`), so the z is
  `T - Σ_ρ U_ρ Σ_d 128^d A_d[ρ] + 256 Σ_t X_t (wb⁺_t - wb⁻_t) - Σ_t (c_t X_t + bd_t B_t - 256 xa_t B_t)`, with
  `-U` exact since `u129` is within `[-64, 64]` (`sx8_neg_u129`);
* the digits recombine to the residue sums (`digits_val`): each digit's step keeps `Σ_d 128^d A_d + 128^n R`, and the
  last `R` is an int8, at the digit count `n_digits` gives a `K` below `2^17` (`nDigits_cases`, `Y` within
  `[-128 KS, 127 KS]`);
* entry by entry, `X W' - X U + 256 X (wb⁺ - wb⁻) - (c X + bd B - 256 xa B) = A B` (`entry_eq`), from `X = A + bd -
  256 xa` and `W' = B + c + U - 256 (wb⁺ - wb⁻)` as int8 values.
-/

namespace Pouw.H1Zy

/-! ### Words and sums -/

theorem sx8_bounds (v : ℕ) : -128 ≤ sx8 v ∧ sx8 v ≤ 127 := by
  unfold sx8; split <;> omega

theorem sx8_u129_bounds (x : ℕ) : -64 ≤ sx8 (u129 x) ∧ sx8 (u129 x) ≤ 64 := by
  unfold sx8 u129; split <;> omega

theorem sx8_neg_u129 (x : ℕ) : sx8 ((256 - u129 x % 256) % 256) = -sx8 (u129 x) := by
  unfold sx8 u129; split <;> split <;> omega

theorem sx8_toNat (b : BitVec 8) : sx8 b.toNat = b.toInt := by
  have h := b.isLt
  rw [BitVec.toInt_eq_toNat_cond]
  unfold sx8
  rw [Nat.mod_eq_of_lt h]
  split <;> split <;> omega

theorem sum_map_range (n : ℕ) (f : ℕ → ℤ) : ((List.range n).map f).sum = ∑ i ∈ Finset.range n, f i := by
  induction n with
  | zero => simp
  | succ n ih => rw [List.range_succ, List.map_append, List.sum_append, ih, Finset.sum_range_succ]; simp

theorem sum_range_mul32 (f : ℕ → ℤ) : ∀ ks : ℕ,
    ∑ t ∈ Finset.range (32 * ks), f t = ∑ s ∈ Finset.range ks, ∑ e ∈ Finset.range 32, f (32 * s + e)
  | 0 => by simp
  | ks + 1 => by rw [show 32 * (ks + 1) = 32 * ks + 32 by ring, Finset.sum_range_add, sum_range_mul32 f ks,
      Finset.sum_range_succ (fun s => ∑ e ∈ Finset.range 32, f (32 * s + e)) ks]

theorem cast_toNat_emod (x : ℤ) : (((x % 2 ^ 32).toNat : ℕ) : ℤ) = x % 2 ^ 32 :=
  Int.toNat_of_nonneg (Int.emod_nonneg _ (by norm_num))

theorem toNat_emod_modEq (x : ℤ) : (((x % 2 ^ 32).toNat : ℕ) : ℤ) ≡ x [ZMOD 2 ^ 32] := by
  rw [cast_toNat_emod]; exact Int.mod_modEq x _

/-- A fold over `0 .. n - 1` whose step adds `g s` mod `2^32` adds their sum. -/
theorem foldl_range_modEq (F : ℕ → ℕ → ℕ) (g : ℕ → ℤ) (init : ℕ)
    (hF : ∀ acc s, (F acc s : ℤ) ≡ acc + g s [ZMOD 2 ^ 32]) :
    ∀ n, (((List.range n).foldl F init : ℕ) : ℤ) ≡ init + ∑ s ∈ Finset.range n, g s [ZMOD 2 ^ 32]
  | 0 => by simp
  | n + 1 => by
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, Finset.sum_range_succ, ← add_assoc]
      exact (hF _ n).trans ((foldl_range_modEq F g init hF n).add_right _)

theorem getD_segOf {α : Type} (f : ℕ → α) (s e : ℕ) (d : α) (he : e < 32) : (segOf f s).getD e d = f (32 * s + e) := by
  simp [segOf, List.getD_eq_getElem?_getD, he]

theorem length_segOf {α : Type} (f : ℕ → α) (s : ℕ) : (segOf f s).length = 32 := by simp [segOf]

theorem segDot_modEq (acc : ℕ) (f g : ℕ → ℕ) (n : ℕ) :
    (segDot acc ((List.range n).map f) ((List.range n).map g) : ℤ) ≡
      acc + ∑ e ∈ Finset.range n, sx8 (f e) * sx8 (g e) [ZMOD 2 ^ 32] := by
  unfold segDot
  rw [List.zipWith_map, List.zipWith_self, sum_map_range]
  exact toNat_emod_modEq _

theorem segDot_segOf_modEq (acc : ℕ) (f g : ℕ → ℕ) (s : ℕ) :
    (segDot acc (segOf f s) (segOf g s) : ℤ) ≡
      acc + ∑ e ∈ Finset.range 32, sx8 (f (32 * s + e)) * sx8 (g (32 * s + e)) [ZMOD 2 ^ 32] :=
  segDot_modEq acc (fun e => f (32 * s + e)) (fun e => g (32 * s + e)) 32

theorem sum_int_modEq {ι : Type} (S : Finset ι) (f g : ι → ℤ) (h : ∀ i ∈ S, f i ≡ g i [ZMOD 2 ^ 32]) :
    ∑ i ∈ S, f i ≡ ∑ i ∈ S, g i [ZMOD 2 ^ 32] := by
  unfold Int.ModEq
  rw [Finset.sum_int_mod S (2 ^ 32) f, Finset.sum_int_mod S (2 ^ 32) g]
  exact congrArg (· % 2 ^ 32) (Finset.sum_congr rfl h)

/-! ### One entry -/

/-- `X = A + bd` as an int8: `A + bd - 256 xa`. -/
theorem sx8_x (a : ℕ) (bd : Bool) :
    sx8 ((a + if bd then 1 else 0) % 256) = sx8 a + (if bd then 1 else 0) - 256 * (if xWrap a bd then 1 else 0) := by
  unfold sx8 xWrap
  cases bd <;> simp only [Bool.and_false, Bool.and_true, Bool.false_eq_true, ↓reduceIte, decide_eq_true_eq] <;>
    split_ifs <;> omega

/-- `W' = B + c + U` as an int8: `B + c + U - 256 (wb⁺ - wb⁻)`. -/
theorem sx8_w (w : ℕ) (c : Bool) (u : ℕ) :
    sx8 ((w + (if c then 1 else 0) + u) % 256) = sx8 w + (if c then 1 else 0) + sx8 u -
      256 * ((if (wrapBits w c u).1 then 1 else 0) - (if (wrapBits w c u).2 then 1 else 0)) := by
  unfold wrapBits
  simp only [decide_eq_true_eq]
  unfold sx8
  cases c <;> simp only [Bool.false_eq_true, ↓reduceIte] <;> split_ifs <;> omega

/-- **One entry of the cell**: its product in `T`, less its U product, plus its wrap correction, less its peel term,
is `A B`. -/
theorem entry_eq (a w : ℕ) (bd c : Bool) (u : ℕ) :
    sx8 ((a + if bd then 1 else 0) % 256) * sx8 ((w + (if c then 1 else 0) + u) % 256) -
        sx8 ((a + if bd then 1 else 0) % 256) * sx8 u +
        256 * (sx8 ((a + if bd then 1 else 0) % 256) *
          ((if (wrapBits w c u).1 then 1 else 0) - (if (wrapBits w c u).2 then 1 else 0))) -
        peelTerm ((a + if bd then 1 else 0) % 256) bd (xWrap a bd) w c =
      sx8 a * sx8 w := by
  unfold peelTerm
  rw [sx8_w, sx8_x]
  ring

/-! ### The digits -/

theorem nDigits_cases (ks : ℕ) (h : ks < 4096) :
    (ks ≤ 1 ∧ nDigits ks = 1) ∨ (2 ≤ ks ∧ ks ≤ 128 ∧ nDigits ks = 2) ∨ (129 ≤ ks ∧ nDigits ks = 3) := by
  unfold nDigits
  rcases Nat.lt_or_ge ks 2 with h1 | h1
  · left
    refine ⟨by omega, ?_⟩
    simp only [nDigitsFrom]
    rw [ite_eq_left (by constructor <;> omega)]
  · right
    obtain ⟨k, rfl⟩ : ∃ k, ks = k + 2 := ⟨ks - 2, by omega⟩
    simp only [nDigitsFrom]
    rw [ite_eq_right (by omega)]
    rcases Nat.lt_or_ge k 127 with h2 | h2
    · left
      refine ⟨by omega, by omega, ?_⟩
      rw [ite_eq_left (by norm_num; omega)]
    · right
      refine ⟨by omega, ?_⟩
      rw [ite_eq_right (by norm_num; omega)]
      obtain ⟨j, rfl⟩ : ∃ j, k = j + 1 := ⟨k - 1, by omega⟩
      simp only [nDigitsFrom]
      rw [ite_eq_left (by norm_num; omega)]

/-- **The digits recombine to the sum**, at a digit count `n_digits` gives with the sum in its range. -/
theorem digits_val (m : ℕ) (Y : ℤ) (z : List Bool)
    (h : (m = 1 ∧ -128 ≤ Y ∧ Y ≤ 127) ∨ (m = 2 ∧ -16384 ≤ Y ∧ Y ≤ 16256) ∨ (m = 3 ∧ -524288 ≤ Y ∧ Y ≤ 520065)) :
    ∑ d ∈ Finset.range m, 128 ^ d * sx8 ((digitsOf m Y z).getD d 0) = Y := by
  rcases h with ⟨rfl, h1, h2⟩ | ⟨rfl, h1, h2⟩ | ⟨rfl, h1, h2⟩
  · simp [digitsOf, sx8]
    split_ifs <;> omega
  · simp only [digitsOf, Nat.reduceSub, List.range_one, List.foldl_cons, List.foldl_nil, Finset.sum_range_succ,
      Finset.range_one, Finset.sum_singleton, pow_zero, one_mul, pow_one]
    cases z.getD 0 false <;> simp [sx8] <;> split_ifs <;> omega
  · simp only [digitsOf, Nat.reduceSub, Finset.sum_range_succ, Finset.range_one, Finset.sum_singleton, pow_zero,
      one_mul, pow_one, show List.range 2 = [0, 1] from rfl, List.foldl_cons, List.foldl_nil]
    cases z.getD 0 false <;> cases z.getD 1 false <;> simp [sx8] <;> split_ifs <;> omega

theorem sum_sx8_bounds (f : ℕ → ℕ) (n : ℕ) :
    -(128 * (n : ℤ)) ≤ ∑ s ∈ Finset.range n, sx8 (f s) ∧ ∑ s ∈ Finset.range n, sx8 (f s) ≤ 127 * n := by
  induction n with
  | zero => simp
  | succ n ih =>
      rw [Finset.sum_range_succ]
      have := sx8_bounds (f n)
      push_cast
      constructor <;> linarith [ih.1, ih.2]

/-! ### The segments' words -/

theorem wrapSeg_segOf_modEq (acc : ℕ) (x : ℕ → ℕ) (wb : ℕ → Bool × Bool) (s : ℕ) :
    (wrapSeg acc (segOf x s) (segOf wb s) : ℤ) ≡ acc + 256 * ∑ e ∈ Finset.range 32, sx8 (x (32 * s + e)) *
      ((if (wb (32 * s + e)).1 then 1 else 0) - (if (wb (32 * s + e)).2 then 1 else 0)) [ZMOD 2 ^ 32] := by
  unfold wrapSeg segOf
  rw [List.zipWith_map, List.zipWith_self, sum_map_range]
  exact toNat_emod_modEq _

theorem peelSeg_segOf_modEq (acc : ℕ) (x : ℕ → ℕ) (bd xa : ℕ → Bool) (w : ℕ → ℕ) (c : ℕ → Bool) (s : ℕ) :
    (peelSeg acc (segOf x s) (segOf bd s) (segOf xa s) (segOf w s) (segOf c s) : ℤ) ≡ acc -
      ∑ e ∈ Finset.range 32, peelTerm (x (32 * s + e)) (bd (32 * s + e)) (xa (32 * s + e)) (w (32 * s + e))
        (c (32 * s + e)) [ZMOD 2 ^ 32] := by
  unfold peelSeg
  rw [length_segOf, sum_map_range]
  have e : ∑ e ∈ Finset.range 32, peelTerm ((segOf x s).getD e 0) ((segOf bd s).getD e false)
      ((segOf xa s).getD e false) ((segOf w s).getD e 0) ((segOf c s).getD e false) =
      ∑ e ∈ Finset.range 32, peelTerm (x (32 * s + e)) (bd (32 * s + e)) (xa (32 * s + e)) (w (32 * s + e))
        (c (32 * s + e)) :=
    Finset.sum_congr rfl fun e he => by
      have he := Finset.mem_range.1 he
      rw [getD_segOf _ _ _ _ he, getD_segOf _ _ _ _ he, getD_segOf _ _ _ _ he, getD_segOf _ _ _ _ he,
        getD_segOf _ _ _ _ he]
  rw [e]
  exact toNat_emod_modEq _

theorem oneStream_modEq (op : ℕ) (m : ℕ) (f : ℕ → ℕ) :
    (oneStream op ((List.range m).map f) : ℤ) ≡ op - ∑ t ∈ Finset.range m, 128 ^ (t + 1) * (f t : ℤ)
      [ZMOD 2 ^ 32] := by
  unfold oneStream
  rw [List.length_map, List.length_range, sum_map_range]
  have e : ∑ t ∈ Finset.range m, (2 ^ (7 * (t + 1)) * ((((List.range m).map f).getD t 0 : ℕ) : ℤ)) =
      ∑ t ∈ Finset.range m, 128 ^ (t + 1) * (f t : ℤ) :=
    Finset.sum_congr rfl fun t ht => by
      rw [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range (Finset.mem_range.1 ht), pow_mul]
      norm_num
  rw [e]
  exact toNat_emod_modEq _

/-! ### The cell -/

/-- **`PouwH1Tile_v3`'s z of a cell is its exact dot mod `2^32`**, at a `K` the tile binds at, for every noise. -/
theorem streamZ_modEq (K : ℕ) (a : ℕ → ℕ) (bd : ℕ → Bool) (zc : ℕ → ℕ → Bool) (w : ℕ → ℕ) (c : ℕ → Bool)
    (ub : ℕ → ℕ) (hK32 : 32 ∣ K) (hK : K < 2 ^ 17) :
    (streamZ K a bd zc w c ub : ℤ) ≡ ∑ t ∈ Finset.range K, sx8 (a t) * sx8 (w t) [ZMOD 2 ^ 32] := by
  obtain ⟨ks, rfl⟩ := hK32
  have hks : ks < 4096 := by omega
  have hL : lay (32 * ks) = ⟨32 * ks, ks, nDigits ks, 32 * ks + 32 * (nDigits ks - 1), 32 * ks + 256,
      (32 * ks + 32 * (nDigits ks - 1) + 511) / 512, (32 * ks + 256 + 511) / 512⟩ := by
    simp only [lay, Nat.mul_div_cancel_left ks (by norm_num : 0 < 32)]
  set m := nDigits ks with hm
  let u : ℕ → ℕ := fun ρ => u129 (ub ρ)
  let X : ℕ → ℕ := fun t => (a t + if bd t then 1 else 0) % 256
  let W : ℕ → ℕ := fun t => (w t + (if c t then 1 else 0) + u (t % 32)) % 256
  let T := (List.range ks).foldl (fun acc s => segDot acc (segOf X s) (segOf W s)) 0
  let Y : ℕ → ℤ := fun ρ => ((List.range ks).map fun s => sx8 (X (32 * s + ρ))).sum
  let A : ℕ → List ℕ := fun ρ => digitsOf m (Y ρ) ((List.range (m - 1)).map fun d => zc d ρ)
  let digit : ℕ → List ℕ := fun d => (List.range 32).map fun ρ => (A ρ).getD d 0
  let op := segDot T ((List.range 32).map fun ρ => (256 - u ρ % 256) % 256) (digit 0)
  let z1 := oneStream op ((List.range (m - 1)).map fun d => segDot 0 ((List.range 32).map u) (digit (d + 1)))
  let wb : ℕ → Bool × Bool := fun t => wrapBits (w t) (c t) (u (t % 32))
  let xa : ℕ → Bool := fun t => xWrap (a t) (bd t)
  let z2 := (List.range ks).foldl (fun acc s => wrapSeg acc (segOf X s) (segOf wb s)) z1
  have e : streamZ (32 * ks) a bd zc w c ub =
      (List.range ks).foldl (fun acc s => peelSeg acc (segOf X s) (segOf bd s) (segOf xa s) (segOf w s) (segOf c s))
        z2 := by
    unfold streamZ
    rw [hL]
  rw [e]
  -- each fold mod 2^32
  have hT := foldl_range_modEq (fun acc s => segDot acc (segOf X s) (segOf W s))
    (fun s => ∑ e ∈ Finset.range 32, sx8 (X (32 * s + e)) * sx8 (W (32 * s + e))) 0
    (fun acc s => segDot_segOf_modEq acc X W s) ks
  have hz2 := foldl_range_modEq (fun acc s => wrapSeg acc (segOf X s) (segOf wb s))
    (fun s => 256 * ∑ e ∈ Finset.range 32, sx8 (X (32 * s + e)) *
      ((if (wb (32 * s + e)).1 then 1 else 0) - (if (wb (32 * s + e)).2 then 1 else 0))) z1
    (fun acc s => wrapSeg_segOf_modEq acc X wb s) ks
  have hz3 := foldl_range_modEq
    (fun acc s => peelSeg acc (segOf X s) (segOf bd s) (segOf xa s) (segOf w s) (segOf c s))
    (fun s => -∑ e ∈ Finset.range 32, peelTerm (X (32 * s + e)) (bd (32 * s + e)) (xa (32 * s + e))
      (w (32 * s + e)) (c (32 * s + e))) z2
    (fun acc s => by rw [← sub_eq_add_neg]; exact peelSeg_segOf_modEq acc X bd xa w c s) ks
  have hop : (op : ℤ) ≡ T + ∑ ρ ∈ Finset.range 32, -sx8 (u ρ) * sx8 ((A ρ).getD 0 0) [ZMOD 2 ^ 32] := by
    have h := segDot_modEq T (fun ρ => (256 - u ρ % 256) % 256) (fun ρ => (A ρ).getD 0 0) 32
    have e2 : ∑ ρ ∈ Finset.range 32, sx8 ((256 - u ρ % 256) % 256) * sx8 ((A ρ).getD 0 0) =
        ∑ ρ ∈ Finset.range 32, -sx8 (u ρ) * sx8 ((A ρ).getD 0 0) :=
      Finset.sum_congr rfl fun ρ _ => by rw [sx8_neg_u129]
    rw [e2] at h
    exact h
  have hz1 : (z1 : ℤ) ≡ op - ∑ t ∈ Finset.range (m - 1), 128 ^ (t + 1) *
      ∑ ρ ∈ Finset.range 32, sx8 (u ρ) * sx8 ((A ρ).getD (t + 1) 0) [ZMOD 2 ^ 32] := by
    refine (oneStream_modEq op (m - 1) _).trans (Int.ModEq.sub_left _ ?_)
    refine sum_int_modEq _ _ _ fun t _ => Int.ModEq.mul_left _ ?_
    have h := segDot_modEq 0 u (fun ρ => (A ρ).getD (t + 1) 0) 32
    rw [Nat.cast_zero, zero_add] at h
    exact h
  -- the digits
  have hY : ∀ ρ, ∑ d ∈ Finset.range m, 128 ^ d * sx8 ((A ρ).getD d 0) =
      ∑ s ∈ Finset.range ks, sx8 (X (32 * s + ρ)) := fun ρ => by
    have hy : Y ρ = ∑ s ∈ Finset.range ks, sx8 (X (32 * s + ρ)) := sum_map_range _ _
    rw [← hy]
    have hb := sum_sx8_bounds (fun s => X (32 * s + ρ)) ks
    rw [← hy] at hb
    apply digits_val
    rcases nDigits_cases ks hks with ⟨h1, h2⟩ | ⟨h1, h2, h3⟩ | ⟨h1, h2⟩
    · left
      refine ⟨by rw [hm, h2], ?_, ?_⟩
      · have : (ks : ℤ) ≤ 1 := by exact_mod_cast h1
        linarith [hb.1]
      · have : (ks : ℤ) ≤ 1 := by exact_mod_cast h1
        linarith [hb.2]
    · right; left
      have : (ks : ℤ) ≤ 128 := by exact_mod_cast h2
      exact ⟨by rw [hm, h3], by linarith [hb.1], by linarith [hb.2]⟩
    · right; right
      have : (ks : ℤ) ≤ 4095 := by exact_mod_cast (show ks ≤ 4095 by omega)
      exact ⟨by rw [hm, h2], by linarith [hb.1], by linarith [hb.2]⟩
  have hm1 : 1 ≤ m := by rcases nDigits_cases ks hks with ⟨_, h⟩ | ⟨_, _, h⟩ | ⟨_, h⟩ <;> omega
  -- the U terms are the X U products
  have hU : ∑ ρ ∈ Finset.range 32, -sx8 (u ρ) * sx8 ((A ρ).getD 0 0) - ∑ t ∈ Finset.range (m - 1), 128 ^ (t + 1) *
      ∑ ρ ∈ Finset.range 32, sx8 (u ρ) * sx8 ((A ρ).getD (t + 1) 0) =
      -∑ s ∈ Finset.range ks, ∑ e ∈ Finset.range 32, sx8 (X (32 * s + e)) * sx8 (u e) := by
    have e1 : ∑ t ∈ Finset.range (m - 1), 128 ^ (t + 1) *
        ∑ ρ ∈ Finset.range 32, sx8 (u ρ) * sx8 ((A ρ).getD (t + 1) 0) =
        ∑ ρ ∈ Finset.range 32, sx8 (u ρ) * ∑ t ∈ Finset.range (m - 1), 128 ^ (t + 1) * sx8 ((A ρ).getD (t + 1) 0) := by
      simp only [Finset.mul_sum]
      rw [Finset.sum_comm]
      exact Finset.sum_congr rfl fun _ _ => Finset.sum_congr rfl fun _ _ => by ring
    have e2 : ∀ ρ, ∑ d ∈ Finset.range m, 128 ^ d * sx8 ((A ρ).getD d 0) =
        ∑ t ∈ Finset.range (m - 1), 128 ^ (t + 1) * sx8 ((A ρ).getD (t + 1) 0) + sx8 ((A ρ).getD 0 0) := fun ρ => by
      conv_lhs => rw [show m = m - 1 + 1 by omega]
      rw [Finset.sum_range_succ']
      simp
    rw [e1, Finset.sum_comm, ← Finset.sum_neg_distrib, ← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun ρ _ => ?_
    rw [← Finset.sum_mul, show ∑ s ∈ Finset.range ks, sx8 (X (32 * s + ρ)) =
      ∑ d ∈ Finset.range m, 128 ^ d * sx8 ((A ρ).getD d 0) from (hY ρ).symm, e2 ρ]
    ring
  -- assemble
  have hz := hz3.trans ((hz2.add_right _).trans (((hz1.trans (hop.sub_right _)).add_right _).add_right _ |>.trans
    ((((hT.add_right _).sub_right _).add_right _).add_right _)))
  refine hz.trans (congrArg (· % (2 ^ 32 : ℤ)) ?_)
  have hent : ∀ s, ∀ e ∈ Finset.range 32, sx8 (a (32 * s + e)) * sx8 (w (32 * s + e)) =
      sx8 (X (32 * s + e)) * sx8 (W (32 * s + e)) - sx8 (X (32 * s + e)) * sx8 (u e) +
        256 * (sx8 (X (32 * s + e)) *
          ((if (wb (32 * s + e)).1 then 1 else 0) - (if (wb (32 * s + e)).2 then 1 else 0))) -
        peelTerm (X (32 * s + e)) (bd (32 * s + e)) (xa (32 * s + e)) (w (32 * s + e)) (c (32 * s + e)) :=
    fun s e he => by
      have h32 : (32 * s + e) % 32 = e := by have := Finset.mem_range.1 he; omega
      rw [← entry_eq (a (32 * s + e)) (w (32 * s + e)) (bd (32 * s + e)) (c (32 * s + e)) (u e)]
      simp only [X, W, wb, xa, h32]
  rw [Nat.cast_zero, zero_add, add_sub_assoc, hU, sum_range_mul32 _ ks,
    Finset.sum_congr rfl fun s _ => Finset.sum_congr rfl (hent s)]
  simp only [Finset.sum_sub_distrib, Finset.sum_add_distrib, Finset.mul_sum, Finset.sum_neg_distrib]
  ring

/-- **`PouwH1Tile_v3`'s z of a cell is `PouwH1Tile_v1`'s** (its EXACT, `GemmAccumulatorS8_v2`'s word) on the cell's
row's and column's codes, at a `K` the tile binds at, for every dither, digit coin, coin bit and noise. -/
theorem streamZ_eq_gemmAccS8 {K : ℕ} (x w : Fin K → BitVec 8) (bd : ℕ → Bool) (zc : ℕ → ℕ → Bool) (c : ℕ → Bool)
    (ub : ℕ → ℕ) (hK32 : 32 ∣ K) (hK : K < 2 ^ 17) :
    BitVec.ofNat 32 (streamZ K (fun t => (codeAt x t).toNat) bd zc (fun t => (codeAt w t).toNat) c ub) =
      gemmAccS8 x w := by
  have h := streamZ_modEq K (fun t => (codeAt x t).toNat) bd zc (fun t => (codeAt w t).toNat) c ub hK32 hK
  have hs : ∑ t ∈ Finset.range K, sx8 (codeAt x t).toNat * sx8 (codeAt w t).toNat =
      ∑ k, (x k).toInt * (w k).toInt := by
    rw [← Fin.sum_univ_eq_sum_range]
    refine Finset.sum_congr rfl fun k _ => ?_
    simp [sx8_toNat, codeAt]
  rw [hs] at h
  rw [gemmAccS8_eq x w hK32 hK]
  apply BitVec.eq_of_toNat_eq
  rw [BitVec.toNat_ofNat, BitVec.toNat_ofInt]
  unfold Int.ModEq at h
  omega

/-! ### At a lane -/

/-- **A v3 lane's cell z is v1's**: `laneZ3` is `GemmAccumulatorS8_v2`'s word on the cell's rows' codes, at a `K` the
tile binds at, for every salt, position and roots. -/
theorem laneZ3_eq {K : ℕ} (salt pos roots : ℕ) (l : Fin 32) (a : Fin 4 → Fin K → BitVec 8)
    (b : Fin 8 → Fin K → BitVec 8) (s : Fin 4) (t : Fin 8) (hK32 : 32 ∣ K) (hK : K < 2 ^ 17) :
    laneZ3 salt pos roots l a b s t = gemmAccS8 (a s) (b t) :=
  streamZ_eq_gemmAccS8 _ _ _ _ _ _ hK32 hK

/-- **A lane's `ok` at a tile whose z is EXACT's is `laneOk`**: so `PouwH1LaneZy_v1{PouwH1Tile_v3}`'s `ok`
(`laneOkZ` at `laneZ3`, by `laneZ3_eq`) is `PouwH1LaneZy_v1{PouwH1Tile_v1}`'s. -/
theorem laneOkZ_eq_laneOk {K nw : ℕ} {P : Type}
    (zf : P → Fin 32 → (Fin 4 → Fin K → BitVec 8) → (Fin 8 → Fin K → BitVec 8) → Fin 4 → Fin 8 → BitVec 32)
    (chain : P → Fin 32 → (Fin 4 → Fin K → BitVec 8) → (Fin 8 → Fin K → BitVec 8) → Fin nw → BitVec 32)
    (dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16) (x : LaneIn K nw P)
    (hz : ∀ p l a b s t, zf p l a b s t = gemmAccS8 (a s) (b t)) :
    laneOkZ zf chain dqW x = laneOk chain dqW x := by
  have : zf = fun _ _ a b s t => gemmAccS8 (a s) (b t) := by
    funext p l a b s t
    exact hz p l a b s t
  subst this
  rfl

end Pouw.H1Zy
