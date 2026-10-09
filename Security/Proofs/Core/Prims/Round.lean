import Definitions.Core.Prims
import Proofs.Core.Fp32.Round
import Proofs.Core.TC.Proofs.Norm

/-!
# `f32ToE4m3Sat` is the nearest E4M3 code, ties to even, saturating

`Verity.Prims.f32ToE4m3Sat` casts an f32 word to an E4M3 code. `dyadicLt` is the comparison of the dyadics it names
(`dyadicLt_units`) and `roundEvenDyadic` their rounding to nearest even (`roundEvenDyadic_spec`). Read through core's
`Verity.Fp32.decode`, in units of `2^-149`:

* a NaN casts to `E4M3_NAN` (`f32ToE4m3Sat_nan`), `±inf` to `±0x7E` (`f32ToE4m3Sat_inf`), and every magnitude from
  `464 · 2^149` (464, the tie between 448 and 480, included) up to `±0x7E` (`f32ToE4m3Sat_sat`);
* below it the code is `8a + q` under the sign, with `a + 140` the exponent `max(⌊log₂ m⌋, 143) - 3` of the E4M3 grid
  at `m` and `q` the magnitude `m / 2^(a + 140)` rounded to nearest even (`f32ToE4m3Sat_fin`);
* the code's value (`e4m3ToF32`, read through `decode`) is the nearest to the f32's of every code's, both zeros and the
  subnormals included, and when another code's value is as near the code is even (`f32ToE4m3Sat_nearest`).

Every code's value is `codeSig c · 2^codeExp c` units, `codeSig c < 16` and `codeExp c ≥ 140`, at most `448 · 2^149`, and
only `0x7F` and `0xFF` are NaN: the kernel checks all 256 codes (`codeChk_all`).
-/

namespace Verity.Prims.Round

open Verity.Prims Verity.Round Verity.Fp32.Round
open Verity.Fp32 (decode Decoded)

/-! ## The dyadics -/

/-- `dyadicLt` on exponents of at least `-149`, in units of `2^-149`. -/
theorem dyadicLt_units (a c k l : ℕ) :
    dyadicLt a ((k : ℤ) - 149) c ((l : ℤ) - 149) = true ↔ a * 2 ^ k < c * 2 ^ l := by
  have key : ∀ x y z : ℕ, a * 2 ^ x < c * 2 ^ y ↔ a * 2 ^ (x + z) < c * 2 ^ (y + z) := fun x y z => by
    rw [pow_add, pow_add, ← Nat.mul_assoc, ← Nat.mul_assoc]
    exact (Nat.mul_lt_mul_right (pow_pos (by norm_num) _)).symm
  simp only [dyadicLt, decide_eq_true_eq, Nat.shiftLeft_eq]
  refine (key _ _ (min k l)).trans ?_
  rw [show ((k : ℤ) - 149 - min ((k : ℤ) - 149) ((l : ℤ) - 149)).toNat + min k l = k by omega,
    show ((l : ℤ) - 149 - min ((k : ℤ) - 149) ((l : ℤ) - 149)).toNat + min k l = l by omega]

theorem dyadicLt_units0 (a c k : ℕ) : dyadicLt a ((k : ℤ) - 149) c 0 = true ↔ a * 2 ^ k < c * 2 ^ 149 := by
  rw [show (0 : ℤ) = ((149 : ℕ) : ℤ) - 149 by norm_num, dyadicLt_units]

theorem dyadicLt_units6 (a k : ℕ) : dyadicLt a ((k : ℤ) - 149) 1 (-6) = true ↔ a * 2 ^ k < 2 ^ 143 := by
  rw [show (-6 : ℤ) = ((143 : ℕ) : ℤ) - 149 by norm_num, dyadicLt_units, Nat.one_mul]

/-- **`roundEvenDyadic` rounds `m · 2^e` to nearest even.** -/
theorem roundEvenDyadic_spec (m : ℕ) (e : ℤ) : RoundsEven ((m : ℚ) * 2 ^ e) (roundEvenDyadic m e) := by
  by_cases he : 0 ≤ e
  · have hr : roundEvenDyadic m e = m * 2 ^ e.toNat := by rw [roundEvenDyadic, ite_eq_left he, Nat.shiftLeft_eq]
    have hv : (m : ℚ) * 2 ^ e = ((m * 2 ^ e.toNat : ℕ) : ℚ) := by
      push_cast; rw [← zpow_natCast, Int.toNat_of_nonneg he]
    rw [hr, hv]
    exact ⟨by simp, by simp⟩
  · have hr : roundEvenDyadic m e = if 2 * (m % 2 ^ (-e).toNat) > 2 ^ (-e).toNat ∨
        (2 * (m % 2 ^ (-e).toNat) = 2 ^ (-e).toNat ∧ m / 2 ^ (-e).toNat % 2 = 1)
        then m / 2 ^ (-e).toNat + 1 else m / 2 ^ (-e).toNat := by
      simp only [roundEvenDyadic, he, ↓reduceIte, Nat.shiftRight_eq_div_pow, Nat.and_one_is_mod]
    have hv : (m : ℚ) * 2 ^ e = (m : ℚ) / ((2 ^ (-e).toNat : ℕ) : ℚ) := by
      push_cast; rw [← zpow_natCast, Int.toNat_of_nonneg (by omega), zpow_neg, div_inv_eq_mul]
    rw [hr, hv]
    exact roundsEven_div m _ (pow_pos (by norm_num) _)

theorem roundEvenDyadic_units (sig kk j : ℕ) :
    RoundsEven (((sig * 2 ^ kk : ℕ) : ℚ) / 2 ^ j) (roundEvenDyadic sig ((kk : ℤ) - j)) := by
  have h := roundEvenDyadic_spec sig ((kk : ℤ) - j)
  have hv : (sig : ℚ) * 2 ^ ((kk : ℤ) - j) = ((sig * 2 ^ kk : ℕ) : ℚ) / 2 ^ j := by
    push_cast; rw [zpow_sub₀ two_ne_zero, zpow_natCast, zpow_natCast, mul_div_assoc]
  rwa [hv] at h

/-! ## Words -/

theorem mask32 {u : ℕ} (hu : u < 2 ^ 32) : u &&& 0xFFFFFFFF = u := by
  rw [show (0xFFFFFFFF : ℕ) = 2 ^ 32 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt hu]

theorem hidden_pos (u : ℕ) : 0 < (u &&& 0x7FFFFF) ||| 0x800000 := by
  rw [show (0x7FFFFF : ℕ) = 2 ^ 23 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod,
    or_hidden (Nat.mod_lt _ (by norm_num))]
  omega

/-- The cast's sign is `decode`'s. -/
theorem sign_eq {u : ℕ} (hu : u < 2 ^ 32) :
    (if u >>> 31 ≠ 0 then 0x80 else 0) = if ((u >>> 31) &&& 1 == 1) = true then 0x80 else 0 := by
  have key : ∀ b < 2, (if b ≠ 0 then 0x80 else 0) = if (b &&& 1 == 1) = true then 0x80 else 0 := by decide
  exact key _ (by rw [Nat.shiftRight_eq_div_pow]; omega)

/-! ## The cast opened -/

/-- `f32ToE4m3Sat` from its rounding on: the rounded significand `n` on the grid `2^stepExp`. -/
def tailN (sign n : ℕ) (stepExp : ℤ) : ℕ :=
  if n = 0 then sign
  else if !dyadicLt n stepExp 464 0 then sign ||| 0x7E
  else if dyadicLt n stepExp 1 (-6) then sign ||| (n <<< (stepExp + 9).toNat)
  else
    let e2 : ℤ := (Verity.TC.bitLength n : ℤ) - 1 + stepExp
    let m3 := ((n <<< 3) >>> (Verity.TC.bitLength n - 1)) - 8
    sign ||| ((e2 + 7).toNat <<< 3) ||| m3

/-- `f32ToE4m3Sat` on a finite nonzero magnitude `sig · 2^ex`, from its saturation test on. -/
def satTail (sign sig : ℕ) (ex : ℤ) : ℕ :=
  if !dyadicLt sig ex 464 0 then sign ||| 0x7E
  else
    let e : ℤ := (Verity.TC.bitLength sig : ℤ) - 1 + ex
    let stepExp : ℤ := if e ≥ -6 then e - 3 else -9
    tailN sign (roundEvenDyadic sig (ex - stepExp)) stepExp

theorem f32ToE4m3Sat_tail {u : ℕ} (hu : u < 2 ^ 32) (hE : (u >>> 23) &&& 0xFF ≠ 0xFF)
    (h0 : ¬ ((u >>> 23) &&& 0xFF = 0 ∧ u &&& 0x7FFFFF = 0)) :
    f32ToE4m3Sat u = satTail (if u >>> 31 ≠ 0 then 0x80 else 0)
      (if (u >>> 23) &&& 0xFF = 0 then u &&& 0x7FFFFF else (1 <<< 23) ||| (u &&& 0x7FFFFF))
      (if (u >>> 23) &&& 0xFF = 0 then -149 else (((u >>> 23) &&& 0xFF : ℕ) : ℤ) - 150) := by
  simp only [f32ToE4m3Sat, mask32 hu]
  rw [ite_eq_right hE, ite_eq_right h0]
  rfl

theorem satTail_sat (sign sig kk : ℕ) (h : 464 * 2 ^ 149 ≤ sig * 2 ^ kk) :
    satTail sign sig ((kk : ℤ) - 149) = sign ||| 0x7E := by
  have c1 : dyadicLt sig ((kk : ℤ) - 149) 464 0 = false := by
    cases hc : dyadicLt sig ((kk : ℤ) - 149) 464 0
    · rfl
    · exact absurd ((dyadicLt_units0 sig 464 kk).1 hc) (not_lt.2 h)
  simp only [satTail, c1, Bool.not_false, eq_self, ↓reduceIte]

theorem or_code {sign k c : ℕ} (hs : sign = 0 ∨ sign = 0x80) (hk : k < 16) (hc : c < 8) :
    sign ||| (k <<< 3) ||| c = sign + (8 * k + c) := by
  have key : ∀ k < 16, ∀ c < 8,
      (0 ||| (k <<< 3) ||| c = 0 + (8 * k + c)) ∧ (0x80 ||| (k <<< 3) ||| c = 0x80 + (8 * k + c)) := by
    decide +kernel
  rcases hs with rfl | rfl
  · exact (key k hk c hc).1
  · exact (key k hk c hc).2

theorem or_small {sign c : ℕ} (hs : sign = 0 ∨ sign = 0x80) (hc : c < 128) : sign ||| c = sign + c := by
  have key : ∀ c < 128, (0 ||| c = 0 + c) ∧ (0x80 ||| c = 0x80 + c) := by decide +kernel
  rcases hs with rfl | rfl
  · exact (key c hc).1
  · exact (key c hc).2

/-- A normal code: `n` in `[8, 16]` on the grid `2^(j - 149)`, whose exponent field is `j - 139`, carried for `n = 16`. -/
theorem tail_code {sign n j : ℕ} (hsign : sign = 0 ∨ sign = 0x80) (h8 : 8 ≤ n) (h16 : n ≤ 16) (hj : 140 ≤ j)
    (hv : 8 * (j - 140) + n ≤ 126) :
    sign ||| ((((Verity.TC.bitLength n : ℕ) : ℤ) - 1 + ((j : ℤ) - 149) + 7).toNat <<< 3) |||
      (((n <<< 3) >>> (Verity.TC.bitLength n - 1)) - 8) = sign + (8 * (j - 140) + n) := by
  rw [show (((Verity.TC.bitLength n : ℕ) : ℤ) - 1 + ((j : ℤ) - 149) + 7).toNat = Verity.TC.bitLength n + j - 143 by
    omega]
  rcases Nat.lt_or_ge n 16 with h | h
  · rw [Verity.TC.Proofs.bitLength_eq n 4 (by omega) (by omega) (by omega),
      show (n <<< 3) >>> (4 - 1) = n by rw [Nat.shiftLeft_eq, Nat.shiftRight_eq_div_pow]; omega,
      or_code hsign (by omega) (by omega)]
    omega
  · obtain rfl : n = 16 := by omega
    rw [Verity.TC.Proofs.bitLength_eq 16 5 (by omega) (by omega) (by omega),
      show (16 <<< 3) >>> (5 - 1) - 8 = 0 by decide, or_code hsign (by omega) (by omega)]
    omega

/-- `tailN` on a rounded significand `n ≤ 16` on the grid `2^(j - 149)`, `j ≥ 140`, normal (`n ≥ 8`) unless `j = 140`,
whose code `8 · (j - 140) + n` is finite: that code under the sign. -/
theorem tail_eval {sign n j : ℕ} (hsign : sign = 0 ∨ sign = 0x80) (hj : 140 ≤ j) (hn16 : n ≤ 16)
    (hlow : j = 140 ∨ 8 ≤ n) (hv : 8 * (j - 140) + n ≤ 126) :
    tailN sign n ((j : ℤ) - 149) = sign + (8 * (j - 140) + n) := by
  rcases Nat.eq_zero_or_pos n with rfl | hn0
  · obtain rfl : j = 140 := hlow.resolve_right (by omega)
    rw [tailN, ite_eq_left rfl]
    omega
  · have c2 : dyadicLt n ((j : ℤ) - 149) 464 0 = true := by
      rw [dyadicLt_units0]
      rcases Nat.lt_or_ge j 154 with h | h
      · calc n * 2 ^ j ≤ 16 * 2 ^ 153 := Nat.mul_le_mul hn16 (Nat.pow_le_pow_right (by norm_num) (by omega))
          _ < 464 * 2 ^ 149 := by norm_num
      · obtain rfl : j = 154 := by omega
        calc n * 2 ^ 154 ≤ 14 * 2 ^ 154 := Nat.mul_le_mul_right _ (by omega)
          _ < 464 * 2 ^ 149 := by norm_num
    rw [tailN, ite_eq_right (by omega : n ≠ 0)]
    simp only [c2, Bool.not_true, Bool.false_eq_true, ↓reduceIte]
    rcases Nat.lt_or_ge n 8 with h8 | h8
    · obtain rfl : j = 140 := hlow.resolve_right (by omega)
      have c3 : dyadicLt n (((140 : ℕ) : ℤ) - 149) 1 (-6) = true := by rw [dyadicLt_units6]; omega
      simp only [c3, ↓reduceIte]
      rw [show (((140 : ℕ) : ℤ) - 149 + 9).toNat = 0 by omega, Nat.shiftLeft_zero, or_small hsign (by omega)]
      omega
    · have c3 : dyadicLt n ((j : ℤ) - 149) 1 (-6) = false := by
        cases hc : dyadicLt n ((j : ℤ) - 149) 1 (-6)
        · rfl
        · have : 2 ^ 143 ≤ n * 2 ^ j :=
            calc 2 ^ 143 = 8 * 2 ^ 140 := by norm_num
              _ ≤ n * 2 ^ j := Nat.mul_le_mul h8 (Nat.pow_le_pow_right (by norm_num) hj)
          exact absurd ((dyadicLt_units6 n j).1 hc) (not_lt.2 this)
      simp only [c3, Bool.false_eq_true, ↓reduceIte]
      exact tail_code hsign h8 hn16 hj hv

/-- `satTail` below the saturation: the code `8 · (max(L, 143) - 143) + q` under the sign, `L = ⌊log₂(sig · 2^kk)⌋` and
`q` the magnitude on the grid `2^(max(L, 143) - 3)` rounded to nearest even. -/
theorem satTail_fin {sign : ℕ} (hsign : sign = 0 ∨ sign = 0x80) {sig kk : ℕ} (hsig : 0 < sig)
    (hm : sig * 2 ^ kk < 464 * 2 ^ 149) :
    ∃ q, RoundsEven (((sig * 2 ^ kk : ℕ) : ℚ) / 2 ^ (max (sig * 2 ^ kk).log2 143 - 3)) q ∧ q ≤ 16 ∧
      ((sig * 2 ^ kk).log2 < 143 ∨ 8 ≤ q) ∧ 8 * (max (sig * 2 ^ kk).log2 143 - 143) + q ≤ 126 ∧
      satTail sign sig ((kk : ℤ) - 149) = sign + (8 * (max (sig * 2 ^ kk).log2 143 - 143) + q) := by
  have hm0 : sig * 2 ^ kk ≠ 0 := (Nat.mul_pos hsig (pow_pos (by norm_num) _)).ne'
  have hlo : 2 ^ (sig * 2 ^ kk).log2 ≤ sig * 2 ^ kk := Nat.log2_self_le hm0
  have hhi : sig * 2 ^ kk < 2 ^ ((sig * 2 ^ kk).log2 + 1) := Nat.lt_log2_self
  have hL : (sig * 2 ^ kk).log2 = sig.log2 + kk := by
    refine (Nat.log2_eq_iff hm0).2 ⟨?_, ?_⟩
    · rw [pow_add]; exact Nat.mul_le_mul_right _ (Nat.log2_self_le (by omega))
    · rw [show sig.log2 + kk + 1 = sig.log2 + 1 + kk by omega, pow_add]
      exact Nat.mul_lt_mul_of_pos_right Nat.lt_log2_self (pow_pos (by norm_num) _)
  have hb : Verity.TC.bitLength sig = sig.log2 + 1 := by rw [Verity.TC.bitLength, ite_eq_right (by omega)]
  have c1 : dyadicLt sig ((kk : ℤ) - 149) 464 0 = true := (dyadicLt_units0 sig 464 kk).2 hm
  have hq := roundEvenDyadic_units sig kk (max (sig * 2 ^ kk).log2 143 - 3)
  refine ⟨_, hq, ?_⟩
  generalize (sig * 2 ^ kk).log2 = L at hlo hhi hL hq ⊢
  generalize hjd : max L 143 - 3 = j at hq ⊢
  have hbl : ((Verity.TC.bitLength sig : ℕ) : ℤ) - 1 + ((kk : ℤ) - 149) = (L : ℤ) - 149 := by rw [hb]; omega
  have hse : (if (L : ℤ) - 149 ≥ -6 then (L : ℤ) - 149 - 3 else -9) = (j : ℤ) - 149 := by split_ifs <;> omega
  have hex : (kk : ℤ) - 149 - ((j : ℤ) - 149) = (kk : ℤ) - j := by ring
  simp only [satTail, c1, hbl, hse, hex, Bool.not_true, Bool.false_eq_true, ↓reduceIte]
  generalize roundEvenDyadic sig ((kk : ℤ) - j) = n at hq ⊢
  have hP : (0 : ℚ) < 2 ^ j := by positivity
  have hPn : 0 < 2 ^ j := pow_pos (by norm_num : 0 < 2) j
  have hxlt : ∀ c : ℕ, 2 * (sig * 2 ^ kk) < (2 * c + 1) * 2 ^ j → n ≤ c := fun c h => hq.le (by
    rw [div_lt_iff₀ hP]
    have h' : ((2 * (sig * 2 ^ kk) : ℕ) : ℚ) < (((2 * c + 1) * 2 ^ j : ℕ) : ℚ) := by exact_mod_cast h
    push_cast at h' ⊢
    linarith)
  have hxge : ∀ c : ℕ, 2 * c * 2 ^ j < 2 * (sig * 2 ^ kk) + 2 ^ j → c ≤ n := fun c h => hq.ge (by
    rw [lt_div_iff₀ hP]
    have h' : ((2 * c * 2 ^ j : ℕ) : ℚ) < ((2 * (sig * 2 ^ kk) + 2 ^ j : ℕ) : ℚ) := by exact_mod_cast h
    push_cast at h' ⊢
    linarith)
  rcases Nat.lt_or_ge L 143 with hL1 | hL1
  · obtain rfl : j = 140 := by omega
    have h143 : sig * 2 ^ kk < 2 ^ 143 := lt_of_lt_of_le hhi (Nat.pow_le_pow_right (by norm_num) (by omega))
    have hn8 : n ≤ 8 := hxlt 8 (by omega)
    refine ⟨by omega, Or.inl hL1, by omega, ?_⟩
    rw [tail_eval hsign le_rfl (by omega) (Or.inl rfl) (by omega)]
    omega
  · have e3 : 2 ^ L = 2 ^ j * 8 := by rw [show L = j + 3 by omega, pow_add]; norm_num
    have hn8 : 8 ≤ n := hxge 8 (by omega)
    have hn16 : n ≤ 16 := hxlt 16 (by omega)
    have hL157 : L ≤ 157 := by
      by_contra h
      have : 2 ^ 158 ≤ 2 ^ L := Nat.pow_le_pow_right (by norm_num) (by omega)
      omega
    have hv : 8 * (j - 140) + n ≤ 126 := by
      rcases Nat.lt_or_ge L 157 with h | h
      · omega
      · obtain rfl : j = 154 := by omega
        have : n ≤ 14 := hxlt 14 (by omega)
        omega
    refine ⟨hn16, Or.inr hn8, by omega, ?_⟩
    rw [tail_eval hsign (by omega) hn16 (Or.inr hn8) hv]
    omega

/-- A finite word of magnitude `0` casts to the zero of its sign. -/
theorem cast_zero {u : ℕ} (hu : u < 2 ^ 32) {s : Bool} (h : decode u = .fin s 0) :
    f32ToE4m3Sat u = if s then 0x80 else 0 := by
  obtain ⟨hE, rfl, hm⟩ := decode_fin_fields h
  have h1 : (u >>> 23) &&& 0xFF = 0 := by
    by_contra h1
    rw [ite_eq_right h1, Nat.shiftLeft_eq] at hm
    have := Nat.mul_pos (hidden_pos u) (pow_pos (by norm_num : 0 < 2) (((u >>> 23) &&& 0xFF) - 1))
    omega
  rw [ite_eq_left h1] at hm
  simp only [f32ToE4m3Sat, mask32 hu]
  rw [ite_eq_right hE, ite_eq_left (And.intro h1 hm.symm), sign_eq hu]

/-- A finite word of nonzero magnitude `m` casts through `satTail`, `m = sig · 2^kk`. -/
theorem cast_tail {u : ℕ} (hu : u < 2 ^ 32) {s : Bool} {m : ℕ} (h : decode u = .fin s m) (hm : 0 < m) :
    ∃ sig kk : ℕ, 0 < sig ∧ m = sig * 2 ^ kk ∧
      f32ToE4m3Sat u = satTail (if s then 0x80 else 0) sig ((kk : ℤ) - 149) := by
  obtain ⟨hE, rfl, rfl⟩ := decode_fin_fields h
  by_cases h1 : (u >>> 23) &&& 0xFF = 0
  · rw [ite_eq_left h1] at hm
    refine ⟨u &&& 0x7FFFFF, 0, hm, by rw [ite_eq_left h1, pow_zero, Nat.mul_one], ?_⟩
    rw [f32ToE4m3Sat_tail hu hE fun h => Nat.pos_iff_ne_zero.1 hm h.2, sign_eq hu, ite_eq_left h1, ite_eq_left h1]
    rfl
  · refine ⟨(u &&& 0x7FFFFF) ||| 0x800000, ((u >>> 23) &&& 0xFF) - 1, hidden_pos u,
      by rw [ite_eq_right h1, Nat.shiftLeft_eq], ?_⟩
    rw [f32ToE4m3Sat_tail hu hE fun h => h1 h.1, sign_eq hu, ite_eq_right h1, ite_eq_right h1]
    congr 1
    · rw [show (1 <<< 23 : ℕ) = 0x800000 from rfl]; exact Nat.or_comm _ _
    · omega

/-! ## The cast on decoded words -/

/-- **A NaN casts to `E4M3_NAN`**, `0x7F`. -/
theorem f32ToE4m3Sat_nan {u : ℕ} (hu : u < 2 ^ 32) (h : decode u = .nan) : f32ToE4m3Sat u = E4M3_NAN := by
  rcases decode_cases u with ⟨-, -, h3⟩ | ⟨h1, h2, -⟩ | ⟨-, h3⟩
  · rw [h3] at h; cases h
  · simp only [f32ToE4m3Sat, mask32 hu]
    rw [ite_eq_left h1, ite_eq_left h2]
  · rw [h3] at h; cases h

/-- **`±inf` casts to `±0x7E`**, 448 under the sign. -/
theorem f32ToE4m3Sat_inf {u : ℕ} (hu : u < 2 ^ 32) {s : Bool} (h : decode u = .inf s) :
    f32ToE4m3Sat u = (if s then 0x80 else 0) ||| 0x7E := by
  rcases decode_cases u with ⟨h1, h2, h3⟩ | ⟨-, -, h3⟩ | ⟨-, h3⟩
  · rw [h3] at h
    obtain rfl := Decoded.inf.inj h
    simp only [f32ToE4m3Sat, mask32 hu]
    rw [ite_eq_left h1, ite_eq_right (show ¬ (u &&& 0x7FFFFF ≠ 0) from fun h => h h2), sign_eq hu]
  · rw [h3] at h; cases h
  · rw [h3] at h; cases h

/-- **Every magnitude from `464 · 2^149` (464) up casts to `±0x7E`**, the tie 464 included. -/
theorem f32ToE4m3Sat_sat {u : ℕ} (hu : u < 2 ^ 32) {s : Bool} {m : ℕ} (h : decode u = .fin s m)
    (hm : 464 * 2 ^ 149 ≤ m) : f32ToE4m3Sat u = (if s then 0x80 else 0) ||| 0x7E := by
  obtain ⟨sig, kk, -, rfl, he⟩ := cast_tail hu h (by omega)
  rw [he, satTail_sat _ _ _ hm]

/-- **Below 464 the code is `8a + q` under the sign**, `a + 140 = max(⌊log₂ m⌋, 143) - 3` and `q` the magnitude
`m / 2^(a + 140)` rounded to nearest even: `q ≤ 16`, at least 8 unless `a = 0`, and the code at most `0x7E`. -/
theorem f32ToE4m3Sat_fin {u : ℕ} (hu : u < 2 ^ 32) {s : Bool} {m : ℕ} (h : decode u = .fin s m)
    (hm : m < 464 * 2 ^ 149) :
    ∃ q, RoundsEven ((m : ℚ) / 2 ^ (max m.log2 143 - 3)) q ∧ q ≤ 16 ∧ (m.log2 < 143 ∨ 8 ≤ q) ∧
      8 * (max m.log2 143 - 143) + q ≤ 126 ∧
      f32ToE4m3Sat u = (if s then 0x80 else 0) + (8 * (max m.log2 143 - 143) + q) := by
  rcases Nat.eq_zero_or_pos m with rfl | hm0
  · refine ⟨0, ⟨by simp, by simp⟩, by norm_num, Or.inl (by rw [Nat.log2_zero]; omega),
      by rw [Nat.log2_zero]; omega, ?_⟩
    rw [cast_zero hu h, Nat.log2_zero]
    omega
  · obtain ⟨sig, kk, hsig, rfl, he⟩ := cast_tail hu h hm0
    rw [he]
    exact satTail_fin (by cases s; exacts [.inl rfl, .inr rfl]) hsig hm

/-! ## Codes -/

/-- A code's significand: the 7-bit field itself below 8 (a subnormal), else the mantissa with its hidden bit. -/
def codeSig (c : ℕ) : ℕ := if c % 128 < 8 then c % 128 else 8 + c % 128 % 8

/-- A code's exponent in units of `2^-149`: 140 (`2^-9`) for a subnormal, else the exponent field plus 139. -/
def codeExp (c : ℕ) : ℕ := if c % 128 < 8 then 140 else c % 128 / 8 + 139

/-- What `codeChk_all` checks of one code: a NaN exactly at `0x7F` and `0xFF`, never `±inf`, and otherwise the sign
`c ≥ 128` and the magnitude `codeSig c · 2^codeExp c`, at most 448. -/
def codeChk (c : ℕ) : Bool :=
  match decode (e4m3ToF32 c) with
  | .fin s m => c % 128 != 127 && s == decide (128 ≤ c) && m == codeSig c * 2 ^ codeExp c &&
      decide (m ≤ 14 * 2 ^ 154)
  | .nan => c % 128 == 127
  | .inf _ => false

theorem codeChk_all : (List.range 256).all codeChk = true := by decide +kernel

theorem code_fin {c : ℕ} (hc : c < 256) {s : Bool} {m : ℕ} (h : decode (e4m3ToF32 c) = .fin s m) :
    c % 128 ≠ 127 ∧ s = decide (128 ≤ c) ∧ m = codeSig c * 2 ^ codeExp c ∧ m ≤ 14 * 2 ^ 154 := by
  have hc' := List.all_eq_true.1 codeChk_all c (List.mem_range.2 hc)
  simp only [codeChk, h, Bool.and_eq_true, bne_iff_ne, ne_eq, beq_iff_eq, decide_eq_true_eq] at hc'
  exact ⟨hc'.1.1.1, hc'.1.1.2, hc'.1.2, hc'.2⟩

theorem code_dec {c : ℕ} (hc : c < 256) (hne : c % 128 ≠ 127) :
    decode (e4m3ToF32 c) = .fin (decide (128 ≤ c)) (codeSig c * 2 ^ codeExp c) := by
  have hc' := List.all_eq_true.1 codeChk_all c (List.mem_range.2 hc)
  cases hd : decode (e4m3ToF32 c) with
  | nan => simp only [codeChk, hd, beq_iff_eq] at hc'; exact absurd hc' hne
  | inf s => simp only [codeChk, hd, Bool.false_eq_true] at hc'
  | fin s m =>
    obtain ⟨-, rfl, rfl, -⟩ := code_fin hc hd
    rfl

theorem code_sign (s : Bool) {v : ℕ} (hv : v < 128) : decide (128 ≤ (if s then 0x80 else 0) + v) = s := by
  cases s
  · exact decide_eq_false (by show ¬ (128 ≤ 0 + v); omega)
  · exact decide_eq_true (by show 128 ≤ 0x80 + v; omega)

theorem codeSig_lt (c : ℕ) : codeSig c < 2 ^ (3 + 1) := by
  unfold codeSig; split_ifs <;> omega

theorem codeExp_ge (c : ℕ) : 140 ≤ codeExp c := by
  unfold codeExp; split_ifs <;> omega

/-- The code `8a + q` under a sign has the value `q · 2^(a + 140)`, `q = 16` carried into the next exponent. -/
theorem codeMag {sign a q : ℕ} (hs : sign = 0 ∨ sign = 0x80) (hq : q ≤ 16) (h8 : a = 0 ∨ 8 ≤ q)
    (hv : 8 * a + q ≤ 126) :
    codeSig (sign + (8 * a + q)) * 2 ^ codeExp (sign + (8 * a + q)) = q * 2 ^ (a + 140) := by
  have hmod : (sign + (8 * a + q)) % 128 = 8 * a + q := by omega
  simp only [codeSig, codeExp, hmod]
  by_cases h1 : 8 * a + q < 8
  · simp only [h1, ↓reduceIte]
    obtain rfl : a = 0 := by omega
    simp only [Nat.mul_zero, Nat.zero_add]
  · simp only [h1, ↓reduceIte]
    rcases Nat.lt_or_ge q 16 with h2 | h2
    · rw [show 8 + (8 * a + q) % 8 = q by omega, show (8 * a + q) / 8 + 139 = a + 140 by omega]
    · obtain rfl : q = 16 := by omega
      rw [show 8 + (8 * a + 16) % 8 = 8 by omega, show (8 * a + 16) / 8 + 139 = a + 141 by omega]
      ring

theorem codeMax {sign : ℕ} (hs : sign = 0 ∨ sign = 0x80) :
    codeSig (sign + 0x7E) * 2 ^ codeExp (sign + 0x7E) = 14 * 2 ^ 154 := by
  rcases hs with rfl | rfl <;> decide +kernel

/-! ## The nearest code -/

/-- **`f32ToE4m3Sat` is the nearest E4M3 code, ties to even.** For a finite word of sign `s` and magnitude `m`, the
code's value decodes to the sign `s` and a magnitude `m₀` whose signed value is at least as near to the word's as every
code's, and when a code of another value is as near, the code is even. From 464 up the code is `±0x7E`
(`f32ToE4m3Sat_sat`), the largest finite code: E4M3 saturates. -/
theorem f32ToE4m3Sat_nearest {u : ℕ} (hu : u < 2 ^ 32) {s : Bool} {m : ℕ} (h : decode u = .fin s m) :
    ∃ m₀ : ℕ, decode (e4m3ToF32 (f32ToE4m3Sat u)) = .fin s m₀ ∧
      (∀ c < 256, ∀ s' m', decode (e4m3ToF32 c) = .fin s' m' →
        |signed s m₀ - signed s m| ≤ |signed s' m' - signed s m|) ∧
      (∀ c < 256, ∀ s' m', decode (e4m3ToF32 c) = .fin s' m' → signed s' m' ≠ signed s m₀ →
        |signed s' m' - signed s m| = |signed s m₀ - signed s m| → f32ToE4m3Sat u % 2 = 0) := by
  have hS : (if s then 0x80 else 0) = 0 ∨ (if s then 0x80 else 0) = 0x80 := by
    cases s; exacts [.inl rfl, .inr rfl]
  have hS2 : (if s then 0x80 else 0) % 2 = 0 := by cases s <;> decide
  have hx0 : (0 : ℚ) ≤ m := by positivity
  by_cases hsat : m < 464 * 2 ^ 149
  · obtain ⟨q, hq, hq16, hq8, hv, he⟩ := f32ToE4m3Sat_fin hu h hsat
    have hj : max m.log2 143 - 3 = (max m.log2 143 - 143) + 140 := by omega
    have hd : decode (e4m3ToF32 (f32ToE4m3Sat u)) = .fin s (q * 2 ^ (max m.log2 143 - 3)) := by
      rw [he, code_dec (by omega) (by omega), code_sign s (by omega), codeMag hS hq16 (by omega) hv, ← hj]
    have hlow : max m.log2 143 - 3 = 140 ∨ (2 : ℚ) ^ (max m.log2 143 - 3 + 3) ≤ m := by
      rcases Nat.lt_or_ge m.log2 143 with hL | hL
      · left; omega
      · right
        have hm0 : m ≠ 0 := by rintro rfl; rw [Nat.log2_zero] at hL; omega
        rw [show max m.log2 143 - 3 + 3 = m.log2 by omega]
        exact_mod_cast Nat.log2_self_le hm0
    refine ⟨q * 2 ^ (max m.log2 143 - 3), hd, ?_, ?_⟩
    · intro c hc s' m' hd'
      obtain ⟨-, -, rfl, -⟩ := code_fin hc hd'
      push_cast
      exact signed_near s s' hx0 (by positivity)
        (nearest_grid (p := 3) (k₀ := 140) hlow hq.near (codeSig_lt c) (codeExp_ge c)).1
    · intro c hc s' m' hd' hne heq
      rw [he]
      suffices hq2 : q % 2 = 0 by omega
      by_cases ht : 2 * |(q : ℚ) - m / 2 ^ (max m.log2 143 - 3)| = 1
      · exact hq.tie ht
      · rcases Nat.eq_zero_or_pos m with rfl | hm0
        · have : q ≤ 0 := hq.le (by norm_num)
          omega
        · exfalso
          obtain ⟨-, -, rfl, -⟩ := code_fin hc hd'
          push_cast at hne heq
          exact signed_tie s s' (by exact_mod_cast hm0) (by positivity) (by positivity)
            ((nearest_grid (p := 3) (k₀ := 140) hlow hq.near (codeSig_lt c) (codeExp_ge c)).2
              (lt_of_le_of_ne hq.near ht)) hne heq
  · have hsat' : 464 * 2 ^ 149 ≤ m := by omega
    have he := f32ToE4m3Sat_sat hu h hsat'
    rw [or_small hS (by norm_num : (0x7E : ℕ) < 128)] at he
    have hd : decode (e4m3ToF32 (f32ToE4m3Sat u)) = .fin s (14 * 2 ^ 154) := by
      rw [he, code_dec (by omega) (by omega), code_sign s (by norm_num), codeMax hS]
    have h2 : ((14 * 2 ^ 154 : ℕ) : ℚ) ≤ m := Nat.cast_le.2 (by omega)
    refine ⟨14 * 2 ^ 154, hd, ?_, fun _ _ _ _ _ _ _ => by rw [he]; omega⟩
    intro c hc s' m' hd'
    obtain ⟨-, -, -, hle⟩ := code_fin hc hd'
    have h1 : (m' : ℚ) ≤ ((14 * 2 ^ 154 : ℕ) : ℚ) := Nat.cast_le.2 hle
    refine signed_near s s' hx0 (by positivity) ?_
    rw [abs_of_nonpos (by linarith), abs_of_nonpos (by linarith)]
    linarith

end Verity.Prims.Round
