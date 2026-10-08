import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring
import Proofs.Pouw.PearlC.Forming
import Proofs.Pouw.PearlC.Freeze

/-!
# P3: the atom-mix's `PairErr` (staging)

`Forming.lean`'s atom-mix instances take the chain's relative error on a flip pair as the named hypothesis `PairErr`.
This file proves it on Pearl's flat pairs, at `ε = 1/64`.

* **The step's error** (`hopperStep_err`). On a finite accumulator term `tc` and finite codes, with the step's exponent
  `E` in `[−113, 120]`, the value the cast reads is within `33·g + (|S| + 33·g)·2^−13` of the exact sum
  `S = c + Σ_l a_l·b_l`, for the grid `g = 2^(E − 13)` (`hopperStep_value`: two truncations toward zero).
* **`f32Word`'s error** (`f32Word_spec`): the accumulator input `f32(α·x)` is finite and within half an FP32 ulp.
* **Flat pairs** (`atomMix_flat_err`, `atomMix_pairErr`). On `(flat (fun _ => false), flatF s')` every noise line code
  is `E′ = e4m3(f32(44β))` (`lineCode`), the noise is `N = E′·Σ_t flatF s' t = (11/4)·E′·(32 − 2j)` (`flatNoise`,
  `flatNoise_eq`), and every product's exponent has `2^e ≤ 16·|E′| ≤ (32/11)·|N|`. With `|E′| ≥ 2^−9`, the step's
  error is at most `(α|x| + |N|)/64`. Flipping the `F` line's signs flips `N` (`flatNoise_not`).
* **The instances without `PairErr`**: `atomMixChain_saltLive_flat` and `rhoD_atomMix_pi_flat`, modulo the noise floor
  and `FlatPairs`. `lineCode_err` bounds `E′` against `44β`, for computing the pair's `c`.
-/

namespace Pouw.PearlC.AtomErr

open Pouw.Fp8Atom Pouw.Fp8Atom.Proofs Pouw.Fp8Atom.H1T Pouw.Fp8Atom.H1T.Proofs Pouw.PearlC Pouw.PearlC.Forming

/-! ## FP32 words -/

/-- A word from its fields: sign bit `sb`, exponent field `e < 255` and mantissa `m < 2^23`. -/
theorem fp32Value_word (sb e m : ℕ) (hs : sb ≤ 1) (he : e < 255) (hm : m < 2 ^ 23) :
    fp32Value (sb * 2 ^ 31 + e * 2 ^ 23 + m) =
      if e = 0 then (if m = 0 then .fin Term.zero else .fin ⟨sb == 1, -126, m⟩)
      else .fin ⟨sb == 1, (e : ℤ) - 127, 2 ^ 23 + m⟩ := by
  have h1 : (sb * 2 ^ 31 + e * 2 ^ 23 + m) >>> 23 % 256 = e := by
    rw [Nat.shiftRight_eq_div_pow]; norm_num only at hm he ⊢; omega
  have h2 : (sb * 2 ^ 31 + e * 2 ^ 23 + m) % 2 ^ 23 = m := by norm_num only at hm ⊢; omega
  have h3 : (sb * 2 ^ 31 + e * 2 ^ 23 + m) >>> 31 % 2 = sb := by
    rw [Nat.shiftRight_eq_div_pow]; norm_num only at hm he hs ⊢; omega
  have h4 : 1 <<< 23 ||| m = 2 ^ 23 + m := by
    rw [← Nat.shiftLeft_add_eq_or_of_lt hm, Nat.one_shiftLeft]
  unfold fp32Value
  simp only [h1, h2, h3, h4, ite_eq_right (show e ≠ 255 by omega)]

theorem abs_roundEven_sub_le (t : ℚ) : |(roundEven t : ℚ) - t| ≤ 1 / 2 := by
  have h1 := Int.floor_le t
  have h2 := Int.lt_floor_add_one t
  unfold roundEven; dsimp only
  split_ifs with ha hb hc <;> push_cast <;> rw [abs_le] <;> constructor <;> (try simp only [not_lt] at *) <;> linarith

/-- **`f32Word`'s value.** Below `2^127` the word is finite, and its value is within half an FP32 ulp of `y`: at most
`|y|·2^−24`, or `2^−150` among the subnormals. -/
theorem f32Word_spec {y : ℚ} (hy : |y| < 2 ^ 127) :
    ∃ t : Term, fp32Value (f32Word y) = .fin t ∧ |t.value - y| ≤ max (|y| / 2 ^ 24) (2 ^ (-150 : ℤ)) := by
  rcases eq_or_ne y 0 with rfl | hy0
  · refine ⟨Term.zero, by simp [f32Word]; rfl, ?_⟩
    simp only [Term.value, Term.zero, Nat.cast_zero, mul_zero, zero_mul, sub_zero, abs_zero]
    exact le_max_of_le_right (two_zpow_pos _).le
  have ha : 0 < |y| := abs_pos.2 hy0
  obtain ⟨hb1, hb2⟩ := binade_spec ha
  have hb127 : binade |y| ≤ 126 := by
    have h : (2 : ℚ) ^ binade |y| < 2 ^ (127 : ℤ) := lt_of_le_of_lt hb1 (by exact_mod_cast hy)
    have := (zpow_lt_zpow_iff_right₀ (one_lt_two (α := ℚ))).1 h
    omega
  set E₀ := max (binade |y|) (-126) with hE₀
  set u : ℚ := 2 ^ (E₀ - 23) with hu_def
  have hu : 0 < u := two_zpow_pos _
  set R := roundEven (|y| / u) with hR
  have hR0 : 0 ≤ R := le_trans (Int.floor_nonneg.2 (div_nonneg ha.le hu.le)) (floor_le_roundEven _)
  have hlt : |y| / u < 2 ^ 24 := by
    rw [div_lt_iff₀ hu]
    calc |y| < 2 ^ (binade |y| + 1) := hb2
      _ ≤ 2 ^ (E₀ + 1) := zpow_le_zpow_right₀ one_le_two (by omega)
      _ = 2 ^ 24 * u := by rw [hu_def, two_pow_24]
  have hR24 : R ≤ 2 ^ 24 := by
    have h1 := roundEven_le (|y| / u)
    have h2 : ⌊|y| / u⌋ < 2 ^ 24 := Int.floor_lt.2 (by exact_mod_cast hlt)
    omega
  have hR23 : -126 ≤ binade |y| → 2 ^ 23 ≤ R := by
    intro h
    have hE : E₀ = binade |y| := max_eq_left h
    have h1 : (2 : ℚ) ^ 23 ≤ |y| / u := by
      rw [le_div_iff₀ hu, hu_def, hE, two_pow_23]; exact hb1
    have h2 : (2 ^ 23 : ℤ) ≤ ⌊|y| / u⌋ := Int.le_floor.2 (by exact_mod_cast h1)
    exact le_trans h2 (floor_le_roundEven _)
  have herr : abs ((R : ℚ) * u - |y|) ≤ u / 2 := by
    have h := abs_roundEven_sub_le (|y| / u)
    rw [← hR] at h
    have e : (R : ℚ) * u - |y| = ((R : ℚ) - |y| / u) * u := by rw [sub_mul, div_mul_cancel₀ _ hu.ne']
    rw [e, abs_mul, abs_of_pos hu]
    nlinarith [abs_nonneg ((R : ℚ) - |y| / u)]
  have hhalf : u / 2 ≤ max (|y| / 2 ^ 24) (2 ^ (-150 : ℤ)) := by
    rcases le_or_gt (-126) (binade |y|) with h | h
    · apply le_max_of_le_left
      have hE : E₀ = binade |y| := max_eq_left h
      have : u * 2 ^ 24 = 2 * 2 ^ binade |y| := by
        rw [hu_def, hE, mul_comm, two_pow_24, zpow_add_one₀ two_ne_zero]; ring
      rw [div_le_div_iff₀ (by norm_num) (by norm_num)]; nlinarith
    · apply le_max_of_le_right
      have hE : E₀ = -126 := max_eq_right h.le
      rw [hu_def, hE]; norm_num
  -- the value `f32Word` encodes
  set sb : ℕ := if y < 0 then 1 else 0 with hsb
  have hs1 : sb ≤ 1 := by rw [hsb]; split <;> omega
  have hs2 : (if y < 0 then 2 ^ 31 else 0) = sb * 2 ^ 31 := by rw [hsb]; split <;> simp
  have hsgn : (if (sb == 1) = true then (-1 : ℚ) else 1) * |y| = y := by
    rw [hsb]; by_cases h : y < 0
    · simp [h, abs_of_neg h]
    · simp [h, abs_of_nonneg (not_lt.1 h)]
  obtain ⟨N₀, hN₀⟩ : ∃ n : ℕ, (n : ℤ) = R := ⟨R.toNat, Int.toNat_of_nonneg hR0⟩
  have hNt : R.toNat = N₀ := by omega
  have key : ∃ t : Term, fp32Value (f32Word y) = .fin t ∧ t.value = (if (sb == 1) = true then -1 else 1) * (R * u) := by
    have hw : f32Word y =
        let E := if N₀ = 2 ^ 24 then E₀ + 1 else E₀
        let N := if N₀ = 2 ^ 24 then 2 ^ 23 else N₀
        if 127 < E then sb * 2 ^ 31 + 0x7F800000
        else if N < 2 ^ 23 then sb * 2 ^ 31 + N
        else sb * 2 ^ 31 + (E + 127).toNat * 2 ^ 23 + (N - 2 ^ 23) := by
      unfold f32Word
      simp only [ite_eq_right ha.ne', ← hE₀, ← hu_def, ← hR, hNt, hs2]
    rw [hw]
    dsimp only
    have hR' : (R : ℚ) = N₀ := by rw [← hN₀]; norm_cast
    by_cases hc : N₀ = 2 ^ 24
    · have h127 : ¬ (127 < E₀ + 1) := by omega
      simp only [hc, ite_true, h127, ite_false, show ¬ (2 ^ 23 < 2 ^ 23) by omega, Nat.sub_self, add_zero]
      have hef : (E₀ + 1 + 127).toNat < 255 := by omega
      have hv := fp32Value_word sb _ 0 hs1 hef (by norm_num)
      rw [Nat.add_zero, ite_eq_right (by omega)] at hv
      rw [hv]
      refine ⟨_, rfl, ?_⟩
      unfold Term.value
      have hx : (((E₀ + 1 + 127).toNat : ℕ) : ℤ) - 127 - 23 = (E₀ - 23) + 1 := by omega
      simp only [hx]
      rw [hR', hc, hu_def, zpow_add_one₀ two_ne_zero]
      push_cast; ring
    · have h127 : ¬ (127 < E₀) := by omega
      simp only [hc, ite_false, h127]
      by_cases hsub : N₀ < 2 ^ 23
      · simp only [hsub, ite_true]
        have hbin : binade |y| < -126 := by
          by_contra h; have := hR23 (not_lt.1 h); omega
        have hE : E₀ = -126 := max_eq_right hbin.le
        rw [show sb * 2 ^ 31 + N₀ = sb * 2 ^ 31 + 0 * 2 ^ 23 + N₀ by ring, fp32Value_word sb 0 N₀ hs1 (by norm_num) hsub,
          ite_eq_left rfl]
        by_cases h0 : N₀ = 0
        · rw [ite_eq_left h0]
          refine ⟨_, rfl, ?_⟩
          rw [hR', h0]; simp [Term.value, Term.zero]
        · rw [ite_eq_right h0]
          refine ⟨_, rfl, ?_⟩
          unfold Term.value
          rw [hR', hu_def, hE]; norm_num
      · simp only [hsub, ite_false]
        have hef : (E₀ + 127).toNat < 255 := by omega
        have hm : N₀ - 2 ^ 23 < 2 ^ 23 := by omega
        rw [fp32Value_word sb _ _ hs1 hef hm, ite_eq_right (by omega)]
        refine ⟨_, rfl, ?_⟩
        unfold Term.value
        rw [hR', show 2 ^ 23 + (N₀ - 2 ^ 23) = N₀ by omega, hu_def,
          show (((E₀ + 127).toNat : ℕ) : ℤ) - 127 - 23 = E₀ - 23 by omega]
        ring
  obtain ⟨t, ht, hv⟩ := key
  refine ⟨t, ht, ?_⟩
  have e : t.value - y = (if (sb == 1) = true then (-1 : ℚ) else 1) * (R * u - |y|) := by
    rw [hv, mul_sub, hsgn]
  rw [e, abs_mul]
  have h1 : |(if (sb == 1) = true then (-1 : ℚ) else 1)| = 1 := by split <;> simp
  rw [h1, one_mul]
  exact le_trans herr hhalf

/-! ## The step on a finite accumulator -/

/-- **The step's value.** On a finite accumulator term `tc` and finite codes, with the step's exponent
`E = stepExp tc a b` between −113 and 120, the step writes a finite word whose value is two truncations toward zero:
the accumulator and each product to the grid `g = 2^(E − 13)`, then their sum to 14 significant bits. -/
theorem hopperStep_value {w : Nat} {tc : Term} (hw : fp32Value w = .fin tc) (a b : Fin 32 → Code)
    (ha : ∀ l, (a l).isNaN = false) (hb : ∀ l, (b l).isNaN = false) (hE1 : -113 ≤ Freeze.stepExp tc a b)
    (hE2 : Freeze.stepExp tc a b ≤ 120) :
    wordQ (hopperStep w a b) = some (trunc14 (truncTo (2 ^ (Freeze.stepExp tc a b - 13)) tc.value +
      ∑ l, truncTo (2 ^ (Freeze.stepExp tc a b - 13)) (val (a l) * val (b l)))) := by
  obtain ⟨hm2, -⟩ := Freeze.fp32Value_fin hw
  set E := Freeze.stepExp tc a b with hEdef
  have hcE : tc.mag ≠ 0 → tc.exp ≤ E := le_maxExp _ _ tc List.mem_cons_self
  have hle : ∀ l, (product (a l) (b l)).mag ≠ 0 → (product (a l) (b l)).exp ≤ E := fun l hp =>
    le_maxExp _ _ _ (List.mem_cons_of_mem _ (List.mem_ofFn.2 ⟨l, rfl⟩)) hp
  set total := alignedSum 14 E (tc :: List.ofFn fun l => product (a l) (b l)) with htdef
  rw [Freeze.hopperStep_fin hw a b ha hb]
  unfold groupSum
  change wordQ (packValue (let t := normalize 14 E total
    if t.mag ≠ 0 ∧ 127 < t.exp then Value.inf t.neg else Value.fin t)) = _
  have hsum : total = aligned 14 E tc + ∑ l, aligned 14 E (product (a l) (b l)) := by
    rw [htdef, alignedSum, List.map_cons, List.sum_cons, List.map_ofFn, List.sum_ofFn]
    simp only [Function.comp_apply]
  have htot : (total : ℚ) * 2 ^ (E - 13) =
      truncTo (2 ^ (E - 13)) tc.value + ∑ l, truncTo (2 ^ (E - 13)) (val (a l) * val (b l)) := by
    rw [hsum]; push_cast; rw [add_mul, Finset.sum_mul, aligned_value hcE, add_right_inj]
    refine Finset.sum_congr rfl fun l _ => ?_
    rw [aligned_value (hle l), productExact _ _ _ _ (val_eq (ha l)) (val_eq (hb l))]
  have hT : total.natAbs < 2 ^ 21 := by
    have hc : (aligned 14 E tc).natAbs < 2 ^ 15 := aligned_natAbs_lt E tc (by omega)
    have h1 : |∑ l : Fin 32, aligned 14 E (product (a l) (b l))| ≤
        ∑ l : Fin 32, |aligned 14 E (product (a l) (b l))| := Finset.abs_sum_le_sum_abs _ _
    have h2 : ∑ l : Fin 32, |aligned 14 E (product (a l) (b l))| ≤ ∑ _l : Fin 32, (2 ^ 15 : ℤ) := by
      refine Finset.sum_le_sum fun l _ => ?_
      have := aligned_natAbs_lt E (product (a l) (b l)) (product_mag_lt _ _)
      rw [Int.abs_eq_natAbs]; omega
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at h2
    rw [Int.abs_eq_natAbs] at h1
    rw [hsum]
    omega
  clear_value total E
  rw [← htot]
  by_cases h0 : total = 0
  · rw [h0]
    simp [normalize, Term.zero, packValue, packTerm, Freeze.wordQ_zero, trunc14_zero]
  obtain ⟨m1, m2, e1, -, e3, hv⟩ := Freeze.normalize_spec' h0 hE1 hT
  have e2 : (normalize 14 E total).exp ≤ 127 := by omega
  dsimp only
  rw [ite_eq_right (fun h => absurd h.2 (by omega)), packValue, wordQ_packTerm m1 m2 e1 e2, hv]

/-- Truncation to 14 significant bits moves `y` by at most `|y|·2^−13`. -/
theorem abs_trunc14_sub_le (y : ℚ) : |trunc14 y - y| ≤ |y| / 2 ^ 13 := by
  rcases eq_or_ne y 0 with rfl | hy
  · simp [trunc14_zero]
  have h1 := (trunc14Cell y y hy).1
  have h2 := (binade_spec (abs_pos.2 hy)).1
  have e : (2 : ℚ) ^ (binade |y| - 13) = 2 ^ binade |y| / 2 ^ 13 := by
    rw [zpow_sub₀ two_ne_zero]; norm_num
  have h3 : (2 : ℚ) ^ binade |y| / 2 ^ 13 ≤ |y| / 2 ^ 13 := div_le_div_of_nonneg_right h2 (by norm_num)
  rw [e] at h1
  linarith

/-- **The step's error.** Under `hopperStep_value`'s hypotheses, the value the cast reads (`wordVal`) is within
`33·g + (|S| + 33·g)·2^−13` of the exact sum `S = c + Σ_l a_l·b_l`, where `c` is the accumulator's value and
`g = 2^(E − 13)` is the step's grid: `33·g` for the first truncation of the accumulator and the 32 products, and the
rest for the second truncation, to 14 significant bits. -/
theorem hopperStep_err {w : Nat} {tc : Term} (hw : fp32Value w = .fin tc) (a b : Fin 32 → Code)
    (ha : ∀ l, (a l).isNaN = false) (hb : ∀ l, (b l).isNaN = false) (hE1 : -113 ≤ Freeze.stepExp tc a b)
    (hE2 : Freeze.stepExp tc a b ≤ 120) :
    |PearlC.wordVal (hopperStep w a b) - (tc.value + ∑ l, val (a l) * val (b l))| ≤
      33 * 2 ^ (Freeze.stepExp tc a b - 13) +
        (|tc.value + ∑ l, val (a l) * val (b l)| + 33 * 2 ^ (Freeze.stepExp tc a b - 13)) / 2 ^ 13 := by
  have hv := hopperStep_value hw a b ha hb hE1 hE2
  set g : ℚ := 2 ^ (Freeze.stepExp tc a b - 13) with hg_def
  have hg : 0 < g := two_zpow_pos _
  set S := tc.value + ∑ l, val (a l) * val (b l) with hS
  set T := truncTo g tc.value + ∑ l, truncTo g (val (a l) * val (b l)) with hT
  have hwv : PearlC.wordVal (hopperStep w a b) = trunc14 T := by
    unfold PearlC.wordVal; rw [hv]
  have hTS : |T - S| ≤ 33 * g := by
    have e : T - S = (truncTo g tc.value - tc.value) +
        ∑ l, (truncTo g (val (a l) * val (b l)) - val (a l) * val (b l)) := by
      rw [hT, hS, Finset.sum_sub_distrib]; ring
    have h1 : |truncTo g tc.value - tc.value| ≤ g := by
      rw [abs_sub_comm]; exact (Freeze.abs_sub_truncTo_lt hg _).le
    have h2 : |∑ l, (truncTo g (val (a l) * val (b l)) - val (a l) * val (b l))| ≤ 32 * g := by
      calc _ ≤ ∑ l, |truncTo g (val (a l) * val (b l)) - val (a l) * val (b l)| := Finset.abs_sum_le_sum_abs _ _
        _ ≤ ∑ _l : Fin 32, g := Finset.sum_le_sum fun l _ => by
            rw [abs_sub_comm]; exact (Freeze.abs_sub_truncTo_lt hg _).le
        _ = 32 * g := by simp
    rw [e]
    obtain ⟨a1, a2⟩ := abs_le.1 h1
    obtain ⟨b1, b2⟩ := abs_le.1 h2
    exact abs_le.2 ⟨by linarith, by linarith⟩
  have h14 := abs_trunc14_sub_le T
  have hTa : |T| ≤ |S| + 33 * g := by
    have := abs_sub_abs_le_abs_sub T S; linarith
  rw [hwv]
  obtain ⟨a1, a2⟩ := abs_le.1 h14
  obtain ⟨b1, b2⟩ := abs_le.1 hTS
  have h3 : |T| / 2 ^ 13 ≤ (|S| + 33 * g) / 2 ^ 13 := div_le_div_of_nonneg_right hTa (by norm_num)
  exact abs_le.2 ⟨by linarith, by linarith⟩

/-! ## The step's grid -/

theorem maxExp_cases (z : ℤ) (ts : List Term) :
    maxExp z ts = z ∨ ∃ t ∈ ts, t.mag ≠ 0 ∧ maxExp z ts = t.exp := by
  unfold maxExp
  induction ts generalizing z with
  | nil => exact Or.inl rfl
  | cons t ts ih =>
    simp only [List.foldl_cons]
    rcases ih (if t.mag ≠ 0 ∧ z < t.exp then t.exp else z) with h | ⟨t', ht', hm, h⟩
    · rw [h]
      split_ifs with hc
      · exact Or.inr ⟨t, List.mem_cons_self, hc.1, rfl⟩
      · exact Or.inl rfl
    · exact Or.inr ⟨t', List.mem_cons_of_mem _ ht', hm, h⟩

/-- `2^E ≤ B` for the step's exponent `E`, when `2^−139 ≤ B` and `2^e ≤ B` for the accumulator and every nonzero
product. -/
theorem two_zpow_stepExp_le (tc : Term) (a b : Fin 32 → Code) {B : ℚ} (hz : (2 : ℚ) ^ (-139 : ℤ) ≤ B)
    (hc : tc.mag ≠ 0 → (2 : ℚ) ^ tc.exp ≤ B)
    (hp : ∀ l, (product (a l) (b l)).mag ≠ 0 → (2 : ℚ) ^ (product (a l) (b l)).exp ≤ B) :
    (2 : ℚ) ^ Freeze.stepExp tc a b ≤ B := by
  unfold Freeze.stepExp
  rcases maxExp_cases (-139) (tc :: List.ofFn fun l => product (a l) (b l)) with e | ⟨t, ht, hm, e⟩ <;> rw [e]
  · exact hz
  · rcases List.mem_cons.1 ht with rfl | ht
    · exact hc hm
    · obtain ⟨l, rfl⟩ := List.mem_ofFn.1 ht
      exact hp l hm

theorem abs_termValue (t : Term) : |t.value| = (t.mag : ℚ) * 2 ^ (t.exp - 23) := by
  unfold Term.value
  split <;> simp [abs_mul, abs_of_pos (two_zpow_pos _)]

/-- A finite FP32 term is zero, subnormal (exponent −126) or normal (significand at least `2^23`). -/
theorem fp32Value_cases {w : Nat} {t : Term} (h : fp32Value w = .fin t) :
    t.mag = 0 ∨ t.exp = -126 ∨ 2 ^ 23 ≤ t.mag := by
  unfold fp32Value at h
  dsimp only at h
  have hm : w % 2 ^ 23 < 2 ^ 23 := Nat.mod_lt _ (by norm_num)
  split_ifs at h <;> cases h
  · exact Or.inl rfl
  · exact Or.inr (Or.inl rfl)
  · refine Or.inr (Or.inr ?_)
    dsimp only
    rw [← Nat.shiftLeft_add_eq_or_of_lt hm, Nat.one_shiftLeft]
    omega

/-- A nonzero finite FP32 term has `2^e ≤ |value| + 2^−126`. -/
theorem two_zpow_acc_le {w : Nat} {t : Term} (h : fp32Value w = .fin t) (hm : t.mag ≠ 0) :
    (2 : ℚ) ^ t.exp ≤ |t.value| + 2 ^ (-126 : ℤ) := by
  have h126 := two_zpow_pos (-126)
  have hv := abs_nonneg t.value
  rcases fp32Value_cases h with h0 | he | h23
  · exact absurd h0 hm
  · rw [he]; linarith
  · have e : (2 : ℚ) ^ t.exp = (2 : ℚ) ^ (23 : ℕ) * 2 ^ (t.exp - 23) := by
      rw [← zpow_natCast, ← zpow_add₀ two_ne_zero]; congr 1; push_cast; ring
    have hp := two_zpow_pos (t.exp - 23)
    have hle : (2 : ℚ) ^ (23 : ℕ) * 2 ^ (t.exp - 23) ≤ (t.mag : ℚ) * 2 ^ (t.exp - 23) :=
      mul_le_mul_of_nonneg_right (by exact_mod_cast h23) hp.le
    rw [e, ← abs_termValue] at *
    linarith

/-- A nonzero product's exponent: `2^e·(sig_a·sig_b) = 64·|a·b|`. -/
theorem product_two_zpow {a b : Code} (ha : a.isNaN = false) (hb : b.isNaN = false)
    (hm : (product a b).mag ≠ 0) :
    (2 : ℚ) ^ (product a b).exp * ((a.sig * b.sig : ℕ) : ℚ) = 64 * |val a * val b| := by
  rw [← productExact a b _ _ (val_eq ha) (val_eq hb), abs_termValue]
  unfold product at hm ⊢
  dsimp only at hm ⊢
  split_ifs at hm ⊢ with h
  · exact absurd rfl hm
  · dsimp only
    have e : (2 : ℚ) ^ ((a.expOr1 : ℤ) + b.expOr1 - 14) =
        2 ^ ((a.expOr1 : ℤ) + b.expOr1 - 14 - 23) * 2 ^ (23 : ℕ) := by
      rw [← zpow_natCast, ← zpow_add₀ two_ne_zero]; congr 1; push_cast; ring
    rw [e, Nat.shiftLeft_eq]
    push_cast
    ring

/-! ## Flat pairs under the atom-mix -/

/-- The noise line's code on a flat `E` line of `+44`s: `E′ = e4m3(f32(β·44))`. -/
def lineCode (β : ℚ) : Code := e4m3Word (f32Word (β * 44))

/-- The noise the atom adds on the flat pair `(flat (fun _ => false), flatF s')`: `E′`'s value times the sum of the
`F` line's values, `(11/4)·(32 − 2j)` for `j` negative entries (`flatNoise_eq`). -/
def flatNoise (β : ℚ) (s' : Fin 32 → Bool) : ℚ := codeVal (lineCode β) * ∑ t, codeVal (flatF s' t)

/-- The cast of a finite FP32 word reads the word's value: `e4m3Word u` and `e4m3` of its value have the same value
(the codes differ only at `−0`). -/
theorem codeVal_e4m3Word {u : Nat} {t : Term} (h : fp32Value u = .fin t) :
    codeVal (e4m3Word u) = codeVal (e4m3 t.value) := by
  have hq : wordQ u = some t.value := by simp [wordQ, h, Value.finite?]
  have hs : t.mag ≠ 0 → t.neg = ((u >>> 31) % 2 == 1) := by
    intro hm
    unfold fp32Value at h
    dsimp only at h
    split_ifs at h <;> cases h
    · exact absurd rfl hm
    · rfl
    · rfl
  by_cases hm : t.mag = 0
  · have h0 : t.value = 0 := by simp [Term.value, hm]
    have z : ∀ c : Code, c.val % 128 = 0 → codeVal c = 0 := by decide +kernel
    have z0 : codeVal (e4m3 0) = 0 := by decide +kernel
    rewrite [h0, z0]
    apply z
    unfold e4m3Word
    rewrite [hq]
    dsimp only
    rewrite [h0, abs_zero, show e4m3Mag 0 = 0 by simp [e4m3Mag]]
    split_ifs <;> rfl
  · have hv : t.value < 0 ↔ t.neg = true := by
      have hp : (0 : ℚ) < (t.mag : ℚ) * 2 ^ (t.exp - 23) := mul_pos (by exact_mod_cast Nat.pos_of_ne_zero hm)
        (two_zpow_pos _)
      have e1 : t.value = (if t.neg then -1 else 1) * ((t.mag : ℚ) * 2 ^ (t.exp - 23)) := by
        unfold Term.value; ring
      rcases Bool.eq_false_or_eq_true t.neg with hn | hn <;> rw [e1, hn]
      · simp only [ite_true, neg_one_mul, neg_lt_zero, iff_true]; exact hp
      · simp only [Bool.false_eq_true, ite_false, one_mul, iff_false, not_lt]; exact hp.le
    have e : (if (u >>> 31) % 2 = 1 then 128 else 0) = (if t.value < 0 then 128 else 0) := by
      have hn := hs hm
      by_cases hb : (u >>> 31) % 2 = 1
      · have ht : t.neg = true := by rw [hn]; simp [hb]
        rw [ite_eq_left hb, ite_eq_left (hv.2 ht)]
      · have ht : t.neg = false := by rw [hn]; simp [hb]
        rw [ite_eq_right hb, ite_eq_right (fun h' => absurd (hv.1 h') (by rw [ht]; simp))]
    congr 1
    unfold e4m3Word e4m3
    rewrite [hq]
    dsimp only
    simp only [e]

/-- **The noise line's value.** For `0 ≤ β` and `44β ≤ 448`, `E′ = e4m3(f32(β·44))` is within `44β/15 + 1/1000` of
`44β`: E4M3's relative error `1/16` (or its subnormal spacing) after FP32's. -/
theorem lineCode_err {β : ℚ} (hβ : 0 ≤ β) (hβ' : β * 44 ≤ 448) :
    |codeVal (lineCode β) - β * 44| ≤ β * 44 / 15 + 1 / 1000 := by
  have hy : |β * 44| = β * 44 := abs_of_nonneg (by positivity)
  obtain ⟨t, hw, hc⟩ := f32Word_spec (y := β * 44) (by rw [hy]; norm_num; linarith)
  rw [lineCode, codeVal_e4m3Word hw]
  have z150 : (2 : ℚ) ^ (-150 : ℤ) = 1 / 2 ^ 150 := by norm_num
  rw [hy, z150] at hc
  have hc' : |t.value - β * 44| ≤ β * 44 / 2 ^ 24 + 1 / 2 ^ 150 := by
    refine le_trans hc (max_le ?_ ?_)
    · have : (0 : ℚ) < 1 / 2 ^ 150 := by norm_num
      linarith
    · have : 0 ≤ β * 44 / 2 ^ 24 := by positivity
      linarith
  have hv : |t.value| ≤ β * 44 + (β * 44 / 2 ^ 24 + 1 / 2 ^ 150) := by
    have := abs_sub_abs_le_abs_sub t.value (β * 44); rw [hy] at this; linarith
  have h1 := e4m3_err (y := t.value) (by linarith)
  have h2 : max (|t.value| / 16) (1 / 1024) ≤ |t.value| / 16 + 1 / 1024 :=
    max_le (by linarith) (by have := abs_nonneg t.value; linarith)
  have h3 := abs_sub_le (codeVal (e4m3 t.value)) t.value (β * 44)
  have hb := mul_nonneg hβ (by norm_num : (0 : ℚ) ≤ 44)
  linarith

theorem sum_flat {r : Nat} (s : Fin r → Bool) :
    ∑ t, codeVal (flatF s t) = 11 / 4 * ((r : ℚ) - 2 * (Finset.univ.filter fun t => s t = true).card) := by
  have h : ∀ t, codeVal (flatF s t) = 11 / 4 - 11 / 2 * (if s t = true then 1 else 0) := by
    intro t; rw [codeVal_flatF]; cases s t <;> norm_num
  simp only [h, Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
    ← Finset.mul_sum, Finset.sum_boole]
  ring

theorem flatNoise_eq (β : ℚ) (s' : Fin 32 → Bool) :
    flatNoise β s' = codeVal (lineCode β) * (11 / 4 * (32 - 2 * (Finset.univ.filter fun t => s' t = true).card)) := by
  rw [flatNoise, sum_flat]; norm_num

theorem sum_flat_not {r : Nat} (s : Fin r → Bool) :
    ∑ t, codeVal (flatF (fun t => !s t) t) = -∑ t, codeVal (flatF s t) := by
  rw [← neg_flatF]; simp only [codeVal_negLine, Finset.sum_neg_distrib]

theorem flatNoise_not (β : ℚ) (s' : Fin 32 → Bool) : flatNoise β (fun t => !s' t) = -flatNoise β s' := by
  rw [flatNoise, flatNoise, sum_flat_not, mul_neg]

/-- A nonzero sum of a flat `F` line of length 32 is at least `11/2` in magnitude. -/
theorem abs_sum_flat_ge (s : Fin 32 → Bool) (h : ∑ t, codeVal (flatF s t) ≠ 0) :
    11 / 2 ≤ |∑ t, codeVal (flatF s t)| := by
  rw [sum_flat] at h ⊢
  set j := (Finset.univ.filter fun t => s t = true).card
  have hj : j ≠ 16 := by rintro hj; apply h; rw [hj]; norm_num
  rcases Nat.lt_or_gt_of_ne hj with hj | hj
  · have : (j : ℚ) ≤ 15 := by exact_mod_cast Nat.le_of_lt_succ hj
    rw [abs_of_nonneg (by push_cast; linarith)]; push_cast; linarith
  · have : (17 : ℚ) ≤ j := by exact_mod_cast hj
    rw [abs_of_neg (by push_cast; linarith)]; push_cast; linarith

theorem e4m3_flat {r : Nat} (s : Fin r → Bool) (t : Fin r) : (flatF s t).isNaN = false ∧ (flatF s t).sig = 11 := by
  unfold flatF; split <;> decide +kernel

/-- `E′` on a flat `E` line of `+44`s is `lineCode`'s, `β = 0` included (`+0`: the zero product's signs agree). -/
theorem lineCodes_flat (β : ℚ) (t : Fin 32) : lineCodes β (flat fun _ => false : Fin 32 → Code) t = lineCode β := by
  have h44 : codeVal (flat (fun _ => false) t : Code) = 44 := by rw [codeVal_flat]; rfl
  have hs : (flat (fun _ : Fin 32 => false) t : Code).sign = false := by
    show (e4m3 (if false = true then -44 else 44)).sign = false
    decide +kernel
  unfold lineCodes mulCode lineCode
  rw [h44, hs]
  by_cases h0 : β * 44 = 0
  · have hβ : β = 0 := by linarith
    subst hβ
    simp [f32Word]
  · rw [ite_eq_right h0]

theorem val_ne_zero_spec : ∀ c : Code, val c ≠ 0 → c.isNaN = false ∧ 1 / 512 ≤ |val c| ∧ 1 ≤ c.sig := by
  decide +kernel

/-- A product of a noise line code `L` and a flat `F` entry: `2^e ≤ 16·|L|`. -/
theorem product_flat_two_zpow {L : Code} (hLn : L.isNaN = false) (hLs : 1 ≤ L.sig) (s' : Fin 32 → Bool) (l : Fin 32)
    (hm : (product L (flatF s' l)).mag ≠ 0) : (2 : ℚ) ^ (product L (flatF s' l)).exp ≤ 16 * |val L| := by
  obtain ⟨hn, hs⟩ := e4m3_flat s' l
  have h := product_two_zpow hLn hn hm
  have hf : |val (flatF s' l)| = 11 / 4 := by
    rw [show val (flatF s' l) = codeVal (flatF s' l) from rfl, codeVal_flatF]; split <;> norm_num
  rw [hs, abs_mul, hf] at h
  push_cast at h
  have hp := two_zpow_pos (product L (flatF s' l)).exp
  have hLs' : (1 : ℚ) ≤ L.sig := by exact_mod_cast hLs
  nlinarith

/-- **The atom-mix on a flat pair.** For `0 ≤ α`, `α·|x| ≤ 448` and a nonzero noise `N = flatNoise β s'`, the value
the cast reads on the pair `(flat (fun _ => false), flatF s')` is within `(α·|x| + |N|)/64` of `α·x + N`. -/
theorem atomMix_flat_err {α x β : ℚ} (s' : Fin 32 → Bool) (hα : 0 ≤ α) (hax : α * |x| ≤ 448)
    (hN : flatNoise β s' ≠ 0) :
    |FormChain.atomMix.value α x β (flat fun _ => false) (flatF s') - (α * x + flatNoise β s')| ≤
      1 / 64 * (α * |x| + |flatNoise β s'|) := by
  have hy : |α * x| = α * |x| := by rw [abs_mul, abs_of_nonneg hα]
  have hA := mul_nonneg hα (abs_nonneg x)
  obtain ⟨tc, hw, hc⟩ := f32Word_spec (y := α * x) (by rw [hy]; norm_num; linarith)
  have hV0 : val (lineCode β) ≠ 0 := fun h => hN (by rw [flatNoise, codeVal, ← val, h, zero_mul])
  have hK0 : ∑ t, codeVal (flatF s' t) ≠ 0 := fun h => hN (by rw [flatNoise, h, mul_zero])
  obtain ⟨hLn, hV512, hLs⟩ := val_ne_zero_spec _ hV0
  obtain ⟨a, ha_def⟩ : ∃ a : Fin 32 → Code, a = fun _ => lineCode β := ⟨_, rfl⟩
  obtain ⟨b, hb_def⟩ : ∃ b : Fin 32 → Code, b = fun t => flatF s' t := ⟨_, rfl⟩
  have hal : ∀ l, a l = lineCode β := fun l => by rw [ha_def]
  have hbl : ∀ l, b l = flatF s' l := fun l => by rw [hb_def]
  have hval : FormChain.atomMix.value α x β (flat fun _ => false) (flatF s') =
      PearlC.wordVal (hopperStep (f32Word (α * x)) a b) := by
    have hla : lineCodes β (flat fun _ => false : Fin 32 → Code) = a := by
      rw [ha_def]; funext t; exact lineCodes_flat β t
    rw [hb_def, ← hla]; rfl
  rw [hval]
  have ha : ∀ l, (a l).isNaN = false := fun l => by rw [hal]; exact hLn
  have hb : ∀ l, (b l).isNaN = false := fun l => by rw [hbl]; exact (e4m3_flat s' l).1
  have hpm : ∀ l, (product (a l) (b l)).mag ≠ 0 := by
    intro l
    rw [hal, hbl]
    have h1 : ((lineCode β).sig * (flatF s' l).sig) <<< 17 ≠ 0 := by
      rw [Nat.shiftLeft_eq, (e4m3_flat s' l).2]
      exact Nat.mul_ne_zero (Nat.mul_ne_zero (by omega) (by norm_num)) (by positivity)
    unfold product; dsimp only; rw [ite_eq_right h1]; exact h1
  -- the powers of two the bound meets, as fractions
  have z150 : (2 : ℚ) ^ (-150 : ℤ) = 1 / 2 ^ 150 := by norm_num
  have z139 : (2 : ℚ) ^ (-139 : ℤ) = 1 / 2 ^ 139 := by norm_num
  have z126 : (2 : ℚ) ^ (-126 : ℤ) = 1 / 2 ^ 126 := by norm_num
  have z120 : (2 : ℚ) ^ (120 : ℤ) = 2 ^ 120 := by norm_num
  -- the accumulator
  have hc0 : |tc.value - α * x| ≤ α * |x| / 2 ^ 24 + 1 / 2 ^ 150 := by
    rw [hy, z150] at hc
    refine le_trans hc (max_le ?_ ?_)
    · have : (0 : ℚ) < 1 / 2 ^ 150 := by norm_num
      linarith
    · have : 0 ≤ α * |x| / 2 ^ 24 := by positivity
      linarith
  have hTv : |tc.value| ≤ α * |x| + |tc.value - α * x| := by
    have := abs_sub_abs_le_abs_sub tc.value (α * x); rw [hy] at this; linarith
  -- the step's exponent
  have hE1 : -113 ≤ Freeze.stepExp tc a b := by
    have h := le_maxExp (-139) (tc :: List.ofFn fun l => product (a l) (b l)) (product (a 0) (b 0))
      (List.mem_cons_of_mem _ (List.mem_ofFn.2 ⟨0, rfl⟩)) (hpm 0)
    have := (product_exp _ _ (hpm 0)).1
    unfold Freeze.stepExp; omega
  have hE2 : Freeze.stepExp tc a b ≤ 120 := by
    refine maxExp_le (-139) 120 _ (by norm_num) fun t ht htm => ?_
    rcases List.mem_cons.1 ht with rfl | ht
    · have h2 := two_zpow_acc_le hw htm
      rw [z126] at h2
      have h3 : (2 : ℚ) ^ t.exp < 2 ^ (120 : ℤ) := by rw [z120]; linarith
      exact ((zpow_lt_zpow_iff_right₀ (one_lt_two (α := ℚ))).1 h3).le
    · obtain ⟨l, rfl⟩ := List.mem_ofFn.1 ht
      exact le_trans (product_exp _ _ htm).2 (by norm_num)
  have herr := hopperStep_err hw a b ha hb hE1 hE2
  have hsum : ∑ l, val (a l) * val (b l) = flatNoise β s' := by
    rw [flatNoise, codeVal, ← val, Finset.mul_sum]
    exact Finset.sum_congr rfl fun l _ => by rw [hal, hbl]; rfl
  rw [hsum] at herr
  -- the grid
  have hNabs : |flatNoise β s'| = |val (lineCode β)| * |∑ t, codeVal (flatF s' t)| := by
    rw [flatNoise, codeVal, ← val, abs_mul]
  have hK88 := abs_sum_flat_ge s' hK0
  have hVN : 11 / 2 * |val (lineCode β)| ≤ |flatNoise β s'| := by
    rw [hNabs, mul_comm]; exact mul_le_mul_of_nonneg_left hK88 (abs_nonneg _)
  have hG : (2 : ℚ) ^ Freeze.stepExp tc a b ≤ |tc.value| + 1 / 2 ^ 126 + 16 * |val (lineCode β)| := by
    have hV := abs_nonneg (val (lineCode β))
    have hT := abs_nonneg tc.value
    refine two_zpow_stepExp_le tc a b (by rw [z139]; linarith) (fun hm => by
      have := two_zpow_acc_le hw hm; rw [z126] at this; linarith) fun l hm => ?_
    rw [hal, hbl] at hm ⊢
    have := product_flat_two_zpow hLn hLs s' l hm
    linarith
  have hg : (2 : ℚ) ^ (Freeze.stepExp tc a b - 13) = 2 ^ Freeze.stepExp tc a b / 2 ^ 13 := by
    rw [zpow_sub₀ two_ne_zero]; norm_num
  rw [hg] at herr
  -- the bound
  have hS : |tc.value + flatNoise β s'| ≤ |tc.value| + |flatNoise β s'| := abs_add_le _ _
  have htri := abs_sub_le (PearlC.wordVal (hopperStep (f32Word (α * x)) a b)) (tc.value + flatNoise β s')
    (α * x + flatNoise β s')
  have e : tc.value + flatNoise β s' - (α * x + flatNoise β s') = tc.value - α * x := by ring
  rw [e] at htri
  linarith

/-- **`PairErr` for the atom-mix at `ε = 1/64`.** On Pearl's flat pairs `(flat (fun _ => false), flatF ¬s')` and
`(flat (fun _ => false), flatF s')`, with `0 ≤ α`, `α·|x| ≤ 448` and the noise `N = flatNoise β s'` positive. -/
theorem atomMix_pairErr {α x β : ℚ} (s' : Fin 32 → Bool) (hα : 0 ≤ α) (hax : α * |x| ≤ 448)
    (hN : 0 < flatNoise β s') :
    PairErr FormChain.atomMix.value α x β (flatNoise β s') (1 / 64)
      (flat fun _ => false, flatF fun t => !s' t) (flat fun _ => false, flatF s') := by
  have hN' := flatNoise_not β s'
  constructor
  · have h := atomMix_flat_err (β := β) (fun t => !s' t) hα hax (by rw [hN']; exact neg_ne_zero.2 hN.ne')
    rw [hN', abs_neg, abs_of_pos hN, ← sub_eq_add_neg] at h
    exact h
  · have h := atomMix_flat_err (β := β) s' hα hax hN.ne'
    rw [abs_of_pos hN] at h
    exact h

/-! ## `atomMixChain_saltLive` and `rhoD_atomMix_pi` without `PairErr` -/

/-- **`atomMixChain_saltLive` on flat pairs, without `PairErr`.** With `FlatPairs S` and the noise floor `ασ ≥ 1`, a
sign pattern `s'` whose flat-pair noise is `flatNoise β s' = c·ασ` with `8/15 ≤ c ≤ 4` makes the element salt-live
when `(81/64)·|x| < (943/64)·c·σ` (`atomMix_pairErr`'s `ε = 1/64`). -/
theorem atomMixChain_saltLive_flat {S : Support 32} (hS : FlatPairs S) {α x β σ c : ℚ} (s' : Fin 32 → Bool)
    (hN : flatNoise β s' = c * (α * σ)) (hα : 0 < α) (hfloor : 1 ≤ α * σ) (hsat : α * (|x| + 4 * σ) ≤ 448)
    (hc : 8 / 15 ≤ c) (hc4 : c ≤ 4) (hx : (1 + 17 * (1 / 64)) * |x| < (15 - 17 * (1 / 64)) * c * σ) :
    SaltLiveVal FormChain.atomMix.value S α x β := by
  have hax : α * |x| ≤ 448 := by
    have : α * (|x| + 4 * σ) = α * |x| + 4 * (α * σ) := by ring
    linarith
  have herr := atomMix_pairErr (β := β) s' hα.le hax (by rw [hN]; exact mul_pos (by linarith) (by linarith))
  rw [hN] at herr
  exact atomMixChain_saltLive (hS _ _) (hS _ _) herr hα hfloor hsat hc hc4 (by norm_num) (by norm_num) hx

/-- **`rhoD_atomMix_pi` on flat pairs, without `PairErr`.** With `FlatPairs S`, the noise floor and §5's scale, per row
a sign pattern `s' i` whose flat-pair noise is `flatNoise (β i) (s' i) = c_i·α_iρ_i` with `648/943 ≤ c_i ≤ 4`: the
liveness rule gives the forming budget with `ρ_D = 1/64`, as for #311's chain (`rhoD_bf16_pi`). -/
theorem rhoD_atomMix_pi_flat (fs : ℕ) (s : Shape) {S : Support Params.pi.r} (hS : FlatPairs S)
    {α β ρ c : Fin s.m → ℚ} {X : Matrix (Fin s.m) (Fin s.k) ℚ} (s' : Fin s.m → Fin 32 → Bool)
    (hN : ∀ i, flatNoise (β i) (s' i) = c i * (α i * ρ i)) (hα : ∀ i, 0 < α i) (hfloor : ∀ i, 1 ≤ α i * ρ i)
    (hsat : ∀ i l, α i * (|X i l| + 4 * ρ i) ≤ 448) (hc : ∀ i, 648 / 943 ≤ c i) (hc4 : ∀ i, c i ≤ 4)
    (hlive : LiveOperand Params.pi.δ ρ X) :
    (fs : ℚ) * s.m * s.k - rhoD * fs * s.m * s.k ≤ fs * (live FormChain.atomMix.value S α β X).card := by
  refine rhoD_atomMix_pi fs s (ε := 1 / 64) (c := c)
    (p := fun i => (flat fun _ => false, flatF fun t => !s' i t)) (q := fun i => (flat fun _ => false, flatF (s' i)))
    (fun i => hS _ _) (fun i => hS _ _) (fun i l => ?_) hα hfloor hsat (fun i => by linarith [hc i]) hc4
    (by norm_num) (by norm_num) hlive
  have hax : α i * |X i l| ≤ 448 := by
    have : α i * (|X i l| + 4 * ρ i) = α i * |X i l| + 4 * (α i * ρ i) := by ring
    linarith [hsat i l, hfloor i]
  have h := atomMix_pairErr (β := β i) (s' i) (hα i).le hax
    (by rw [hN i]; exact mul_pos (by linarith [hc i]) (by linarith [hfloor i]))
  rw [hN i] at h
  exact h

end Pouw.PearlC.AtomErr
