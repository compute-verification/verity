import Proofs.Pouw.PearlC.PeelExists
import Proofs.Pouw.PearlC.TileCheck4Words

/-!
# The FP4 peel's rows exist (proofs)

`Pc4EntryOK` asks that the FP4 peel form `C̃_ij`'s word and the rows `P_A,i`, `P_B,j` (`PeelFp4Proofs.peelRows4` on
`sm120Nvf4`): that both rows form (`formRow`), and that no chain, FMA or BF16 narrowing leaves the finite range. At
`64 ∣ k ≤ 2^16` that holds on every pair of rows that `Row4OK` admits, under `Pc4OpsAgree`, by magnitudes:

* **A firing** (`step4_fin`): from a finite accumulator of magnitude `x`, a firing whose group terms exist writes a
  finite word at most `x` plus the groups' scaled dots, below `2^128`, as `truncRZ` truncates toward zero
  (`truncRZ_abs_le`, `toF32RZ_le`). A group's scaled dot is at most `16·36·S_a·S_b` at scale bytes of values at most
  `S_a` and `S_b` (`groupTerm_le`, `groupTerms_le`).
* **A chain** (`chainN_fin`, `cTilde_fin`): `K/64` firings from `+0` stay at most `K/64·2304·S_a·S_b`. Every scale byte
  a chain reads is valid: a formed row's are `e4m3` casts or `0x01`, never NaN (`scaleLe_448`, at most `448`), and the
  bases' are `1.0` (`scaleLe_one`), so every chain is finite and at most `2^64` (`ch4_fin`).
* **A formed row** (`formRow_some`): a noise atom's accumulator is `f32(α·x)`, finite since `α·|x| ≤ 2689`
  (`RowStats`' `alpha4_mul_le`), and its firing adds at most `2304·448/16` (`noiseElem_fin`), so the row forms, with
  `K` codes and `K/16` scale bytes, none NaN.
* **The rows** (`peelRows4_isSome`): `P_A,i` and `P_B,j` exist (`PeelExists.paRow_fin`, `pbRow_fin`), with the weight
  row's `Ẽ` finite BF16 words at most `448` at a scale byte that is not NaN (`eTilde_le`, by `decide`).

`pc4EntryOK_of_rows` is `Pc4EntryOK` from the two rows' `Row4OK`, so `TileCheckWords`' `tileCheck4_computesWords`
names no hypothesis at an entry.
-/

namespace Pouw.PearlC.PeelExists

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.Fp4 Pouw.PearlC.Fp4Freeze Pouw.PearlC.PeelExact Pouw.PearlC.Fp4Dev
  Pouw.PearlC.RowCircuit

/-! ## A firing -/

/-- `truncRZ` truncates toward zero: its multiple of `2^lsb` is at most `m·2^e` in magnitude. -/
theorem truncRZ_abs_le (m e lsb : ℤ) : |(truncRZ m e lsb : ℚ)| * 2 ^ lsb ≤ |(m : ℚ)| * 2 ^ e := by
  have two : (2 : ℚ) ≠ 0 := two_ne_zero
  unfold truncRZ
  by_cases h : lsb ≤ e
  · rw [ite_eq_left h]
    push_cast
    rw [abs_mul, abs_of_pos (by positivity : (0 : ℚ) < 2 ^ (e - lsb).toNat), mul_assoc, ← zpow_natCast,
      ← zpow_add₀ two, Int.toNat_of_nonneg (by omega), show e - lsb + lsb = e by ring]
  · rw [ite_eq_right h]
    set s := (lsb - e).toNat with hs
    have hsz : (s : ℤ) = lsb - e := Int.toNat_of_nonneg (by omega)
    have hq : ((m.natAbs >>> s : ℕ) : ℚ) * 2 ^ s ≤ m.natAbs := by
      rw [Nat.shiftRight_eq_div_pow]; exact_mod_cast Nat.div_mul_le_self _ _
    have habs : |((if m < 0 then -((m.natAbs >>> s : ℕ) : ℤ) else ((m.natAbs >>> s : ℕ) : ℤ) : ℤ) : ℚ)| =
        ((m.natAbs >>> s : ℕ) : ℚ) := by
      split_ifs
      · rw [Int.cast_neg, abs_neg, Int.cast_natCast, Nat.abs_cast]
      · rw [Int.cast_natCast, Nat.abs_cast]
    simp only []
    rw [habs, ← natAbs_cast, show lsb = (s : ℤ) + e by omega, zpow_add₀ two, zpow_natCast, ← mul_assoc]
    exact mul_le_mul_of_nonneg_right hq (zpow_pos (by norm_num) e).le

/-- `bitLength` of a nonzero number below `2^a` is at most `a`. -/
theorem bitLength_le {q a : ℕ} (hq : q < 2 ^ a) : bitLength q ≤ a := by
  unfold bitLength
  split_ifs with h0
  · omega
  · have := (Nat.log2_lt h0).2 hq
    omega

/-- **`toF32RZ` truncates toward zero**: below `2^128` in magnitude, `N·2^lsb` is written as a finite word of value at
most `N·2^lsb` in magnitude. -/
theorem toF32RZ_le {N lsb : ℤ} (hN : |(N : ℚ)| * 2 ^ lsb < 2 ^ 128) :
    ∃ w, toF32RZ N lsb = some w ∧ ∃ y, wordQ w = some y ∧ |y| ≤ |(N : ℚ)| * 2 ^ lsb := by
  have two : (2 : ℚ) ≠ 0 := two_ne_zero
  have hpos : ∀ z : ℤ, (0 : ℚ) < 2 ^ z := fun z => zpow_pos (by norm_num) z
  set M := N.natAbs with hM
  set eV : ℤ := (bitLength M : ℤ) - 1 + lsb with heV
  set g : ℤ := if -126 ≤ eV then eV - 23 else -149 with hg
  set Q : ℕ := if 0 < g - lsb then M >>> (g - lsb).toNat else M <<< (-(g - lsb)).toNat with hQ
  have hMabs : (M : ℚ) = |(N : ℚ)| := natAbs_cast N
  have hQb : (Q : ℚ) * 2 ^ g ≤ M * 2 ^ lsb ∧ (M : ℚ) * 2 ^ lsb < (Q + 1) * 2 ^ g := by
    rw [hQ]
    split_ifs with hs
    · set s := (g - lsb).toNat with hsd
      have hsz : (s : ℤ) = g - lsb := Int.toNat_of_nonneg hs.le
      have hg' : g = (s : ℤ) + lsb := by omega
      rw [Nat.shiftRight_eq_div_pow, hg', zpow_add₀ two, zpow_natCast, ← mul_assoc, ← mul_assoc]
      constructor
      · exact mul_le_mul_of_nonneg_right (by exact_mod_cast Nat.div_mul_le_self M (2 ^ s)) (hpos _).le
      · refine mul_lt_mul_of_pos_right ?_ (hpos _)
        have := Nat.lt_mul_div_succ M (show 0 < 2 ^ s by positivity)
        rw [mul_comm] at this
        exact_mod_cast this
    · set s := (-(g - lsb)).toNat with hsd
      have hsz : (s : ℤ) = lsb - g := by rw [hsd, Int.toNat_of_nonneg (by omega)]; ring
      have hl : lsb = (s : ℤ) + g := by omega
      have he : ((M <<< s : ℕ) : ℚ) * 2 ^ g = M * 2 ^ lsb := by
        rw [Nat.shiftLeft_eq, hl, zpow_add₀ two, zpow_natCast]; push_cast; ring
      refine ⟨he.le, ?_⟩
      rw [← he]
      exact mul_lt_mul_of_pos_right (by linarith) (hpos _)
  by_cases hQ0 : Q = 0
  · refine ⟨0, ?_, 0, wordQ_zero, by rw [abs_zero]; exact mul_nonneg (abs_nonneg _) (hpos _).le⟩
    unfold toF32RZ
    simp only []
    rw [← hM, ← heV, ← hg, ← hQ, ite_eq_left hQ0]
  have hM0 : M ≠ 0 := by
    intro h0
    apply hQ0
    rw [hQ, h0]
    split_ifs <;> simp
  have hN0 : N ≠ 0 := by rintro rfl; exact hM0 rfl
  have hbM : bitLength M = M.log2 + 1 := bitLength_of_bounds (Nat.log2_self_le hM0) Nat.lt_log2_self
  have hMlo : (2 : ℚ) ^ eV ≤ M * 2 ^ lsb := by
    rw [heV, hbM, show ((M.log2 + 1 : ℕ) : ℤ) - 1 + lsb = (M.log2 : ℤ) + lsb by push_cast; ring,
      zpow_add₀ two, zpow_natCast]
    exact mul_le_mul_of_nonneg_right (by exact_mod_cast Nat.log2_self_le hM0) (hpos _).le
  have hMhi : (M : ℚ) * 2 ^ lsb < 2 ^ (eV + 1) := by
    rw [heV, hbM, show ((M.log2 + 1 : ℕ) : ℤ) - 1 + lsb + 1 = ((M.log2 + 1 : ℕ) : ℤ) + lsb by push_cast; ring,
      zpow_add₀ two, zpow_natCast]
    exact mul_lt_mul_of_pos_right (by exact_mod_cast Nat.lt_log2_self) (hpos _)
  have hQ24 : Q < 2 ^ 24 := by
    have h1 : (Q : ℚ) * 2 ^ g < 2 ^ (24 : ℕ) * 2 ^ g := by
      calc (Q : ℚ) * 2 ^ g ≤ M * 2 ^ lsb := hQb.1
        _ < 2 ^ (eV + 1) := hMhi
        _ ≤ 2 ^ (24 : ℕ) * 2 ^ g := by
          rw [← zpow_natCast, ← zpow_add₀ two]
          exact zpow_le_zpow_right₀ (by norm_num) (by rw [hg]; split_ifs <;> push_cast <;> omega)
    exact_mod_cast lt_of_mul_lt_mul_right h1 (hpos _).le
  have hgV : g = if -126 ≤ ((bitLength Q : ℕ) : ℤ) - 1 + g then ((bitLength Q : ℕ) : ℤ) - 1 + g - 23 else -149 := by
    by_cases hn : -126 ≤ eV
    · have hge : g = eV - 23 := by rw [hg, ite_eq_left hn]
      have hQlo : 2 ^ 23 ≤ Q := by
        have h1 : (2 : ℚ) ^ (23 : ℕ) * 2 ^ g < (Q + 1) * 2 ^ g := by
          rw [← zpow_natCast, ← zpow_add₀ two, show ((23 : ℕ) : ℤ) + g = eV by rw [hge]; push_cast; ring]
          exact lt_of_le_of_lt hMlo hQb.2
        have := lt_of_mul_lt_mul_right h1 (hpos _).le
        have : (2 : ℕ) ^ 23 < Q + 1 := by exact_mod_cast this
        omega
      have hbQ : bitLength Q = 24 := bitLength_of_bounds hQlo hQ24
      rw [hbQ, ite_eq_left (by push_cast; omega)]
      push_cast; omega
    · have hge : g = -149 := by rw [hg, ite_eq_right hn]
      have hQ23 : Q < 2 ^ 23 := by
        have h1 : (Q : ℚ) * 2 ^ g < 2 ^ (23 : ℕ) * 2 ^ g := by
          calc (Q : ℚ) * 2 ^ g ≤ M * 2 ^ lsb := hQb.1
            _ < 2 ^ (eV + 1) := hMhi
            _ ≤ 2 ^ (23 : ℕ) * 2 ^ g := by
              rw [← zpow_natCast, ← zpow_add₀ two]
              exact zpow_le_zpow_right₀ (by norm_num) (by rw [hge]; push_cast; omega)
        exact_mod_cast lt_of_mul_lt_mul_right h1 (hpos _).le
      have := bitLength_le hQ23
      rw [ite_eq_right (by omega)]
      exact hge
  set Ns : ℤ := if N < 0 then -(Q : ℤ) else Q with hNs
  have hNsabs : Ns.natAbs = Q := by rw [hNs]; split_ifs <;> simp
  have hNsneg : Ns < 0 ↔ N < 0 := by
    rw [hNs]
    by_cases hn : N < 0
    · simp only [hn, ite_true, iff_true]; omega
    · simp only [hn, ite_false, iff_false]; omega
  have heq : toF32RZ N lsb = toF32RZ Ns g := by
    unfold toF32RZ
    simp only []
    rw [← hM, ← heV, ← hg, ← hQ, hNsabs, ← hgV, sub_self, ite_eq_right (lt_irrefl 0), neg_zero, Int.toNat_zero,
      Nat.shiftLeft_zero]
    simp only [hNsneg]
  have hNsQ : |(Ns : ℚ)| = Q := by rw [← natAbs_cast, hNsabs]
  have hfit : Fits ((Ns : ℚ) * 2 ^ g) := by
    refine Or.inr ⟨Ns, g, ?_, ?_, rfl, ?_⟩
    · rw [Int.abs_eq_natAbs, hNsabs]; exact_mod_cast hQ24
    · rw [hg]; split_ifs with hn <;> omega
    · rw [abs_mul, abs_of_pos (hpos g), hNsQ]
      calc (Q : ℚ) * 2 ^ g ≤ M * 2 ^ lsb := hQb.1
        _ = |(N : ℚ)| * 2 ^ lsb := by rw [hMabs]
        _ < 2 ^ 128 := hN
  obtain ⟨w, hw, -, hwv⟩ := toF32RZ_exact (by rw [← Int.natAbs_ne_zero, hNsabs]; exact hQ0) hfit
  refine ⟨w, heq ▸ hw, _, hwv, ?_⟩
  rw [abs_mul, abs_of_pos (hpos g), hNsQ, ← hMabs]
  exact hQb.1

/-- The step's sum of truncations is at most the groups' scaled dots in magnitude. -/
theorem sum_truncRZ_abs_le (lsb : ℤ) : ∀ ts : List GroupTerm,
    |(((ts.map fun t => truncRZ t.mant t.unit lsb).sum : ℤ) : ℚ)| * 2 ^ lsb ≤
      (ts.map fun t => |(t.mant : ℚ)| * 2 ^ t.unit).sum
  | [] => by simp
  | t :: ts => by
    simp only [List.map_cons, List.sum_cons, Int.cast_add]
    have h1 := truncRZ_abs_le t.mant t.unit lsb
    have h2 := sum_truncRZ_abs_le lsb ts
    have hp : (0 : ℚ) ≤ 2 ^ lsb := (zpow_pos (by norm_num) _).le
    calc |((truncRZ t.mant t.unit lsb : ℤ) : ℚ) + (((ts.map fun t => truncRZ t.mant t.unit lsb).sum : ℤ) : ℚ)| *
          2 ^ lsb
        ≤ (|((truncRZ t.mant t.unit lsb : ℤ) : ℚ)| +
            |(((ts.map fun t => truncRZ t.mant t.unit lsb).sum : ℤ) : ℚ)|) * 2 ^ lsb :=
          mul_le_mul_of_nonneg_right (abs_add_le _ _) hp
      _ ≤ _ := by rw [add_mul]; exact add_le_add h1 h2

/-- **A firing from a finite accumulator is finite**: on operands whose group terms exist, from an accumulator of value
`x`, the word written is finite and at most `|x|` plus the groups' scaled dots in magnitude, below `2^128`. -/
theorem step4_fin (p : Fp4Atom) {acc : ℕ} {x : ℚ} (hx : wordQ acc = some x) {a b : List Nib}
    {sa sb : List (Fin 256)} {ts : List GroupTerm} (hts : p.groupTerms a b sa sb = some ts) {T : ℚ}
    (hT : (ts.map fun t => |(t.mant : ℚ)| * 2 ^ t.unit).sum ≤ T) (hB : |x| + T < 2 ^ 128) :
    ∃ w, p.step acc a b sa sb = some w ∧ ∃ y, wordQ w = some y ∧ |y| ≤ |x| + T := by
  obtain ⟨tc, htc, rfl⟩ := PeelExactProofs.wordQ_fin hx
  have hT0 : 0 ≤ T := le_trans (List.sum_nonneg fun y hy => by
    obtain ⟨t, -, rfl⟩ := List.mem_map.1 hy
    exact mul_nonneg (abs_nonneg _) (zpow_pos (by norm_num) _).le) hT
  have h0 : 0 ≤ |tc.value| + T := add_nonneg (abs_nonneg _) hT0
  unfold Fp4Atom.step
  simp only [hts, htc, Value.finite?, Option.bind_eq_bind, Option.bind_some]
  split
  · exact ⟨0, rfl, 0, wordQ_zero, by rw [abs_zero]; exact h0⟩
  · rename_i x0 xs _
    have hcmv : |((if tc.neg then -(tc.mag : ℤ) else (tc.mag : ℤ) : ℤ) : ℚ)| * 2 ^ (tc.exp - 23) = |tc.value| := by
      rw [abs_value]
      split_ifs <;> simp
    generalize (if tc.neg then -(tc.mag : ℤ) else (tc.mag : ℤ)) = cm at hcmv ⊢
    generalize max (xs.foldl max x0) p.gridFloor = lsb
    have hbound : |((((ts.map fun t => truncRZ t.mant t.unit lsb).sum + truncRZ cm (tc.exp - 23) lsb : ℤ)) : ℚ)| *
        2 ^ lsb ≤ |tc.value| + T := by
      rw [Int.cast_add]
      calc _ ≤ (|(((ts.map fun t => truncRZ t.mant t.unit lsb).sum : ℤ) : ℚ)| +
            |((truncRZ cm (tc.exp - 23) lsb : ℤ) : ℚ)|) * 2 ^ lsb :=
          mul_le_mul_of_nonneg_right (abs_add_le _ _) (zpow_pos (by norm_num) _).le
        _ ≤ T + |tc.value| := by
          rw [add_mul]
          exact add_le_add ((sum_truncRZ_abs_le lsb ts).trans hT) (hcmv ▸ truncRZ_abs_le cm (tc.exp - 23) lsb)
        _ = |tc.value| + T := add_comm _ _
    by_cases htot : (ts.map fun t => truncRZ t.mant t.unit lsb).sum + truncRZ cm (tc.exp - 23) lsb = 0
    · rw [ite_eq_left htot]
      exact ⟨0, rfl, 0, wordQ_zero, by rw [abs_zero]; exact h0⟩
    · rw [ite_eq_right htot]
      obtain ⟨w, hw, y, hy, hyb⟩ := toF32RZ_le (lt_of_le_of_lt hbound hB)
      exact ⟨w, hw, y, hy, hyb.trans hbound⟩

/-! ## A firing's group terms -/

theorem numerator_natAbs_le : ∀ c : Nib, (numerator c).natAbs ≤ 12 := by decide +kernel

theorem sum_abs_le_mul (B : ℤ) : ∀ l : List ℤ, (∀ x ∈ l, |x| ≤ B) → |l.sum| ≤ l.length * B
  | [], _ => by simp
  | x :: l, h => by
    have h1 := h x List.mem_cons_self
    have h2 := sum_abs_le_mul B l fun y hy => h y (List.mem_cons_of_mem _ hy)
    simp only [List.sum_cons, List.length_cons]
    push_cast
    calc |x + l.sum| ≤ |x| + |l.sum| := abs_add_le _ _
      _ ≤ B + l.length * B := add_le_add h1 h2
      _ = (l.length + 1) * B := by ring

/-- A scale byte of `f` with a value at most `S`. -/
def ScaleLe (f : ScaleFmt) (S : ℚ) (s : Fin 256) : Prop := ∃ d, scaleDyadic f s = some d ∧ (d.1 : ℚ) * 2 ^ d.2 ≤ S

/-- **A group's scaled dot is at most `group·36·S_a·S_b`**: at most `group` products of numerators (quarters, each at
most `12·12`), times the two scales. -/
theorem groupTerm_le (p : Fp4Atom) (a b : List Nib) {sa sb : List (Fin 256)} (j : ℕ) {Sa Sb : ℚ}
    {s1 s2 : Fin 256} (h1 : sa[j * p.group / p.block]? = some s1) (h2 : sb[j * p.group / p.block]? = some s2)
    (hs1 : ScaleLe p.fmt Sa s1) (hs2 : ScaleLe p.fmt Sb s2) :
    ∃ t, p.groupTerm a b sa sb j = some t ∧ |(t.mant : ℚ)| * 2 ^ t.unit ≤ p.group * 36 * Sa * Sb := by
  obtain ⟨⟨ma, ea⟩, hd1, hS1⟩ := hs1
  obtain ⟨⟨mb, eb⟩, hd2, hS2⟩ := hs2
  simp only at hS1 hS2
  set ps := (((a.drop (j * p.group)).take p.group).zip ((b.drop (j * p.group)).take p.group)).map fun x =>
    numerator x.1 * numerator x.2 with hps
  refine ⟨⟨ps.sum * ma * mb, -2 + ea + eb, ps.any (· != 0) && ma != 0 && mb != 0⟩, ?_, ?_⟩
  · simp only [Fp4Atom.groupTerm, h1, h2, hd1, hd2, Option.bind_eq_bind, Option.bind_some, Option.pure_def, ← hps]
  · have hlen : ps.length ≤ p.group := by
      rw [hps, List.length_map, List.length_zip]
      exact (min_le_left _ _).trans (List.length_take_le _ _)
    have hsum : |ps.sum| ≤ ps.length * 144 := sum_abs_le_mul 144 ps fun x hx => by
      rw [hps] at hx
      obtain ⟨y, -, rfl⟩ := List.mem_map.1 hx
      rw [abs_mul]
      have ha := numerator_natAbs_le y.1
      have hb := numerator_natAbs_le y.2
      rw [Int.abs_eq_natAbs, Int.abs_eq_natAbs]
      have : (numerator y.1).natAbs * (numerator y.2).natAbs ≤ 144 := Nat.mul_le_mul ha hb
      exact_mod_cast this
    have hsumQ : |(ps.sum : ℚ)| ≤ p.group * 144 := by
      have : |ps.sum| ≤ (p.group : ℤ) * 144 :=
        hsum.trans (mul_le_mul_of_nonneg_right (by exact_mod_cast hlen) (by norm_num))
      rw [← Int.cast_abs]; exact_mod_cast this
    have two : (2 : ℚ) ≠ 0 := two_ne_zero
    have hma : (0 : ℚ) ≤ ma := Nat.cast_nonneg _
    have hmb : (0 : ℚ) ≤ mb := Nat.cast_nonneg _
    have hSa : 0 ≤ (ma : ℚ) * 2 ^ ea := mul_nonneg hma (zpow_pos (by norm_num) _).le
    have hSb : 0 ≤ (mb : ℚ) * 2 ^ eb := mul_nonneg hmb (zpow_pos (by norm_num) _).le
    show |((ps.sum * ma * mb : ℤ) : ℚ)| * 2 ^ (-2 + ea + eb) ≤ _
    rw [show ((ps.sum * ma * mb : ℤ) : ℚ) = (ps.sum : ℚ) * ma * mb by push_cast; ring, abs_mul, abs_mul,
      abs_of_nonneg hma, abs_of_nonneg hmb, zpow_add₀ two, zpow_add₀ two]
    calc |(ps.sum : ℚ)| * ma * mb * (2 ^ (-2 : ℤ) * 2 ^ ea * 2 ^ eb)
        = |(ps.sum : ℚ)| / 4 * ((ma : ℚ) * 2 ^ ea) * ((mb : ℚ) * 2 ^ eb) := by norm_num; ring
      _ ≤ (p.group * 144) / 4 * Sa * Sb := by
          have hA : |(ps.sum : ℚ)| / 4 ≤ (p.group * 144) / 4 := by linarith
          have h0 : 0 ≤ |(ps.sum : ℚ)| / 4 := by positivity
          exact mul_le_mul (mul_le_mul hA hS1 hSa (h0.trans hA)) hS2 hSb
            (mul_nonneg (h0.trans hA) (hSa.trans hS1))
      _ = p.group * 36 * Sa * Sb := by ring

theorem mapM_le {f : ℕ → Option GroupTerm} {B : ℚ} :
    ∀ l : List ℕ, (∀ j ∈ l, ∃ t, f j = some t ∧ |(t.mant : ℚ)| * 2 ^ t.unit ≤ B) →
      ∃ ts, l.mapM f = some ts ∧ (ts.map fun t => |(t.mant : ℚ)| * 2 ^ t.unit).sum ≤ l.length * B
  | [], _ => ⟨[], rfl, by simp⟩
  | j :: l, h => by
    obtain ⟨t, ht, htb⟩ := h j List.mem_cons_self
    obtain ⟨ts, hts, hb⟩ := mapM_le l fun j' hj' => h j' (List.mem_cons_of_mem _ hj')
    refine ⟨t :: ts, by rw [List.mapM_cons, ht, hts]; rfl, ?_⟩
    simp only [List.map_cons, List.sum_cons, List.length_cons]
    push_cast
    linarith

/-- **An NVFP4 firing's group terms exist and are at most `2304·S_a·S_b`** on 64 code pairs and 4 scale bytes per
operand, every byte of `sa` valid and at most `S_a`, of `sb` at most `S_b`. -/
theorem groupTerms_le {a b : List Nib} {sa sb : List (Fin 256)} {Sa Sb : ℚ} (ha : a.length = 64)
    (hb : b.length = 64) (hsa : sa.length = 4) (hsb : sb.length = 4) (hSa : ∀ s ∈ sa, ScaleLe .ue4m3 Sa s)
    (hSb : ∀ s ∈ sb, ScaleLe .ue4m3 Sb s) :
    ∃ ts, sm120Nvf4.groupTerms a b sa sb = some ts ∧
      (ts.map fun t => |(t.mant : ℚ)| * 2 ^ t.unit).sum ≤ 2304 * Sa * Sb := by
  unfold Fp4Atom.groupTerms
  rw [ite_eq_left ⟨ha, hb, hsa, hsb⟩]
  obtain ⟨ts, hts, hle⟩ := mapM_le (f := sm120Nvf4.groupTerm a b sa sb) (B := 16 * 36 * Sa * Sb)
    (List.range (sm120Nvf4.k / sm120Nvf4.group)) fun j hj => by
      have hj4 : j < 4 := List.mem_range.1 hj
      have hidx : j * sm120Nvf4.group / sm120Nvf4.block = j := by
        show j * 16 / 16 = j
        omega
      have h1 : sa[j * sm120Nvf4.group / sm120Nvf4.block]? = some sa[j] := by
        rw [hidx]; exact List.getElem?_eq_getElem (by omega)
      have h2 : sb[j * sm120Nvf4.group / sm120Nvf4.block]? = some sb[j] := by
        rw [hidx]; exact List.getElem?_eq_getElem (by omega)
      exact groupTerm_le sm120Nvf4 a b j h1 h2 (hSa _ (List.getElem_mem _)) (hSb _ (List.getElem_mem _))
  refine ⟨ts, hts, hle.trans (le_of_eq ?_)⟩
  rw [List.length_range]
  show ((64 / 16 : ℕ) : ℚ) * ((16 : ℕ) * 36 * Sa * Sb) = _
  norm_num
  ring

/-! ## The chain -/

/-- **NVFP4's chain is finite**: `n` firings from a finite accumulator of magnitude `x`, every scale byte of `sa` valid
and at most `S_a` and of `sb` at most `S_b`, write a finite word at most `x + n·2304·S_a·S_b`, below `2^128`. -/
theorem chainN_fin {Sa Sb : ℚ} (hSa0 : 0 ≤ Sa) (hSb0 : 0 ≤ Sb) :
    ∀ (n c : ℕ) (a b : List Nib) (sa sb : List (Fin 256)) (x : ℚ), wordQ c = some x → a.length = n * 64 →
      b.length = n * 64 → sa.length = n * 4 → sb.length = n * 4 → (∀ s ∈ sa, ScaleLe .ue4m3 Sa s) →
      (∀ s ∈ sb, ScaleLe .ue4m3 Sb s) → |x| + n * (2304 * Sa * Sb) < 2 ^ 128 →
      ∃ w, sm120Nvf4.chainN n c a b sa sb = some w ∧ ∃ y, wordQ w = some y ∧ |y| ≤ |x| + n * (2304 * Sa * Sb)
  | 0, c, _, _, _, _, x, hx, _, _, _, _, _, _, _ => ⟨c, rfl, x, hx, by simp⟩
  | n + 1, c, a, b, sa, sb, x, hx, ha, hb, hsa, hsb, hSa, hSb, hB => by
    have hT0 : 0 ≤ 2304 * Sa * Sb := by positivity
    have hk : sm120Nvf4.k = 64 := rfl
    have hs : sm120Nvf4.scales = 4 := rfl
    obtain ⟨ts, hts, hT⟩ := groupTerms_le (a := a.take 64) (b := b.take 64) (sa := sa.take 4) (sb := sb.take 4)
      (by rw [List.length_take]; omega) (by rw [List.length_take]; omega) (by rw [List.length_take]; omega)
      (by rw [List.length_take]; omega) (fun s hs => hSa s (List.mem_of_mem_take hs))
      (fun s hs => hSb s (List.mem_of_mem_take hs))
    push_cast at hB
    obtain ⟨d, hd, y, hy, hyb⟩ := step4_fin sm120Nvf4 hx hts hT (by nlinarith)
    obtain ⟨w, hw, y', hy', hyb'⟩ := chainN_fin hSa0 hSb0 n d (a.drop 64) (b.drop 64) (sa.drop 4) (sb.drop 4) y hy
      (by rw [List.length_drop]; omega) (by rw [List.length_drop]; omega) (by rw [List.length_drop]; omega)
      (by rw [List.length_drop]; omega) (fun s hs => hSa s (List.mem_of_mem_drop hs))
      (fun s hs => hSb s (List.mem_of_mem_drop hs)) (by nlinarith)
    refine ⟨w, ?_, y', hy', ?_⟩
    · show (do
        let d ← sm120Nvf4.step c (a.take sm120Nvf4.k) (b.take sm120Nvf4.k) (sa.take sm120Nvf4.scales)
          (sb.take sm120Nvf4.scales)
        sm120Nvf4.chainN n d (a.drop sm120Nvf4.k) (b.drop sm120Nvf4.k) (sa.drop sm120Nvf4.scales)
          (sb.drop sm120Nvf4.scales)) = some w
      rw [hk, hs, hd]
      exact hw
    · push_cast
      nlinarith

/-- **NVFP4's `C̃` is finite** over `K` code pairs (`64 ∣ K`) and `K/16` scale bytes per operand, each valid and at most
`S_a`, `S_b`: at most `K/64·2304·S_a·S_b`, where that is below `2^128`. -/
theorem cTilde_fin {Sa Sb : ℚ} (hSa0 : 0 ≤ Sa) (hSb0 : 0 ≤ Sb) {K : ℕ} (hK : 64 ∣ K) {a b : List Nib}
    {sa sb : List (Fin 256)} (ha : a.length = K) (hb : b.length = K) (hsa : sa.length = K / 16)
    (hsb : sb.length = K / 16) (hSa : ∀ s ∈ sa, ScaleLe .ue4m3 Sa s) (hSb : ∀ s ∈ sb, ScaleLe .ue4m3 Sb s)
    (hB : ((K / 64 : ℕ) : ℚ) * (2304 * Sa * Sb) < 2 ^ 128) :
    ∃ w, cTilde sm120Nvf4 a b sa sb = some w ∧ ∃ y, wordQ w = some y ∧ |y| ≤ (K / 64 : ℕ) * (2304 * Sa * Sb) := by
  obtain ⟨q, rfl⟩ := hK
  have hq : 64 * q / 64 = q := by omega
  have hq16 : 64 * q / 16 = q * 4 := by omega
  obtain ⟨w, hw, y, hy, hyb⟩ := chainN_fin hSa0 hSb0 q 0 a b sa sb 0 Freeze.wordQ_zero (by rw [ha]; ring)
    (by rw [hb]; ring) (by rw [hsa, hq16]) (by rw [hsb, hq16]) hSa hSb (by rw [abs_zero, zero_add, ← hq]; exact hB)
  refine ⟨w, ?_, y, hy, by rw [hq]; simpa using hyb⟩
  unfold cTilde Fp4Atom.chain
  have hk : sm120Nvf4.k = 64 := rfl
  have hs : sm120Nvf4.scales = 4 := rfl
  rw [ite_eq_left ⟨by rw [ha, hb], by rw [ha, hk]; omega, by rw [hsa, ha, hk, hs, hq16, hq], by rw [hsb, ha, hk, hs, hq16, hq]⟩,
    ha, hk, hq]
  exact hw

theorem eTilde_elem_le : ∀ c : Nib, ∀ s : Fin 256, s.val % 128 ≠ 127 →
    ((wordQ ((bf16WordOf (|decode c| * codeVal s / 16) ||| (if c.sign then 0x8000 else 0)) <<< 16)).map
      fun v => decide (|v| ≤ 448)) = some true := by
  decide +kernel

theorem scaleDyadic_le : ∀ s : Fin 256, s.val % 128 ≠ 127 →
    ((scaleDyadic .ue4m3 s).map fun d => decide ((d.1 : ℚ) * 2 ^ d.2 ≤ 448)) = some true := by
  decide +kernel

theorem scaleDyadic_one : scaleDyadic .ue4m3 0x38 = some (8, -3) := by decide +kernel
theorem scaleDyadic_f : scaleDyadic .ue4m3 0x18 = some (8, -7) := by decide +kernel

/-! ## The scale bytes and `Ẽ` -/

theorem scaleLe_448 {s : Fin 256} (hs : s.val % 128 ≠ 127) : ScaleLe .ue4m3 448 s := by
  have h := scaleDyadic_le s hs
  rcases hd : scaleDyadic .ue4m3 s with _ | d
  · rw [hd] at h; cases h
  · rw [hd] at h
    exact ⟨d, hd, of_decide_eq_true (Option.some.inj h)⟩

theorem scaleLe_one : ScaleLe .ue4m3 1 0x38 := ⟨(8, -3), scaleDyadic_one, by norm_num⟩

theorem scaleLe_f : ScaleLe .ue4m3 (1 / 16) 0x18 := ⟨(8, -7), scaleDyadic_f, by norm_num⟩

/-- `Ẽ`'s words at a scale byte that is not NaN are finite BF16 words of magnitude at most `448`. -/
theorem eTilde_le (e : Fin 32 → Nib) {aE : Fin 256} (h : aE.val % 128 ≠ 127) (q : Fin 32) :
    ∃ a, wordQ (eTilde e aE q <<< 16) = some a ∧ |a| ≤ 448 := by
  have hq : ((wordQ (eTilde e aE q <<< 16)).map fun v => decide (|v| ≤ 448)) = some true := eTilde_elem_le (e q) aE h
  rcases hw : wordQ (eTilde e aE q <<< 16) with _ | a
  · rw [hw] at hq; cases hq
  · rw [hw] at hq
    exact ⟨a, rfl, of_decide_eq_true (Option.some.inj hq)⟩

/-! ## A formed row -/

theorem rnWord_false (v : ℚ) : rnWord v false = f32Word v := by
  unfold rnWord
  by_cases h : v = 0
  · rw [ite_eq_left h, h]; simp [f32Word]
  · rw [ite_eq_right h]

/-- **NVFP4's noise element is finite** from an accumulator `f32(α·x)` with `|α·x| ≤ 2689`: `E′`'s blocks at
`e4m3(β)` (at most `448`, never NaN) and `F`'s at `2⁻⁴` add at most `2304·448/16` to it. -/
theorem noiseElem_fin {α x β : ℚ} (h : |α * x| ≤ 2689) (e f : Fin 32 → Nib) :
    ∃ w, nvf4NoiseElem α x β e f = some w := by
  obtain ⟨y, hy⟩ := RowCircuit.rnWord_finite (v := α * x) false (by linarith)
  have herr := PeelExactProofs.rnWord_err hy
  rw [rnWord_false] at hy
  have hyb : |y| ≤ 2690 := by
    have h150 : (2 : ℚ) ^ (-150 : ℤ) ≤ 1 / 2 := by norm_num
    have h1 := abs_sub_abs_le_abs_sub y (α * x)
    linarith
  obtain ⟨ts, hts, hT⟩ := groupTerms_le (a := noiseOperand sm120Nvf4 e) (b := noiseOperand sm120Nvf4 f)
    (sa := noiseScales sm120Nvf4 (e4m3 β)) (sb := noiseScales sm120Nvf4 (fScale sm120Nvf4.fmt)) (Sa := 448)
    (Sb := 1 / 16) (by simp [noiseOperand, sm120Nvf4]) (by simp [noiseOperand, sm120Nvf4])
    (by rw [noiseScales, List.length_replicate]; rfl) (by rw [noiseScales, List.length_replicate]; rfl)
    (fun s hs => by
      rw [noiseScales] at hs
      rw [List.eq_of_mem_replicate hs]
      exact scaleLe_448 (RowCircuit.e4m3_notNaN β))
    (fun s hs => by
      rw [noiseScales] at hs
      rw [List.eq_of_mem_replicate hs]
      exact scaleLe_f)
  obtain ⟨w, hw, -⟩ := step4_fin sm120Nvf4 hy hts hT (by linarith)
  exact ⟨w, hw⟩

/-- **A row whose `α·x` is at most `2689` everywhere forms**, to `K` codes and `K/16` scale bytes, none NaN. -/
theorem formRow_some {K : ℕ} (h16 : 16 ∣ K) {α β : ℚ} {X : Fin K → ℚ} (hα : ∀ t, |α * X t| ≤ 2689)
    (e : Fin 32 → Nib) (F : Fin K → Fin 32 → Nib) :
    ∃ r, formRow α β X e F = some r ∧ r.1.length = K ∧ r.2.length = K / 16 ∧ ∀ s ∈ r.2, s.val % 128 ≠ 127 := by
  choose d hd using fun t => noiseElem_fin (β := β) (hα t) e (F t)
  have hds : (List.ofFn fun t => nvf4NoiseElem α (X t) β e (F t)).mapM id = some (List.ofFn d) :=
    mapM_ofFn id _ d hd
  have hr : formRow α β X e F = some ((((List.ofFn d).toChunks 16).map nvf4Block).map Prod.fst |>.flatten,
      (((List.ofFn d).toChunks 16).map nvf4Block).map Prod.snd) := by
    simp only [formRow, hds, Option.bind_eq_bind, Option.bind_some, Option.pure_def]
  obtain ⟨h1, h2⟩ := RowCircuit.formRow_length h16 hr
  refine ⟨_, hr, h1, h2, fun s hs => ?_⟩
  simp only [List.map_map, List.mem_map, Function.comp_apply] at hs
  obtain ⟨c, -, rfl⟩ := hs
  show (nvf4Scale _).val % 128 ≠ 127
  unfold nvf4Scale
  rcases le_total (e4m3 (wordVal (f32Word ((c.map fun w => |wordVal w|).foldl max 0 * sixth)))) 1 with h | h
  · rw [max_eq_right h]; decide
  · rw [max_eq_left h]; exact RowCircuit.e4m3_notNaN _

/-! ## The peel's rows -/

/-- A finite chain word of NVFP4, at most `2^64`, where `K/64·2304·S_a·S_b` is. -/
theorem ch4_fin {Sa Sb : ℚ} (hSa0 : 0 ≤ Sa) (hSb0 : 0 ≤ Sb) {K : ℕ} (hK : 64 ∣ K) {a b : List Nib}
    {sa sb : List (Fin 256)} (ha : a.length = K) (hb : b.length = K) (hsa : sa.length = K / 16)
    (hsb : sb.length = K / 16) (hSa : ∀ s ∈ sa, ScaleLe .ue4m3 Sa s) (hSb : ∀ s ∈ sb, ScaleLe .ue4m3 Sb s)
    (hB : ((K / 64 : ℕ) : ℚ) * (2304 * Sa * Sb) ≤ 2 ^ 64) :
    ∃ w, PeelFp4Proofs.ch4 sm120Nvf4 a b sa sb = some w ∧ ∃ y, wordQ w = some y ∧ |y| ≤ 2 ^ 64 := by
  obtain ⟨w, hw, y, hy, hyb⟩ := cTilde_fin hSa0 hSb0 hK ha hb hsa hsb hSa hSb
    (lt_of_le_of_lt hB (by norm_num))
  exact ⟨w, by rw [PeelFp4Proofs.ch4, hw, Option.bind_some, RowCircuit.finite?_of_wordQ hy], y, hy, hyb.trans hB⟩

/-- **The FP4 peel's rows exist** at `64 ∣ k ≤ 2^16`, on formed rows `a`, `b` of `k` codes and `k/16` scale bytes,
none NaN, with row `j`'s scaled noise `beB j` finite BF16 words of magnitude at most `448`: every chain is finite and
at most `2^64`, so `P_A,i` and `P_B,j` exist (`paRow_fin`, `pbRow_fin`). -/
theorem peelRows4_isSome {m n k : ℕ} (hk : 64 ∣ k) (hk' : k ≤ 2 ^ 16) {A : Op4 m} {B : Op4 n}
    (FA FB : Matrix (Fin k) (Fin 32) Nib) (beA : Matrix (Fin m) (Fin 32) ℕ) (beB : Matrix (Fin n) (Fin 32) ℕ)
    (i : Fin m) (j : Fin n) {a b : Row4} (hA : A i = some a) (hB : B j = some b)
    (ha : a.1.length = k ∧ a.2.length = k / 16 ∧ ∀ s ∈ a.2, s.val % 128 ≠ 127)
    (hb : b.1.length = k ∧ b.2.length = k / 16 ∧ ∀ s ∈ b.2, s.val % 128 ≠ 127)
    (heB : ∀ q, ∃ v, wordQ (beB j q <<< 16) = some v ∧ |v| ≤ 448) :
    (PeelFp4Proofs.peelRows4 sm120Nvf4 A B FA FB beA beB i j).isSome := by
  have hk64 : ((k / 64 : ℕ) : ℚ) ≤ 1024 := by exact_mod_cast (show k / 64 ≤ 1024 by omega)
  have hbd : ∀ {S : ℚ}, 0 ≤ S → S ≤ 2304 * 448 * 448 → ((k / 64 : ℕ) : ℚ) * S ≤ 2 ^ 64 := by
    intro S h0 h
    calc ((k / 64 : ℕ) : ℚ) * S ≤ 1024 * (2304 * 448 * 448) := mul_le_mul hk64 h h0 (by norm_num)
      _ ≤ 2 ^ 64 := by norm_num
  have hone : (onesScales sm120Nvf4 k).length = k / 16 := by
    rw [onesScales, List.length_replicate]; rfl
  have hone' : ∀ s ∈ onesScales sm120Nvf4 k, ScaleLe .ue4m3 1 s := fun s hs => by
    rw [onesScales] at hs
    rw [List.eq_of_mem_replicate hs]
    exact scaleLe_one
  have hcol : ∀ (F : Matrix (Fin k) (Fin 32) Nib) q, (colF F q).length = k := fun F q => List.length_ofFn
  have hSa : ∀ s ∈ a.2, ScaleLe .ue4m3 448 s := fun s hs => scaleLe_448 (ha.2.2 s hs)
  have hSb : ∀ s ∈ b.2, ScaleLe .ue4m3 448 s := fun s hs => scaleLe_448 (hb.2.2 s hs)
  obtain ⟨c, hc, -⟩ := ch4_fin (by norm_num) (by norm_num) hk ha.1 hb.1 ha.2.1 hb.2.1 hSa hSb
    (hbd (by norm_num) le_rfl)
  choose af hafE afv hafv hafb using fun q : Fin 32 => ch4_fin (Sa := 448) (Sb := 1) (by norm_num) (by norm_num) hk
    ha.1 (hcol FB q) ha.2.1 hone hSa hone' (hbd (by norm_num) (by norm_num))
  choose bf hbfE bfv hbfv hbfb using fun q : Fin 32 => ch4_fin (Sa := 448) (Sb := 1) (by norm_num) (by norm_num) hk
    hb.1 (hcol FA q) hb.2.1 hone hSb hone' (hbd (by norm_num) (by norm_num))
  choose g hgE gv hgv hgb using fun p q : Fin 32 => ch4_fin (Sa := 1) (Sb := 1) (by norm_num) (by norm_num) hk
    (hcol FB p) (hcol FA q) hone hone hone' hone' (hbd (by norm_num) (by norm_num))
  have hafL : (List.ofFn fun q => PeelFp4Proofs.ch4 sm120Nvf4 a.1 (colF FB q) a.2 (onesScales sm120Nvf4 k)).mapM id =
      some (List.ofFn af) := mapM_ofFn id _ af hafE
  have hbfL : (List.ofFn fun q => PeelFp4Proofs.ch4 sm120Nvf4 b.1 (colF FA q) b.2 (onesScales sm120Nvf4 k)).mapM id =
      some (List.ofFn bf) := mapM_ofFn id _ bf hbfE
  have hgL : (List.ofFn fun p => (List.ofFn fun q => PeelFp4Proofs.ch4 sm120Nvf4 (colF FB p) (colF FA q)
      (onesScales sm120Nvf4 k) (onesScales sm120Nvf4 k)).mapM id).mapM id =
      some (List.ofFn fun p => List.ofFn (g p)) := mapM_ofFn id _ _ fun p => mapM_ofFn id _ (g p) (hgE p)
  obtain ⟨pa, hpa⟩ := paRow_fin (List.ofFn (beA i)) (fun p => WholeCircuit.ch4_lt (hafE p))
    fun p => ⟨afv p, hafv p, hafb p⟩
  obtain ⟨pb, hpb⟩ := pbRow_fin heB (fun q => ⟨bfv q, hbfv q, hbfb q⟩) fun p q => ⟨gv p q, hgv p q, hgb p q⟩
  unfold PeelFp4Proofs.peelRows4
  simp only [hA, hB, hc, hafL, hbfL, hgL, hpa, hpb, Option.bind_eq_bind, Option.bind_some, Option.pure_def,
    Option.isSome_some]

/-- **`Pc4EntryOK` from the two rows' `Row4OK`**, at `64 ∣ k ≤ 2^16`, under `Pc4OpsAgree`:
each row's `α·x` is at most `2689` (`alpha4_mul_le`), so both rows form (`formRow_some`), and the weight row's `Ẽ`
is at `e4m3(β)`, never NaN (`eTilde_le`). -/
theorem pc4EntryOK_of_rows {ops : RowOps} (hops : Pc4OpsAgree ops) {m n k : ℕ} (hk : 64 ∣ k) (hk' : k ≤ 2 ^ 16)
    (z : Lines4 m n k) (act wt : Codes) (i : Fin m) (j : Fin n)
    (hRa : Row4OK ops k (wordsVal m k act i)) (hRb : Row4OK ops k (wordsVal n k wt j)) :
    WholeCircuit.Pc4EntryOK ops z act wt i j := by
  have hα : ∀ {X : Fin k → ℚ}, Row4OK ops k X → ∀ t, |wordVal (alpha4 ops (encRow16 k X) k) * X t| ≤ 2689 := by
    intro X hR t
    rw [abs_mul, abs_of_pos hR.alpha_pos]
    exact RowCircuit.alpha4_mul_le hops hR t
  have h16 : 16 ∣ k := dvd_trans (by norm_num) hk
  obtain ⟨a, ha, haL⟩ := formRow_some h16 (β := WholeCircuit.beta4V ops (wordsVal m k act i)) (hα hRa) (z.EA i) z.FA
  obtain ⟨b, hb, hbL⟩ := formRow_some h16 (β := WholeCircuit.beta4V ops (wordsVal n k wt j)) (hα hRb) (z.EB j) z.FB
  exact ⟨peelRows4_isSome hk hk' z.FA z.FB _ _ i j ha hb haL hbL fun q => eTilde_le (z.EB j) (RowCircuit.e4m3_notNaN _) q⟩

end Pouw.PearlC.PeelExists
