import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Definitions.Pouw.PearlC.Fp4Skip
import Proofs.Pouw.PearlC.AccumProofs
import Proofs.Pouw.PearlC.AtomErr

/-!
# Pearl-C4's replayed debit: exactness, and the soundness of terms 2 and 3

Proofs about `Pouw.PearlC.Fp4Skip` on any `Fp4Atom` (`sm120Nvf4`, `sm120Mxf4`), for TT_OUT-FP4's debit.

* **The exactness lemma** (`step_exact`, `chainMask_exact`, `cTilde_exact`, `cTilde_dot`). A step is exact when
  - `NoLoss`: no addend loses a bit to the step's grid `2^lsb` (`gridLsb`), i.e. every group's scaled dot and the
    accumulator's significand are multiples of `2^lsb` (`fp4_census.py`'s step with no `glost` and no `alost`); and
  - `Fits`: the exact sum is a binary32 value (no `nlost`, no overflow).

  Such a step writes exactly the accumulator plus `Σ mant·2^unit` (`step_exact`). An `ExactRun`, whose unskipped steps
  are all exact from their exact partial sums, writes the word of the exact sum of the unskipped atoms' contributions
  (`chainMask_exact`). From `+0`, C̃ is then the exact dot product, taken group by group with each group's block scales
  (`cTilde_dot`). `OnGrid` reads `NoLoss` off the windows: 27 fraction bits below every participating group and 35 below
  the accumulator. It is sufficient when `accWindow ≥ 24` (`OnGrid.noLoss`).
* **The gap: D₄ gives neither hypothesis.**
  - `NoLoss`: D-SS lets a pair's unit weights span 31.6 binades, past the 27-bit window, and a participating group whose
    dot cancels still anchors the grid. `noLoss_gap` is an `sm120Nvf4` firing with scale bytes inside D-SS (a spread
    of 2^13.1 per operand) whose grid drops a bit of a sum that fits binary32. Debit term 5 (sub-grid blocks) charges the
    blocks that drop out entirely. A partial loss like this one skips nothing.
  - `Fits`: the running sum's bit span, which D₄ doesn't bound (`K` up to 2^16). Once the sum needs more than 24 bits,
    the binary32 write drops the low ones.

  So exactness is a per-run hypothesis (`ExactRun`), which the census measures; it does not follow from D₄.
* **Term 2** (`fp4PureFlags_iff`, `unitPureFlags_iff`). Unconditionally, atom `t` is flagged iff the chain is in the
  model and skipping `t` alone leaves its word unchanged. So the debit charges exactly the atoms a program may skip.
  On exact runs (the chain and the lone skip both) it flags exactly the atoms whose exact contribution is 0, the
  zero-sum atoms (`fp4PureFlags_exact_iff`). Off them it also flags frozen atoms.
* **Term 3** (`blockTerms_congr`, `deadPair_salt_free`, `deadBlocks_salt_free`). A block's group terms read only its
  codes and scale byte on each side. This needs `BlockShape` (groups inside blocks inside firings), which both atoms
  have. Assume `DeadFixes`: the forming gives every dead block the same codes and scale byte at every salt. Then a dead ×
  dead (word, block) is the same at every salt, and every pair-word that `pairWords` counts is precomputable.
  `DeadFixes` is not given by D₄. `dead` screens the clean row for noise below 1/8 of the code step, so the codes
  barely depend on the salt; whether they depend on it at all is a property of the forming.
* **What terms 1 and 4 need** (stated, not proved here):
  - Term 1 (2:4) needs:
    - a per-code salt-dead flag at the FP4 forming;
    - the count of 4-groups with at least 2 flagged codes, per (row, k-block);
    - the 2:4 saving (0.5 per MAC, 1.0 when all 4 are salt-dead) as a hardware price.

    Its base split reproduces words only on exact runs (`ExactRun`). What it leaves (a 2:4 pass over a correction that
    is not E2M1) is TT_OUT-FP4's `no-exact-rewrite/fp4-sm120`.
  - Term 4 (forming) needs:
    - the same salt-dead flag;
    - the FP4 `needed`: element `(i, l)` is read by atom `l / p.k` of a word the flags leave unflagged (`Skip.needed`,
      whose atoms are 32 wide, at the FP4 atom's width);
    - the salt-dead bound `#salt-dead ≤ ρ_D·m·k`, as a hypothesis;
    - `f_s`, as a price.
-/

namespace Pouw.PearlC.Fp4Freeze

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.Fp4 Pouw.PearlC.Fp4Skip Pouw.PearlC.AccumProofs

/-- A value binary32 holds exactly: zero, or a 24-bit significand at an exponent of at least −149, below `2^128`. -/
def Fits (v : ℚ) : Prop := v = 0 ∨ ∃ m e : ℤ, |m| < 2 ^ 24 ∧ -149 ≤ e ∧ v = m * 2 ^ e ∧ |v| < 2 ^ 128

theorem bitLength_of_bounds {q a : ℕ} (h1 : 2 ^ a ≤ q) (h2 : q < 2 ^ (a + 1)) : bitLength q = a + 1 := by
  have hq : q ≠ 0 := by have := Nat.one_le_two_pow (n := a); omega
  unfold bitLength
  rw [ite_eq_right_iff.2 (fun h => absurd h hq)]
  have := (Nat.le_log2 hq).2 h1
  have := (Nat.log2_lt hq).2 h2
  omega

theorem stepWord_fields {sb E r : ℕ} (hs : sb ≤ 1) (hE : E < 255) (hr : r < 2 ^ 23) (h0 : E * 2 ^ 23 + r ≠ 0) :
    StepWord (sb * 2 ^ 31 + E * 2 ^ 23 + r) := by
  have hlt : E * 2 ^ 23 + r < 2 ^ 31 := by
    calc E * 2 ^ 23 + r < 255 * 2 ^ 23 + 2 ^ 23 := by nlinarith
      _ ≤ 2 ^ 31 := by norm_num
  refine ⟨?_, ?_, ?_⟩
  · have : sb * 2 ^ 31 ≤ 2 ^ 31 := by nlinarith
    have h : (2 : ℕ) ^ 32 = 2 ^ 31 + 2 ^ 31 := by norm_num
    linarith
  · rw [show (0x80000000 : ℕ) = 2 ^ 31 by norm_num]
    intro h
    rcases Nat.le_one_iff_eq_zero_or_eq_one.1 hs with rfl | rfl <;> omega
  · unfold PeelExact.finite?
    have : (sb * 2 ^ 31 + E * 2 ^ 23 + r) >>> 23 % 256 = E := by
      rw [Nat.shiftRight_eq_div_pow, show sb * 2 ^ 31 + E * 2 ^ 23 + r = r + (sb * 256 + E) * 2 ^ 23 by ring,
        Nat.add_mul_div_right _ _ (by norm_num), Nat.div_eq_of_lt hr, zero_add]
      omega
    rw [this, ite_eq_right_iff.2 (fun h => absurd h (by omega))]

theorem or_sign {bits : ℕ} (hb : bits < 2 ^ 31) (P : Prop) [Decidable P] :
    (bits ||| (if P then 0x80000000 else 0)) = (if P then 1 else 0) * 2 ^ 31 + bits := by
  by_cases hP : P
  · have := Nat.shiftLeft_add_eq_or_of_lt hb 1
    simp only [Nat.one_shiftLeft] at this
    simp only [hP, ite_true, one_mul]
    rw [Nat.lor_comm, show (0x80000000 : ℕ) = 2 ^ 31 by norm_num, ← this]
  · simp [hP]

theorem shift_or (E Q : ℕ) : (E <<< 23 ||| (Q &&& 0x7FFFFF)) = E * 2 ^ 23 + Q % 2 ^ 23 := by
  rw [show (0x7FFFFF : ℕ) = 2 ^ 23 - 1 by norm_num, Nat.and_two_pow_sub_one_eq_mod,
    ← Nat.shiftLeft_add_eq_or_of_lt (Nat.mod_lt _ (by norm_num)), Nat.shiftLeft_eq]

theorem natAbs_cast (N : ℤ) : ((N.natAbs : ℕ) : ℚ) = |(N : ℚ)| := by
  rw [Nat.cast_natAbs, Int.cast_abs]

/-- **`toF32RZ` is exact on what binary32 holds**: a nonzero `N·2^lsb` that `Fits` is written as a word of exactly that
value, a word as a step leaves it (below `2^32`, not `−0`, finite). -/
theorem toF32RZ_exact {N lsb : ℤ} (hN : N ≠ 0) (hfit : Fits ((N : ℚ) * 2 ^ lsb)) :
    ∃ w, toF32RZ N lsb = some w ∧ StepWord w ∧ wordQ w = some ((N : ℚ) * 2 ^ lsb) := by
  have two : (1 : ℚ) < 2 := by norm_num
  set v : ℚ := (N : ℚ) * 2 ^ lsb with hv
  have hv0 : v ≠ 0 := mul_ne_zero (by exact_mod_cast hN) (zpow_ne_zero _ two_ne_zero)
  obtain ⟨m, e, hm, he, hve, hvb⟩ := hfit.resolve_left hv0
  set M := N.natAbs with hM
  have hM0 : M ≠ 0 := Int.natAbs_ne_zero.2 hN
  set L := M.log2 with hL
  have habs : |v| = (M : ℚ) * 2 ^ lsb := by
    rw [hv, abs_mul, abs_of_pos (zpow_pos (show (0 : ℚ) < 2 by norm_num) _), hM, natAbs_cast]
  have hlo : (2 : ℚ) ^ ((L : ℤ) + lsb) ≤ |v| := by
    rw [habs, zpow_add₀ two_ne_zero, zpow_natCast]
    exact mul_le_mul_of_nonneg_right (by exact_mod_cast Nat.log2_self_le hM0) (zpow_pos (show (0 : ℚ) < 2 by norm_num) _).le
  have hhi : |v| < (2 : ℚ) ^ ((L : ℤ) + 1 + lsb) := by
    rw [habs, zpow_add₀ two_ne_zero, show ((L : ℤ) + 1) = ((L + 1 : ℕ) : ℤ) by push_cast; ring, zpow_natCast]
    exact mul_lt_mul_of_pos_right (by exact_mod_cast Nat.lt_log2_self) (zpow_pos (show (0 : ℚ) < 2 by norm_num) _)
  set eV : ℤ := (L : ℤ) + lsb with heV
  have hm0 : m ≠ 0 := by rintro rfl; simp at hve; exact hv0 hve
  have habsm : |v| = |(m : ℚ)| * 2 ^ e := by rw [hve, abs_mul, abs_of_pos (zpow_pos (show (0 : ℚ) < 2 by norm_num) _)]
  have hme : eV < 24 + e := by
    have h1 : |v| < (2 : ℚ) ^ (24 + e) := by
      rw [habsm, zpow_add₀ two_ne_zero, show ((24 : ℤ)) = ((24 : ℕ) : ℤ) by rfl, zpow_natCast]
      exact mul_lt_mul_of_pos_right (by exact_mod_cast hm) (zpow_pos (show (0 : ℚ) < 2 by norm_num) _)
    exact (zpow_lt_zpow_iff_right₀ two).1 (lt_of_le_of_lt hlo h1)
  have hmax : eV < 128 := by
    have : (2 : ℚ) ^ eV < 2 ^ (128 : ℤ) := lt_of_le_of_lt hlo (by simpa using hvb)
    exact (zpow_lt_zpow_iff_right₀ two).1 this
  have hmin : -150 < eV := by
    have h1 : (2 : ℚ) ^ e ≤ |v| := by
      rw [habsm]
      have : (1 : ℚ) ≤ |(m : ℚ)| := by
        rw [← Int.cast_abs]; exact_mod_cast Int.one_le_abs hm0
      nlinarith [zpow_pos (show (0 : ℚ) < 2 by norm_num) e]
    have : (2 : ℚ) ^ e < 2 ^ (eV + 1) := lt_of_le_of_lt h1 (by rw [heV]; convert hhi using 2; ring)
    have := (zpow_lt_zpow_iff_right₀ two).1 this
    omega
  set g : ℤ := if -126 ≤ eV then eV - 23 else -149 with hg
  have hge : g ≤ e := by rw [hg]; split_ifs <;> omega
  have hgV : g ≤ eV := by rw [hg]; split_ifs <;> omega
  set Q : ℕ := m.natAbs * 2 ^ (e - g).toNat with hQ
  have hQv : (M : ℚ) * 2 ^ lsb = (Q : ℚ) * 2 ^ g := by
    rw [← habs, habsm, hQ]
    push_cast
    rw [natAbs_cast, mul_assoc, ← zpow_natCast, ← zpow_add₀ two_ne_zero, Int.toNat_of_nonneg (by omega)]
    ring_nf
  have hQ0 : Q ≠ 0 := by
    rw [hQ]; exact Nat.mul_ne_zero (Int.natAbs_ne_zero.2 hm0) (by positivity)
  have hgpos : (0 : ℚ) < 2 ^ g := zpow_pos (show (0 : ℚ) < 2 by norm_num) _
  have hQabs : (Q : ℚ) * 2 ^ g = |v| := by rw [← hQv, habs]
  set a : ℕ := (eV - g).toNat with ha
  have haz : (a : ℤ) = eV - g := Int.toNat_of_nonneg (by omega)
  have hQlo : 2 ^ a ≤ Q := by
    have h1 : (2 : ℚ) ^ (a : ℤ) * 2 ^ g ≤ Q * 2 ^ g := by
      rw [hQabs, ← zpow_add₀ two_ne_zero, haz, show eV - g + g = eV by ring]; exact hlo
    have h2 := le_of_mul_le_mul_right h1 hgpos
    rw [zpow_natCast] at h2
    exact_mod_cast h2
  have hQhi : Q < 2 ^ (a + 1) := by
    have h1 : (Q : ℚ) * 2 ^ g < 2 ^ ((a : ℤ) + 1) * 2 ^ g := by
      rw [hQabs, ← zpow_add₀ two_ne_zero, haz, show eV - g + 1 + g = (L : ℤ) + 1 + lsb by rw [heV]; ring]
      exact hhi
    have h2 := lt_of_mul_lt_mul_right h1 hgpos.le
    rw [show ((a : ℤ) + 1) = ((a + 1 : ℕ) : ℤ) by push_cast; ring, zpow_natCast] at h2
    exact_mod_cast h2
  have hbQ : bitLength Q = a + 1 := bitLength_of_bounds hQlo hQhi
  have hbM : ((bitLength M : ℕ) : ℤ) - 1 + lsb = eV := by
    rw [bitLength_of_bounds (Nat.log2_self_le hM0) Nat.lt_log2_self, heV]; push_cast; ring
  have hq : (if 0 < g - lsb then M >>> (g - lsb).toNat else M <<< (-(g - lsb)).toNat) = Q := by
    split_ifs with hs
    · have hsz : (((g - lsb).toNat : ℕ) : ℤ) = g - lsb := Int.toNat_of_nonneg hs.le
      have hMQ : (M : ℚ) = ((Q * 2 ^ (g - lsb).toNat : ℕ) : ℚ) := by
        push_cast
        rw [← zpow_natCast, hsz]
        have := hQv
        rw [show g = (g - lsb) + lsb by ring, zpow_add₀ two_ne_zero, ← mul_assoc] at this
        exact mul_right_cancel₀ (zpow_ne_zero _ two_ne_zero) this
      have hMQ' : M = Q * 2 ^ (g - lsb).toNat := by exact_mod_cast hMQ
      rw [Nat.shiftRight_eq_div_pow, hMQ', Nat.mul_div_cancel _ (by positivity)]
    · have hsz : (((-(g - lsb)).toNat : ℕ) : ℤ) = lsb - g := by rw [Int.toNat_of_nonneg (by omega)]; ring
      have hMQ : ((M * 2 ^ (-(g - lsb)).toNat : ℕ) : ℚ) = (Q : ℚ) := by
        push_cast
        rw [← zpow_natCast, hsz]
        have := hQv
        rw [show lsb = (lsb - g) + g by ring, zpow_add₀ two_ne_zero, ← mul_assoc] at this
        exact mul_right_cancel₀ (zpow_ne_zero _ two_ne_zero) this
      rw [Nat.shiftLeft_eq]; exact_mod_cast hMQ
  have hNM : (N : ℚ) = (if N < 0 then -1 else 1) * (M : ℚ) := by
    rw [hM, natAbs_cast]
    split_ifs with hn
    · rw [abs_of_neg (by exact_mod_cast hn)]; ring
    · rw [abs_of_nonneg (by exact_mod_cast (not_lt.1 hn))]; ring
  have hvQ : v = (if N < 0 then -1 else 1) * ((Q : ℚ) * 2 ^ g) := by
    rw [hv, hNM, mul_assoc, hQv]
  have hsb : (if N < 0 then 1 else 0) ≤ 1 := by split_ifs <;> omega
  unfold toF32RZ
  simp only []
  rw [← hM, hbM, ← hg, hq, ite_eq_right_iff.2 (fun h => absurd h hQ0), hbQ,
    show ((a + 1 : ℕ) : ℤ) - 1 + g = eV by push_cast; omega, ite_eq_right_iff.2 (fun h => absurd h (by omega))]
  by_cases hn : -126 ≤ eV
  · have hg' : g = eV - 23 := by rw [hg, ite_eq_left hn]
    have ha23 : a = 23 := by omega
    rw [ha23] at hQlo hQhi
    set E : ℕ := (eV + 127).toNat with hE
    have hEz : (E : ℤ) = eV + 127 := Int.toNat_of_nonneg (by omega)
    have hr : Q % 2 ^ 23 < 2 ^ 23 := Nat.mod_lt _ (by norm_num)
    have hQr : 2 ^ 23 + Q % 2 ^ 23 = Q := by omega
    have hE255 : E < 255 := by omega
    have hE0 : E ≠ 0 := by omega
    have hEr : E * 2 ^ 23 + Q % 2 ^ 23 ≠ 0 := by
      intro h; exact hE0 (by nlinarith [Nat.zero_le (Q % 2 ^ 23)])
    have hbits : E * 2 ^ 23 + Q % 2 ^ 23 < 2 ^ 31 := by
      calc E * 2 ^ 23 + Q % 2 ^ 23 < 254 * 2 ^ 23 + 2 ^ 23 := by nlinarith
        _ ≤ 2 ^ 31 := by norm_num
    rw [ite_eq_left hn, shift_or, or_sign hbits, ← add_assoc]
    refine ⟨_, rfl, stepWord_fields hsb hE255 hr hEr, ?_⟩
    unfold wordQ
    rw [AtomErr.fp32Value_word _ _ _ hsb hE255 hr, ite_eq_right hE0, hQr, hvQ, hg']
    simp only [Value.finite?, Option.map_some, Term.value, hEz]
    congr 1
    rw [show eV + 127 - 127 - 23 = eV - 23 by ring]
    by_cases hN0 : N < 0
    · rw [ite_eq_left hN0, ite_eq_left hN0, ite_eq_left (by rfl)]; ring
    · rw [ite_eq_right hN0, ite_eq_right hN0, ite_eq_right (by decide)]; ring
  · have hg' : g = -149 := by rw [hg, ite_eq_right hn]
    have hQ23 : Q < 2 ^ 23 := lt_of_lt_of_le hQhi (Nat.pow_le_pow_right (by norm_num) (by omega))
    have hbits : 0 * 2 ^ 23 + Q < 2 ^ 31 := by
      calc 0 * 2 ^ 23 + Q < 2 ^ 23 := by rw [zero_mul, zero_add]; exact hQ23
        _ ≤ 2 ^ 31 := by norm_num
    rw [ite_eq_right hn, show Q = 0 * 2 ^ 23 + Q by ring, or_sign hbits]
    refine ⟨_, rfl, ?_, ?_⟩
    · rw [← add_assoc]; exact stepWord_fields hsb (by norm_num) hQ23 (by omega)
    · unfold wordQ
      rw [← add_assoc, AtomErr.fp32Value_word _ _ _ hsb (by norm_num) hQ23, ite_eq_left rfl, ite_eq_right hQ0, hvQ, hg']
      simp only [Value.finite?, Option.map_some, Term.value]
      congr 1
      rw [show (-126 - 23 : ℤ) = -149 by norm_num]
      by_cases hN0 : N < 0
      · rw [ite_eq_left hN0, ite_eq_left hN0, ite_eq_left (by rfl)]; ring
      · rw [ite_eq_right hN0, ite_eq_right hN0, ite_eq_right (by decide)]; ring

/-! ## One step -/

/-- The exact value of group terms: `Σ mant·2^unit`. -/
def termsValue (ts : List GroupTerm) : ℚ := (ts.map fun t => (t.mant : ℚ) * 2 ^ t.unit).sum

/-- The step's grid anchors on group terms `ts` and the accumulator term `c` (`step`'s `anchors`): `X_g − fracBits` for
each participating group, and `E_acc − (accWindow − 1)` if the accumulator is nonzero. -/
def anchors (p : Fp4Atom) (ts : List GroupTerm) (c : Term) : List ℤ :=
  ts.filterMap (fun t => if t.live then some (t.unit - p.fracBits) else none) ++
    (if c.mag = 0 then [] else [c.exp - ((p.accWindow - 1 : Nat) : Int)])

/-- The step's grid exponent `lsb`: the largest anchor, and at least the grid floor (the floor if there is no anchor,
where the step writes `+0`). -/
def gridLsb (p : Fp4Atom) (ts : List GroupTerm) (c : Term) : ℤ :=
  match anchors p ts c with
  | [] => p.gridFloor
  | x :: xs => max (xs.foldl max x) p.gridFloor

/-- **No addend loses a bit to the grid** (`fp4_census.py`'s step without `glost` or `alost`): every group's scaled dot
`mant·2^unit` and the accumulator's significand `mag·2^(exp − 23)` are multiples of `2^lsb`. -/
structure NoLoss (p : Fp4Atom) (ts : List GroupTerm) (c : Term) : Prop where
  groups : ∀ h ∈ ts, h.unit < gridLsb p ts c → (2 ^ (gridLsb p ts c - h.unit).toNat : ℤ) ∣ h.mant
  acc : c.exp - 23 < gridLsb p ts c → (2 ^ (gridLsb p ts c - (c.exp - 23)).toNat : ℤ) ∣ c.mag

/-- **The windows' reading of `NoLoss`**, sufficient but not necessary (it ignores trailing zero bits, such as UE8M0's
mantissa 8): each nonzero group's unit weight is at most `fracBits` (27) below every participating group's and at most
`accWindow − 1` (35) below a nonzero accumulator's exponent, a nonzero accumulator's ulp is at most `fracBits` below
every participating group's unit weight, and both are at or above the grid floor. -/
structure OnGrid (p : Fp4Atom) (ts : List GroupTerm) (c : Term) : Prop where
  groups : ∀ g ∈ ts, ∀ h ∈ ts, g.live = true → h.mant ≠ 0 → g.unit - p.fracBits ≤ h.unit
  acc_groups : c.mag ≠ 0 → ∀ h ∈ ts, h.mant ≠ 0 → c.exp - ((p.accWindow - 1 : ℕ) : ℤ) ≤ h.unit
  groups_acc : c.mag ≠ 0 → ∀ g ∈ ts, g.live = true → g.unit - p.fracBits ≤ c.exp - 23
  floor_groups : ∀ h ∈ ts, h.mant ≠ 0 → p.gridFloor ≤ h.unit
  floor_acc : c.mag ≠ 0 → p.gridFloor ≤ c.exp - 23

theorem mem_of_mapM {α β : Type} {f : α → Option β} :
    ∀ {l : List α} {l' : List β}, l.mapM f = some l' → ∀ y ∈ l', ∃ x ∈ l, f x = some y
  | [], l', h, y, hy => by
    simp only [List.mapM_nil, pure, Option.some.injEq] at h; subst h; cases hy
  | x :: xs, l', h, y, hy => by
    rw [List.mapM_cons] at h
    cases hx : f x with
    | none => simp [hx] at h
    | some z =>
      cases hxs : xs.mapM f with
      | none => simp [hx, hxs] at h
      | some zs =>
        simp only [hx, hxs, bind, Option.bind, pure, Option.some.injEq] at h
        subst h
        rcases List.mem_cons.1 hy with rfl | hy
        · exact ⟨x, List.mem_cons_self, hx⟩
        · obtain ⟨u, hu, hfu⟩ := mem_of_mapM hxs y hy
          exact ⟨u, List.mem_cons_of_mem _ hu, hfu⟩

theorem groupTerm_dead {p : Fp4Atom} {a b : List Nib} {sa sb : List (Fin 256)} {j : ℕ} {t : GroupTerm}
    (h : p.groupTerm a b sa sb j = some t) (hl : t.live = false) : t.mant = 0 := by
  unfold Fp4Atom.groupTerm at h
  cases hsa : sa[j * p.group / p.block]? with
  | none => simp [hsa] at h
  | some x =>
  cases hsb : sb[j * p.group / p.block]? with
  | none => simp [hsa, hsb] at h
  | some y =>
  cases hx : scaleDyadic p.fmt x with
  | none => simp [hsa, hsb, hx] at h
  | some dx =>
  cases hy : scaleDyadic p.fmt y with
  | none => simp [hsa, hsb, hy] at h
  | some dy =>
  simp only [hsa, hsb, hx, hy, bind, Option.bind, pure, Option.some.injEq] at h
  subst h
  simp only [Bool.and_eq_false_iff, bne_eq_false_iff_eq] at hl
  rcases hl with (hl | hl) | hl
  · have : ∀ z ∈ ((((a.drop (j * p.group)).take p.group).zip ((b.drop (j * p.group)).take p.group)).map
        fun x => numerator x.1 * numerator x.2), z = 0 := by
      intro z hz
      have := List.any_eq_false.1 hl z hz
      simpa using this
    simp [List.sum_eq_zero this]
  · simp [hl]
  · simp [hl]

theorem groupTerms_dead {p : Fp4Atom} {a b : List Nib} {sa sb : List (Fin 256)} {ts : List GroupTerm}
    (h : p.groupTerms a b sa sb = some ts) : ∀ t ∈ ts, t.live = false → t.mant = 0 := by
  intro t ht hl
  unfold Fp4Atom.groupTerms at h
  split_ifs at h
  obtain ⟨j, -, hj⟩ := mem_of_mapM h t ht
  exact groupTerm_dead hj hl

theorem foldl_max_le {B : ℤ} : ∀ (xs : List ℤ) (x : ℤ), x ≤ B → (∀ y ∈ xs, y ≤ B) → xs.foldl max x ≤ B
  | [], _, hx, _ => hx
  | y :: ys, x, hx, hys => foldl_max_le ys (max x y) (max_le hx (hys y List.mem_cons_self))
      fun z hz => hys z (List.mem_cons_of_mem _ hz)

theorem gridLsb_le {p : Fp4Atom} {ts : List GroupTerm} {c : Term} {B : ℤ} (hA : ∀ y ∈ anchors p ts c, y ≤ B)
    (hF : p.gridFloor ≤ B) : gridLsb p ts c ≤ B := by
  unfold gridLsb
  split
  · exact hF
  · rename_i x xs h
    rw [h] at hA
    exact max_le (foldl_max_le xs x (hA x List.mem_cons_self) fun y hy => hA y (List.mem_cons_of_mem _ hy)) hF

theorem anchors_le {p : Fp4Atom} {ts : List GroupTerm} {c : Term} (hw : 24 ≤ p.accWindow) (hg : OnGrid p ts c) :
    ∀ y ∈ anchors p ts c, (∀ t ∈ ts, t.mant ≠ 0 → y ≤ t.unit) ∧ (c.mag ≠ 0 → y ≤ c.exp - 23) := by
  have hw' : (23 : ℤ) ≤ ((p.accWindow - 1 : ℕ) : ℤ) := by omega
  intro y hy
  unfold anchors at hy
  rw [List.mem_append] at hy
  rcases hy with hy | hy
  · obtain ⟨g, hgm, hgy⟩ := List.mem_filterMap.1 hy
    cases hl : g.live
    · simp [hl] at hgy
    · simp only [hl, ite_true, Option.some.injEq] at hgy
      subst hgy
      exact ⟨fun t ht hm => hg.groups g hgm t ht hl hm, fun h0 => hg.groups_acc h0 g hgm hl⟩
  · by_cases h0 : c.mag = 0
    · rw [ite_eq_left h0] at hy; cases hy
    · rw [ite_eq_right h0, List.mem_singleton] at hy
      subst hy
      exact ⟨fun t ht hm => hg.acc_groups h0 t ht hm, fun _ => by omega⟩

/-- `OnGrid` gives `NoLoss` when the accumulator window is at least binary32's 24 bits. -/
theorem OnGrid.noLoss {p : Fp4Atom} {ts : List GroupTerm} {c : Term} (hw : 24 ≤ p.accWindow) (hg : OnGrid p ts c) :
    NoLoss p ts c where
  groups h hh hlt := by
    by_cases hm : h.mant = 0
    · rw [hm]; exact dvd_zero _
    · exact absurd hlt (not_lt.2 (gridLsb_le (fun y hy => (anchors_le hw hg y hy).1 h hh hm) (hg.floor_groups h hh hm)))
  acc hlt := by
    by_cases hm : c.mag = 0
    · rw [hm]; exact dvd_zero _
    · exact absurd hlt (not_lt.2 (gridLsb_le (fun y hy => (anchors_le hw hg y hy).2 hm) (hg.floor_acc hm)))

theorem truncRZ_exact {m e lsb : ℤ} (h : e < lsb → (2 ^ (lsb - e).toNat : ℤ) ∣ m) :
    (truncRZ m e lsb : ℚ) * 2 ^ lsb = m * 2 ^ e := by
  unfold truncRZ
  split_ifs with hle hneg
  · push_cast
    rw [mul_assoc, ← zpow_natCast, ← zpow_add₀ two_ne_zero, Int.toNat_of_nonneg (by linarith)]
    ring_nf
  all_goals
    obtain ⟨q, hq⟩ := h (not_le.1 hle)
    set s := (lsb - e).toNat with hs
    have hl : lsb = (s : ℤ) + e := by omega
    have hpow : (2 : ℚ) ^ lsb = 2 ^ s * 2 ^ e := by rw [hl, zpow_add₀ two_ne_zero, zpow_natCast]
    have habs : m.natAbs >>> s = q.natAbs := by
      rw [Nat.shiftRight_eq_div_pow, hq, Int.natAbs_mul, Int.natAbs_pow]
      exact Nat.mul_div_cancel_left _ (Nat.two_pow_pos s)
    have hp : (0 : ℤ) < 2 ^ s := by positivity
    rw [habs, hpow, hq]
  · have hq0 : q ≤ 0 := by
      by_contra h'
      have := mul_pos hp (lt_of_not_ge h')
      linarith
    rw [Int.ofNat_natAbs_of_nonpos hq0]
    push_cast; ring
  · have hq0 : 0 ≤ q := by
      by_contra h'
      have := mul_neg_of_pos_of_neg hp (lt_of_not_ge h')
      linarith
    rw [Int.natAbs_of_nonneg hq0]
    push_cast; ring

theorem sum_truncRZ {lsb : ℤ} : ∀ {ts : List GroupTerm},
    (∀ t ∈ ts, t.unit < lsb → (2 ^ (lsb - t.unit).toNat : ℤ) ∣ t.mant) →
    (((ts.map fun t => truncRZ t.mant t.unit lsb).sum : ℤ) : ℚ) * 2 ^ lsb = termsValue ts
  | [], _ => by simp [termsValue]
  | t :: ts, h => by
    have ih := sum_truncRZ (ts := ts) fun u hu => h u (List.mem_cons_of_mem _ hu)
    unfold termsValue at ih ⊢
    simp only [List.map_cons, List.sum_cons] at ih ⊢
    push_cast at ih ⊢
    rw [add_mul, ih, truncRZ_exact (h t List.mem_cons_self)]

theorem term_value_signed (c : Term) :
    ((((if c.neg then -(c.mag : ℤ) else c.mag) : ℤ) : ℚ)) * 2 ^ (c.exp - 23) = c.value := by
  unfold Term.value; split_ifs <;> push_cast <;> ring

theorem stepWord_zero : StepWord 0 := ⟨by norm_num, by norm_num, by decide⟩

theorem wordQ_zero : wordQ 0 = some 0 := by simp [wordQ, fp32Value, Term.zero, Term.value, Value.finite?]

/-- **One exact step**: on group terms `ts` and a finite accumulator term `c`, a step on which no addend loses a bit to
the grid and whose exact result `Fits` binary32 writes exactly `c + Σ mant·2^unit`, as a step word. -/
theorem step_exact (p : Fp4Atom) {acc : Nat} {a b : List Nib} {sa sb : List (Fin 256)} {ts : List GroupTerm}
    {c : Term} (hts : p.groupTerms a b sa sb = some ts) (hc : fp32Value acc = .fin c) (hg : NoLoss p ts c)
    (hfit : Fits (c.value + termsValue ts)) :
    ∃ d, p.step acc a b sa sb = some d ∧ StepWord d ∧ wordQ d = some (c.value + termsValue ts) := by
  have hdead := groupTerms_dead hts
  unfold Fp4Atom.step
  simp only [hts, hc, Value.finite?, bind, Option.bind]
  split
  · rename_i hnil
    rw [List.append_eq_nil_iff] at hnil
    obtain ⟨hf, ha⟩ := hnil
    have hc0 : c.mag = 0 := by by_contra h; rw [ite_eq_right h] at ha; cases ha
    have hmz : ∀ t ∈ ts, t.mant = 0 := by
      intro t ht
      cases hl : t.live
      · exact hdead t ht hl
      · have : t.unit - p.fracBits ∈ ts.filterMap (fun t => if t.live then some (t.unit - p.fracBits) else none) :=
          List.mem_filterMap.2 ⟨t, ht, by simp [hl]⟩
        rw [hf] at this; cases this
    have hv : c.value + termsValue ts = 0 := by
      have : termsValue ts = 0 := by
        unfold termsValue
        exact List.sum_eq_zero fun x hx => by
          obtain ⟨t, ht, rfl⟩ := List.mem_map.1 hx
          simp [hmz t ht]
      rw [this]; simp [Term.value, hc0]
    exact ⟨0, rfl, stepWord_zero, by rw [hv, wordQ_zero]⟩
  · rename_i x xs hcons
    set lsb := max (xs.foldl max x) p.gridFloor with hlsb
    have hL : gridLsb p ts c = lsb := by
      unfold gridLsb anchors
      rw [hcons]
    have hcm : (truncRZ (if c.neg then -(c.mag : ℤ) else c.mag) (c.exp - 23) lsb : ℚ) * 2 ^ lsb = c.value := by
      rw [truncRZ_exact, term_value_signed]
      intro hlt
      have := hg.acc (by rw [hL]; exact hlt)
      rw [hL] at this
      split_ifs
      · exact (Int.dvd_neg).2 this
      · exact this
    set total := (ts.map fun t => truncRZ t.mant t.unit lsb).sum +
      truncRZ (if c.neg then -(c.mag : ℤ) else c.mag) (c.exp - 23) lsb with htotal
    have hval : (total : ℚ) * 2 ^ lsb = c.value + termsValue ts := by
      rw [htotal]; push_cast
      rw [add_mul, ← Int.cast_list_sum, sum_truncRZ (fun t ht hlt => by
        have := hg.groups t ht (by rw [hL]; exact hlt)
        rwa [hL] at this), hcm, add_comm]
    split_ifs with ht0
    · refine ⟨0, rfl, stepWord_zero, ?_⟩
      rw [← hval, ht0]; simp [wordQ_zero]
    · rw [← hval] at hfit ⊢
      exact toF32RZ_exact ht0 hfit

/-! ## The chain -/

/-- The `u`-th slice of width `w`. -/
def slice {α : Type} (l : List α) (u w : Nat) : List α := (l.drop (u * w)).take w

theorem slice_zero {α : Type} (l : List α) (w : Nat) : slice l 0 w = l.take w := by simp [slice]

theorem slice_drop {α : Type} (l : List α) (u w : Nat) : slice (l.drop w) u w = slice l (u + 1) w := by
  simp only [slice, List.drop_drop]
  congr 2
  ring

/-- **Atom `u`'s exact contribution**: `Σ mant·2^unit` over its group terms, the exact sum of its scaled products; 0
outside the model. -/
def atomVal (p : Fp4Atom) (a b : List Nib) (sa sb : List (Fin 256)) (u : Nat) : ℚ :=
  ((p.groupTerms (slice a u p.k) (slice b u p.k) (slice sa u p.scales) (slice sb u p.scales)).map termsValue).getD 0

theorem atomVal_drop (p : Fp4Atom) (a b : List Nib) (sa sb : List (Fin 256)) (u : Nat) :
    atomVal p (a.drop p.k) (b.drop p.k) (sa.drop p.scales) (sb.drop p.scales) u = atomVal p a b sa sb (u + 1) := by
  simp only [atomVal, slice_drop]

/-- The exact partial sum ahead of atom `u` on the run with skip mask `ms` from the value `v`: `v` plus every earlier
unskipped atom's exact contribution. -/
def partialSum (p : Fp4Atom) (ms : List Bool) (v : ℚ) (a b : List Nib) (sa sb : List (Fin 256)) (u : Nat) : ℚ :=
  v + ∑ t ∈ Finset.range u, if ms.getD t false then 0 else atomVal p a b sa sb t

/-- **An exact step from the value `v`**: the firing is in the model with group terms `ts`, `v + Σ mant·2^unit`
`Fits` binary32, and the step word of value `v` meets the grid condition with `ts`. -/
def ExactAt (p : Fp4Atom) (v : ℚ) (a b : List Nib) (sa sb : List (Fin 256)) : Prop :=
  ∃ ts, p.groupTerms a b sa sb = some ts ∧ Fits (v + termsValue ts) ∧
    ∀ w c, StepWord w → wordQ w = some v → fp32Value w = .fin c → NoLoss p ts c

/-- **An exact run** of the first `n` atoms with skip mask `ms` from the value `v`: every atom the mask doesn't skip is
an exact step from its exact partial sum. -/
def ExactRun (p : Fp4Atom) (ms : List Bool) (v : ℚ) (a b : List Nib) (sa sb : List (Fin 256)) (n : Nat) : Prop :=
  ∀ u < n, ms.getD u false = false →
    ExactAt p (partialSum p ms v a b sa sb u) (slice a u p.k) (slice b u p.k) (slice sa u p.scales) (slice sb u p.scales)

/-- The lengths `chain` accepts. -/
def Shaped (p : Fp4Atom) (a b : List Nib) (sa sb : List (Fin 256)) : Prop :=
  b.length = a.length ∧ a.length % p.k = 0 ∧ sa.length = a.length / p.k * p.scales ∧
    sb.length = a.length / p.k * p.scales

theorem getD_tail (ms : List Bool) (t : Nat) : ms.tail.getD t false = ms.getD (t + 1) false := by
  cases ms <;> simp

theorem headD_eq_getD (ms : List Bool) : ms.headD false = ms.getD 0 false := by
  cases ms <;> simp

theorem partialSum_zero (p : Fp4Atom) (ms : List Bool) (v : ℚ) (a b : List Nib) (sa sb : List (Fin 256)) :
    partialSum p ms v a b sa sb 0 = v := by
  simp [partialSum]

theorem partialSum_succ (p : Fp4Atom) (ms : List Bool) (v : ℚ) (a b : List Nib) (sa sb : List (Fin 256)) (u : Nat) :
    partialSum p ms v a b sa sb (u + 1) =
      partialSum p ms.tail (v + if ms.headD false then 0 else atomVal p a b sa sb 0) (a.drop p.k) (b.drop p.k)
        (sa.drop p.scales) (sb.drop p.scales) u := by
  unfold partialSum
  rw [Finset.sum_range_succ']
  simp only [atomVal_drop, getD_tail, headD_eq_getD]
  ring

theorem exactRun_succ {p : Fp4Atom} {ms : List Bool} {v : ℚ} {a b : List Nib} {sa sb : List (Fin 256)} {n : Nat}
    (h : ExactRun p ms v a b sa sb (n + 1)) :
    ExactRun p ms.tail (v + if ms.headD false then 0 else atomVal p a b sa sb 0) (a.drop p.k) (b.drop p.k)
      (sa.drop p.scales) (sb.drop p.scales) n := by
  intro u hu hm
  rw [getD_tail] at hm
  have := h (u + 1) (by omega) hm
  rwa [partialSum_succ, ← slice_drop, ← slice_drop, ← slice_drop, ← slice_drop] at this

/-- **An exact masked run writes its exact partial sum**: from a step word of value `v`, `n` firings whose unskipped
steps are exact write the step word of value `v + Σ` of the unskipped atoms' exact contributions. -/
theorem chainMaskN_exact (p : Fp4Atom) :
    ∀ (n : Nat) (ms : List Bool) (c : Nat) (v : ℚ) (a b : List Nib) (sa sb : List (Fin 256)), StepWord c →
      wordQ c = some v → ExactRun p ms v a b sa sb n →
      ∃ w, chainMaskN p n ms c a b sa sb = some w ∧ StepWord w ∧ wordQ w = some (partialSum p ms v a b sa sb n)
  | 0, ms, c, v, a, b, sa, sb, hc, hv, _ => ⟨c, rfl, hc, by rw [partialSum_zero]; exact hv⟩
  | n + 1, ms, c, v, a, b, sa, sb, hc, hv, hr => by
    have hr' := exactRun_succ hr
    by_cases hm : ms.headD false = true
    · have hv' : wordQ c = some (v + if ms.headD false then 0 else atomVal p a b sa sb 0) := by
        rw [ite_eq_left hm, add_zero]; exact hv
      obtain ⟨w, h1, h2, h3⟩ := chainMaskN_exact p n ms.tail c _ _ _ _ _ hc hv' hr'
      refine ⟨w, ?_, h2, ?_⟩
      · simp only [chainMaskN, hm, ite_true]; exact h1
      · rw [partialSum_succ]; exact h3
    · have hm0 : ms.getD 0 false = false := by rw [← headD_eq_getD]; simpa using hm
      obtain ⟨ts, hts, hfit, hgrid⟩ := hr 0 (Nat.succ_pos n) hm0
      rw [partialSum_zero] at hfit hgrid
      simp only [slice_zero] at hts
      obtain ⟨c', hc', hcv⟩ := PeelExactProofs.wordQ_fin hv
      have hg := hgrid c c' hc hv hc'
      rw [← hcv] at hfit
      obtain ⟨d, hd, hdw, hdv⟩ := step_exact p hts hc' hg hfit
      have hav : atomVal p a b sa sb 0 = termsValue ts := by simp [atomVal, slice_zero, hts]
      have hv' : wordQ d = some (v + if ms.headD false then 0 else atomVal p a b sa sb 0) := by
        rw [ite_eq_right hm, hav, hdv, hcv]
      obtain ⟨w, h1, h2, h3⟩ := chainMaskN_exact p n ms.tail d _ _ _ _ _ hdw hv' hr'
      refine ⟨w, ?_, h2, ?_⟩
      · simp only [chainMaskN, hm, hd, Bool.false_eq_true, ite_false, bind, Option.bind]; exact h1
      · rw [partialSum_succ]; exact h3

theorem chainMaskN_nil (p : Fp4Atom) :
    ∀ (n c : Nat) (a b : List Nib) (sa sb : List (Fin 256)), chainMaskN p n [] c a b sa sb = p.chainN n c a b sa sb
  | 0, _, _, _, _, _ => rfl
  | n + 1, c, a, b, sa, sb => by
    simp only [chainMaskN, Fp4Atom.chainN, List.headD_nil, List.tail_nil, Bool.false_eq_true, ite_false,
      chainMaskN_nil p n]

/-- The empty mask is the chain. -/
theorem chainMask_nil (p : Fp4Atom) (c : Nat) (a b : List Nib) (sa sb : List (Fin 256)) :
    chainMask p [] c a b sa sb = p.chain c a b sa sb := by
  unfold chainMask Fp4Atom.chain
  simp only [chainMaskN_nil]

theorem chainMask_exact (p : Fp4Atom) {ms : List Bool} {c : Nat} {v : ℚ} {a b : List Nib}
    {sa sb : List (Fin 256)} (hs : Shaped p a b sa sb) (hc : StepWord c) (hv : wordQ c = some v)
    (hr : ExactRun p ms v a b sa sb (a.length / p.k)) :
    ∃ w, chainMask p ms c a b sa sb = some w ∧ StepWord w ∧
      wordQ w = some (partialSum p ms v a b sa sb (a.length / p.k)) := by
  unfold chainMask
  unfold Shaped at hs
  rw [ite_eq_left hs]
  exact chainMaskN_exact p _ ms c v a b sa sb hc hv hr

/-- **The exactness lemma**: on an exact run from `+0`, C̃ is the step word of the exact sum of the atoms' exact
contributions. -/
theorem cTilde_exact (p : Fp4Atom) {a b : List Nib} {sa sb : List (Fin 256)}
    (hs : Shaped p a b sa sb) (hr : ExactRun p [] 0 a b sa sb (a.length / p.k)) :
    ∃ w, cTilde p a b sa sb = some w ∧ StepWord w ∧
      wordQ w = some (∑ u ∈ Finset.range (a.length / p.k), atomVal p a b sa sb u) := by
  obtain ⟨w, h1, h2, h3⟩ := chainMask_exact p hs stepWord_zero wordQ_zero hr
  rw [chainMask_nil] at h1
  refine ⟨w, h1, h2, ?_⟩
  rw [h3]
  simp [partialSum]

/-! ## Term 2: the lone-skip flags -/

/-- **Term 2's flag is the lone skip's invariance**, unconditionally: atom `t` is flagged iff it is one of the chain's
atoms and the chain with atom `t` alone skipped writes the chain's word. So the debit is sound (it charges only atoms a
program may skip without changing the word) and complete (it charges every such atom). -/
theorem fp4PureFlags_iff (p : Fp4Atom) (c : Nat) (a b : List Nib) (sa sb : List (Fin 256)) (t : Nat) :
    fp4PureFlags p c a b sa sb t = true ↔
      t < a.length / p.k ∧ ∃ w, p.chain c a b sa sb = some w ∧ skipOne p t c a b sa sb = some w := by
  unfold fp4PureFlags
  simp only [Bool.and_eq_true, decide_eq_true_eq, Option.isSome_iff_exists]
  constructor
  · rintro ⟨⟨ht, w, hw⟩, hs⟩
    exact ⟨ht, w, hw, hs.trans hw⟩
  · rintro ⟨ht, w, hw, hs⟩
    exact ⟨⟨ht, w, hw⟩, hs.trans hw.symm⟩

/-- `fp4PureFlags_iff` on a unit's rows, from `+0`. -/
theorem unitPureFlags_iff {m n k s : Nat} (p : Fp4Atom) (Aq : Matrix (Fin m) (Fin k) Nib)
    (SA : Matrix (Fin m) (Fin s) (Fin 256)) (Bq : Matrix (Fin n) (Fin k) Nib) (SB : Matrix (Fin n) (Fin s) (Fin 256))
    (i : Fin m) (j : Fin n) (t : Nat) :
    unitPureFlags p Aq SA Bq SB i j t = true ↔
      t < k / p.k ∧ ∃ w, p.chain 0 (List.ofFn (Aq i)) (List.ofFn (Bq j)) (List.ofFn (SA i)) (List.ofFn (SB j)) = some w ∧
        skipOne p t 0 (List.ofFn (Aq i)) (List.ofFn (Bq j)) (List.ofFn (SA i)) (List.ofFn (SB j)) = some w := by
  unfold unitPureFlags
  rw [fp4PureFlags_iff, List.length_ofFn]

theorem loneMask_getD (t u : Nat) : (loneMask t).getD u false = decide (u = t) := by
  induction t generalizing u with
  | zero => cases u <;> simp [loneMask]
  | succ t ih =>
    cases u with
    | zero => simp [loneMask, List.replicate_succ]
    | succ u =>
      have : loneMask (t + 1) = false :: loneMask t := by simp [loneMask, List.replicate_succ]
      rw [this, List.getD_cons_succ, ih]
      simp

theorem partialSum_lone (p : Fp4Atom) (v : ℚ) (a b : List Nib) (sa sb : List (Fin 256)) {t n : Nat} (ht : t < n) :
    partialSum p [] v a b sa sb n = partialSum p (loneMask t) v a b sa sb n + atomVal p a b sa sb t := by
  unfold partialSum
  simp only [List.getD_nil, Bool.false_eq_true, ite_false, loneMask_getD, decide_eq_true_eq]
  have : ∀ u, atomVal p a b sa sb u = (if u = t then 0 else atomVal p a b sa sb u) +
      (if u = t then atomVal p a b sa sb u else 0) := by
    intro u; split_ifs <;> simp
  rw [Finset.sum_congr rfl fun u _ => this u, Finset.sum_add_distrib, Finset.sum_ite_eq',
    ite_eq_left (Finset.mem_range.2 ht)]
  ring

theorem stepWord_inj {w₁ w₂ : Nat} (h₁ : StepWord w₁) (h₂ : StepWord w₂) {y : ℚ} (hy₁ : wordQ w₁ = some y)
    (hy₂ : wordQ w₂ = some y) : w₁ = w₂ := by
  by_cases h0 : y = 0
  · subst h0
    rw [wordQ_zero_word h₁ hy₁, wordQ_zero_word h₂ hy₂]
  · rw [← f32Word_word h₁.1 (finite?_exp h₁.2.2) hy₁ h0, f32Word_word h₂.1 (finite?_exp h₂.2.2) hy₂ h0]

/-- **Under exactness, term 2 flags exactly the atoms that contribute 0**: when the chain and the chain with atom `t`
skipped are exact runs, atom `t` is flagged iff its exact contribution is 0. This covers the exact zero-sum atoms; a
frozen atom (a nonzero contribution the chain drops) is flagged by `fp4PureFlags_iff` outside exactness. -/
theorem fp4PureFlags_exact_iff (p : Fp4Atom) {c : Nat} {v : ℚ} {a b : List Nib}
    {sa sb : List (Fin 256)} {t : Nat} (hs : Shaped p a b sa sb) (hc : StepWord c) (hv : wordQ c = some v)
    (ht : t < a.length / p.k) (hfull : ExactRun p [] v a b sa sb (a.length / p.k))
    (hskip : ExactRun p (loneMask t) v a b sa sb (a.length / p.k)) :
    fp4PureFlags p c a b sa sb t = true ↔ atomVal p a b sa sb t = 0 := by
  obtain ⟨w₁, h1, hw₁, hv₁⟩ := chainMask_exact p hs hc hv hfull
  obtain ⟨w₂, h2, hw₂, hv₂⟩ := chainMask_exact p hs hc hv hskip
  rw [chainMask_nil] at h1
  rw [fp4PureFlags_iff, partialSum_lone p v a b sa sb ht] at *
  unfold skipOne
  rw [h1, h2]
  constructor
  · rintro ⟨-, w, hw, hw'⟩
    obtain rfl := Option.some.inj hw
    obtain rfl := Option.some.inj hw'
    have := hv₁.symm.trans hv₂
    have := Option.some.inj this
    linarith
  · intro h0
    rw [h0, add_zero] at hv₁
    exact ⟨ht, w₁, rfl, by rw [stepWord_inj hw₂ hw₁ hv₂ hv₁]⟩

/-! ## The exact dot product -/

theorem numerator_decode (c : Nib) : (numerator c : ℚ) = 2 * Fp4.decode c := by
  fin_cases c <;> norm_num [numerator, Fp4.decode, Nib.sign, Nib.sig, Nib.expField, Nib.manField, Nib.expOr1, Nib.lsbExp]

theorem sum_numerators (L : List (Nib × Nib)) :
    (((L.map fun x => numerator x.1 * numerator x.2).sum : ℤ) : ℚ) = 4 * (L.map fun x => Fp4.decode x.1 * Fp4.decode x.2).sum := by
  induction L with
  | nil => simp
  | cons x L ih =>
    simp only [List.map_cons, List.sum_cons, Int.cast_add, Int.cast_mul, numerator_decode, ih]
    ring

/-- A scale byte's value, 0 if it is absent or invalid. -/
def scaleAt (f : ScaleFmt) (s : List (Fin 256)) (i : Nat) : ℚ := ((s[i]?).bind (scaleValue f)).getD 0

/-- **Group `j`'s exact dot**: the sum of its code pairs' exact products, times its block's scale on each side (a group
lies in one block). -/
def groupDot (p : Fp4Atom) (a b : List Nib) (sa sb : List (Fin 256)) (j : Nat) : ℚ :=
  (((slice a j p.group).zip (slice b j p.group)).map fun x => Fp4.decode x.1 * Fp4.decode x.2).sum *
    scaleAt p.fmt sa (j * p.group / p.block) * scaleAt p.fmt sb (j * p.group / p.block)

/-- A group term's value is the group's exact dot. -/
theorem groupTerm_value {p : Fp4Atom} {a b : List Nib} {sa sb : List (Fin 256)} {j : ℕ} {t : GroupTerm}
    (h : p.groupTerm a b sa sb j = some t) : (t.mant : ℚ) * 2 ^ t.unit = groupDot p a b sa sb j := by
  unfold Fp4Atom.groupTerm at h
  cases hsa : sa[j * p.group / p.block]? with
  | none => simp [hsa] at h
  | some x =>
  cases hsb : sb[j * p.group / p.block]? with
  | none => simp [hsa, hsb] at h
  | some y =>
  cases hx : scaleDyadic p.fmt x with
  | none => simp [hsa, hsb, hx] at h
  | some dx =>
  cases hy : scaleDyadic p.fmt y with
  | none => simp [hsa, hsb, hy] at h
  | some dy =>
  obtain ⟨ma, ea⟩ := dx
  obtain ⟨mb, eb⟩ := dy
  simp only [hsa, hsb, hx, hy, bind, Option.bind, pure, Option.some.injEq] at h
  subst h
  simp only [groupDot, scaleAt, slice, hsa, hsb, Option.bind_some, scaleValue, hx, hy, Option.map_some,
    Option.getD_some]
  rw [Int.cast_mul, Int.cast_mul, sum_numerators, zpow_add₀ two_ne_zero, zpow_add₀ two_ne_zero]
  push_cast
  norm_num
  ring

theorem termsValue_mapM {f : ℕ → Option GroupTerm} {G : ℕ → ℚ}
    (hf : ∀ j t, f j = some t → (t.mant : ℚ) * 2 ^ t.unit = G j) :
    ∀ {l : List ℕ} {ts : List GroupTerm}, l.mapM f = some ts → termsValue ts = (l.map G).sum
  | [], ts, h => by
    simp only [List.mapM_nil, pure, Option.some.injEq] at h; subst h; simp [termsValue]
  | x :: xs, ts, h => by
    rw [List.mapM_cons] at h
    cases hx : f x with
    | none => simp [hx] at h
    | some z =>
      cases hxs : xs.mapM f with
      | none => simp [hx, hxs] at h
      | some zs =>
        simp only [hx, hxs, bind, Option.bind, pure, Option.some.injEq] at h
        subst h
        have ih := termsValue_mapM hf hxs
        unfold termsValue at ih ⊢
        simp only [List.map_cons, List.sum_cons, ih, hf x z hx]

theorem sum_list_range (G : ℕ → ℚ) (n : ℕ) : ((List.range n).map G).sum = ∑ j ∈ Finset.range n, G j := by
  induction n with
  | zero => simp
  | succ n ih => rw [List.range_succ, List.map_append, List.sum_append, ih, Finset.sum_range_succ]; simp

/-- A firing's group terms sum to its exact dot, group by group. -/
theorem groupTerms_value {p : Fp4Atom} {a b : List Nib} {sa sb : List (Fin 256)} {ts : List GroupTerm}
    (h : p.groupTerms a b sa sb = some ts) :
    termsValue ts = ∑ j ∈ Finset.range (p.k / p.group), groupDot p a b sa sb j := by
  unfold Fp4Atom.groupTerms at h
  split_ifs at h
  rw [termsValue_mapM (fun j t ht => groupTerm_value ht) h, sum_list_range]

/-- **Atom `u`'s exact dot**: the sum of its groups' exact dots. -/
def atomDot (p : Fp4Atom) (a b : List Nib) (sa sb : List (Fin 256)) (u : Nat) : ℚ :=
  ∑ j ∈ Finset.range (p.k / p.group),
    groupDot p (slice a u p.k) (slice b u p.k) (slice sa u p.scales) (slice sb u p.scales) j

/-- **C̃ is the exact dot product on an exact run**: `cTilde_exact` with each atom's contribution its exact dot. -/
theorem cTilde_dot (p : Fp4Atom) {a b : List Nib} {sa sb : List (Fin 256)} (hs : Shaped p a b sa sb)
    (hr : ExactRun p [] 0 a b sa sb (a.length / p.k)) :
    ∃ w, cTilde p a b sa sb = some w ∧ StepWord w ∧
      wordQ w = some (∑ u ∈ Finset.range (a.length / p.k), atomDot p a b sa sb u) := by
  obtain ⟨w, h1, h2, h3⟩ := cTilde_exact p hs hr
  refine ⟨w, h1, h2, ?_⟩
  rw [h3]
  congr 1
  refine Finset.sum_congr rfl fun u hu => ?_
  obtain ⟨ts, hts, -, -⟩ := hr u (Finset.mem_range.1 hu) (by simp)
  simp only [atomVal, atomDot, hts, Option.map_some, Option.getD_some, groupTerms_value hts]

/-! ## Term 3: dead × dead block pairs -/

/-- The block layout `blockTerms` reads: groups of `group` inside blocks of `block` inside firings of `k`. Both sm_120
atoms have it (`sm120Nvf4_blockShape`, `sm120Mxf4_blockShape`). -/
def BlockShape (p : Fp4Atom) : Prop := 0 < p.group ∧ p.group ∣ p.block ∧ p.block ∣ p.k ∧ 0 < p.k

theorem sm120Nvf4_blockShape : BlockShape sm120Nvf4 := by unfold BlockShape; decide

theorem sm120Mxf4_blockShape : BlockShape sm120Mxf4 := by unfold BlockShape; decide

theorem block_index {G B K β g : ℕ} (hG : 0 < G) (hGB : G ∣ B) (hBK : B ∣ K) (hK : 0 < K) (hg : g < B / G) :
    β * B / K * K + (β * B % K / G + g) * G = β * B + g * G ∧ (β * B % K / G + g) * G + G ≤ K ∧
      (β * B % K / G + g) * G / B < K / B ∧ β * B / K * (K / B) + (β * B % K / G + g) * G / B = β ∧
      g * G + G ≤ B := by
  obtain ⟨c, rfl⟩ := hGB
  obtain ⟨d, rfl⟩ := hBK
  have hc : 0 < c := by rw [Nat.mul_div_cancel_left c hG] at hg; omega
  have hd : 0 < d := Nat.pos_of_ne_zero (by rintro rfl; simp at hK)
  have hGc : 0 < G * c := Nat.mul_pos hG hc
  rw [Nat.mul_div_cancel_left c hG] at hg
  have ht : β * (G * c) / (G * c * d) = β / d := by rw [mul_comm β]; exact Nat.mul_div_mul_left β d hGc
  have hr : β * (G * c) % (G * c * d) / G = c * (β % d) := by
    rw [mul_comm β, Nat.mul_mod_mul_left, mul_assoc]; exact Nat.mul_div_cancel_left _ hG
  have hKB : G * c * d / (G * c) = d := Nat.mul_div_cancel_left d hGc
  have hj : (c * (β % d) + g) * G / (G * c) = β % d := by
    rw [mul_comm G c, Nat.mul_div_mul_right _ _ hG, add_comm, Nat.add_mul_div_left _ _ hc,
      Nat.div_eq_of_lt hg, zero_add]
  rw [ht, hr, hKB, hj]
  have hβ := Nat.div_add_mod β d
  have hs := Nat.mod_lt β hd
  generalize β / d = q at *
  generalize β % d = s at *
  subst hβ
  have h1 : c * s + g + 1 ≤ c * d := by nlinarith
  refine ⟨by ring, ?_, hs, by ring, ?_⟩
  · nlinarith [Nat.mul_le_mul_right G h1]
  · nlinarith

theorem group_of_block {α : Type} (l : List α) {t K j G β B g : ℕ} (h1 : t * K + j * G = β * B + g * G)
    (h2 : j * G + G ≤ K) (h5 : g * G + G ≤ B) :
    (((l.drop (t * K)).take K).drop (j * G)).take G = (((l.drop (β * B)).take B).drop (g * G)).take G := by
  rw [List.drop_take, List.take_take, List.drop_drop, List.drop_take, List.take_take, List.drop_drop, h1,
    min_eq_left (by omega), min_eq_left (by omega)]

theorem scale_of_block {α : Type} (l : List α) {t S i β : ℕ} (h3 : i < S) (h4 : t * S + i = β) :
    ((l.drop (t * S)).take S)[i]? = l[β]? := by
  rw [List.getElem?_take, ite_eq_left h3, List.getElem?_drop, h4]

theorem groupTerm_congr (p : Fp4Atom) {A B A' B' : List Nib} {SA SB SA' SB' : List (Fin 256)} {j j' : ℕ}
    (h1 : (A.drop (j * p.group)).take p.group = (A'.drop (j' * p.group)).take p.group)
    (h2 : (B.drop (j * p.group)).take p.group = (B'.drop (j' * p.group)).take p.group)
    (h3 : SA[j * p.group / p.block]? = SA'[j' * p.group / p.block]?)
    (h4 : SB[j * p.group / p.block]? = SB'[j' * p.group / p.block]?) :
    p.groupTerm A B SA SB j = p.groupTerm A' B' SA' SB' j' := by
  unfold Fp4Atom.groupTerm
  rw [h1, h2, h3, h4]

/-- **A block's group terms read only the block**: its codes and its scale byte on each side. -/
theorem blockTerms_congr {p : Fp4Atom} (hp : BlockShape p) {a a' b b' : List Nib} {sa sa' sb sb' : List (Fin 256)}
    {β : ℕ} (hA : blockOf p a sa β = blockOf p a' sa' β) (hB : blockOf p b sb β = blockOf p b' sb' β) :
    blockTerms p a b sa sb β = blockTerms p a' b' sa' sb' β := by
  obtain ⟨hG, hGB, hBK, hK⟩ := hp
  unfold blockOf at hA hB
  obtain ⟨hA1, hA2⟩ := Prod.mk.inj hA
  obtain ⟨hB1, hB2⟩ := Prod.mk.inj hB
  unfold blockTerms
  apply List.map_congr_left
  intro g hg
  rw [List.mem_range] at hg
  obtain ⟨h1, h2, h3, h4, h5⟩ := block_index (β := β) hG hGB hBK hK hg
  apply groupTerm_congr
  · rw [group_of_block a h1 h2 h5, group_of_block a' h1 h2 h5, hA1]
  · rw [group_of_block b h1 h2 h5, group_of_block b' h1 h2 h5, hB1]
  · unfold Fp4Atom.scales
    rw [scale_of_block sa h3 h4, scale_of_block sa' h3 h4, hA2]
  · unfold Fp4Atom.scales
    rw [scale_of_block sb h3 h4, scale_of_block sb' h3 h4, hB2]

/-- **Term 3's hypothesis, not given by D₄**: at every salt, the forming `form` (a salt to the row's codes and scale
bytes) gives each block that is dead on the clean row `x` the same codes and scale byte. `dead` is a salt-free screen
(noise below 0.16 of the block's code step), so its codes *barely* depend on the salt; this is the exact version. -/
def DeadFixes {σ : Type} {k : Nat} (p : Fp4Atom) (x : Fin k → ℚ) (form : σ → List Nib × List (Fin 256)) : Prop :=
  ∀ β, dead p x β = true → ∀ s s', blockOf p (form s).1 (form s).2 β = blockOf p (form s').1 (form s').2 β

/-- **A dead × dead (word, k-block) is salt-free**: where block `β` is dead on both clean rows and both formings fix
their dead blocks, the block's group terms, the (word, block)'s whole contribution to its firing, are the same at every
salt, so a program can compute them before the salt. -/
theorem deadPair_salt_free {σ : Type} {k : Nat} {p : Fp4Atom} (hp : BlockShape p) {x y : Fin k → ℚ}
    {formA formB : σ → List Nib × List (Fin 256)} (hA : DeadFixes p x formA) (hB : DeadFixes p y formB) {β : ℕ}
    (hβ : deadPair p x y β = true) (s s' : σ) :
    blockTerms p (formA s).1 (formB s).1 (formA s).2 (formB s).2 β =
      blockTerms p (formA s').1 (formB s').1 (formA s').2 (formB s').2 β := by
  unfold deadPair at hβ
  rw [Bool.and_eq_true] at hβ
  exact blockTerms_congr hp (hA β hβ.1 s s') (hB β hβ.2 s s')

/-- **Term 3's debit is sound**: every (word, block) that `pairWords` counts is a dead × dead pair, salt-free by
`deadPair_salt_free`. -/
theorem deadBlocks_salt_free {σ : Type} {m n k : Nat} {p : Fp4Atom} (hp : BlockShape p) {X : Matrix (Fin m) (Fin k) ℚ}
    {Y : Matrix (Fin n) (Fin k) ℚ} {formA : Fin m → σ → List Nib × List (Fin 256)}
    {formB : Fin n → σ → List Nib × List (Fin 256)} (hA : ∀ i, DeadFixes p (X i) (formA i))
    (hB : ∀ j, DeadFixes p (Y j) (formB j)) {i : Fin m} {j : Fin n} {β : ℕ}
    (hβ : β ∈ deadBlocks p k (unitDeadPairs p X Y) i j) (s s' : σ) :
    blockTerms p (formA i s).1 (formB j s).1 (formA i s).2 (formB j s).2 β =
      blockTerms p (formA i s').1 (formB j s').1 (formA i s').2 (formB j s').2 β :=
  deadPair_salt_free hp (hA i) (hB j) (Finset.mem_filter.1 hβ).2 s s'

/-! ## The gap: D-SS's scale bytes don't give `NoLoss` -/

/-- The witness's A codes: group 0 `6, −6`, group 1 `1/2, 6`, the rest 0. -/
def gapA : List Nib := [7, 15] ++ List.replicate 14 0 ++ [1, 7] ++ List.replicate 46 0

/-- The witness's B codes: group 0 `6, 6`, group 1 `1/2, 0, 6`, the rest 0. -/
def gapB : List Nib := [7, 7] ++ List.replicate 14 0 ++ [1, 0, 7] ++ List.replicate 45 0

/-- The witness's UE4M3 scale bytes on both sides: `0x78` (256) on block 0 and `0x0F` (15·2⁻⁹) on blocks 1–3, inside
D-SS's range `0x01..0x7E` with a spread of `2^17/15 ≈ 2^13.1`, under even T1-b's `2^15`. -/
def gapS : List (Fin 256) := [0x78, 0x0F, 0x0F, 0x0F]

/-- **`NoLoss` is a hypothesis beyond D-SS**: on this `sm120Nvf4` firing from `+0`, with scale bytes inside D-SS, group 0
participates and cancels, anchoring the grid at `2^−19`; group 1's scaled dot `225·2^−20` loses its last bit, and the
step writes `224·2^−20` though the exact sum `225·2^−20` fits binary32. -/
theorem noLoss_gap : ∃ ts c, sm120Nvf4.groupTerms gapA gapB gapS gapS = some ts ∧ fp32Value 0 = .fin c ∧
    ¬ NoLoss sm120Nvf4 ts c ∧ c.value + termsValue ts = 225 * 2 ^ (-20 : ℤ) ∧
    Fits (c.value + termsValue ts) ∧ sm120Nvf4.step 0 gapA gapB gapS gapS = some 0x39600000 ∧
    wordQ 0x39600000 = some (224 * 2 ^ (-20 : ℤ)) := by
  refine ⟨[⟨0, 8, true⟩, ⟨225, -20, true⟩, ⟨0, -20, false⟩, ⟨0, -20, false⟩], Term.zero, rfl, rfl, ?_, ?_, ?_, rfl,
    by decide +kernel⟩
  · intro h
    have := h.groups ⟨225, -20, true⟩ (by simp) (by decide +kernel)
    revert this
    decide +kernel
  · norm_num [termsValue, Term.value, Term.zero]
  · norm_num [termsValue, Term.value, Term.zero]
    right
    exact ⟨225, -20, by norm_num, by norm_num, by norm_num, by norm_num⟩

end Pouw.PearlC.Fp4Freeze
