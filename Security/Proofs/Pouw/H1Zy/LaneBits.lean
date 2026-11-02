import Definitions.Pouw.H1Zy.LaneBits
import Proofs.Pouw.H1Zy.CombineBits
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic.LinearCombination
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# The lane's pieces on bits compute their word views, and so does their composition

* **The multiply** (`immaS8Bits_bv`): an s8 × s8 product is its Baugh–Wooley array's weighted sum plus `_BW_CONSTANT`
  (`bw_sum`, a polynomial identity in the operands' bits), and the accumulator's 33-bit sign extension is its s32 value
  plus `2^33` times its sign (`acc_sum`), so the columns' sum is the exact sum mod `2^33` (`colSum_emod`). That sum lies in
  `[-2^32, 2^32)` (`dot_bounds`), so bits 31 and 32 of its 33 bits differ exactly when it leaves s32, bit 32 then its
  sign, and the clamp is `satS32`'s. Through the steps, `gemmAccS8Bits_bv`.
* **The record check, the picks and the partials**: `zyCellBits_bv`, `zyCheckBits_bv`, `muxCand_bv` with `pickBy_map`,
  `mask32Bits_bv`, and `eq32Bits_bitsOf`.
* **The lane** (`laneBits_bv`): on any lane's inputs' bits, at the dequant's and the chain's word views on bits,
  `laneBits` is `laneOk`.
-/

namespace Pouw.H1Zy

theorem mod_two_pow_eq_sum (n : ℕ) : ∀ w, n % 2 ^ w = ∑ i : Fin w, 2 ^ i.val * bval (n.testBit i.val)
  | 0 => by simp [Nat.mod_one]
  | w + 1 => by
      rw [Nat.mod_pow_succ, mod_two_pow_eq_sum n w, Fin.sum_univ_castSucc]
      simp only [Fin.val_castSucc, Fin.val_last]
      congr 1
      rw [Nat.testBit_eq_decide_div_mod_eq]
      have := Nat.mod_lt (n / 2 ^ w) (show 2 > 0 by norm_num)
      unfold bval
      split <;> simp_all

theorem toNat_eq_sum {w : ℕ} (x : BitVec w) : x.toNat = ∑ i : Fin w, 2 ^ i.val * bval (bv x i) := by
  rw [← Nat.mod_eq_of_lt x.isLt, mod_two_pow_eq_sum]
  rfl

theorem bval_not (b : Bool) : (bval (!b) : ℤ) = 1 - bval b := by cases b <;> simp [bval]
theorem bval_and (a b : Bool) : (bval (a && b) : ℤ) = bval a * bval b := by cases a <;> cases b <;> simp [bval]

theorem toInt8 (x : BitVec 8) : x.toInt = (∑ i : Fin 7, 2 ^ i.val * (bval (bv x i.castSucc) : ℤ)) -
    2 ^ 7 * bval (bv x 7) := by
  rw [BitVec.toInt_eq_toNat_cond, toNat_eq_sum, Fin.sum_univ_castSucc]
  have h7 : bval (bv x (Fin.last 7)) ≤ 1 := by unfold bval; split <;> omega
  have hs := x.isLt
  rw [toNat_eq_sum, Fin.sum_univ_castSucc] at hs
  simp only [Fin.val_castSucc, show Fin.last 7 = (7 : Fin 8) from rfl] at hs h7 ⊢
  push_cast
  have hlo : ∑ i : Fin 7, 2 ^ i.val * bval (bv x i.castSucc) < 2 ^ 7 := by
    have := (BitVec.setWidth 7 x).isLt
    rw [toNat_eq_sum] at this
    convert this using 2 with i
    simp [bv]
  rcases Nat.lt_or_ge (bval (bv x 7)) 1 with h | h
  · have h0 : bval (bv x 7) = 0 := by omega
    simp only [h0, mul_zero, add_zero] at hs ⊢
    simp only [Fin.coe_ofNat_eq_mod, Nat.reduceMod]
    split_ifs with hc
    · push_cast; ring
    · exfalso; omega
  · have h1 : bval (bv x 7) = 1 := by omega
    simp only [h1, mul_one] at hs ⊢
    simp only [Fin.coe_ofNat_eq_mod, Nat.reduceMod]
    split_ifs with hc
    · exfalso; omega
    · push_cast; ring

/-- **Baugh–Wooley**: an s8 × s8 product is its array's weighted sum plus `bwConst`. -/
theorem bw_sum (x y : BitVec 8) :
    ((∑ i : Fin 8, ∑ j : Fin 8, 2 ^ (i.val + j.val) * bval (bwBit (bv x) (bv y) i j) : ℕ) : ℤ) + bwConst =
      x.toInt * y.toInt := by
  rw [toInt8 x, toInt8 y]
  push_cast
  simp only [Fin.sum_univ_eight, Fin.sum_univ_seven, bwBit]
  simp (config := { decide := true }) only [Fin.isValue, Fin.val_zero, Fin.val_one, Fin.val_two,
    ite_true, ite_false, bval_not, bval_and]
  simp only [bwConst, Fin.coe_ofNat_eq_mod, Nat.reduceMod, Fin.castSucc_zero]
  simp only [show Fin.castSucc (1 : Fin 7) = (1 : Fin 8) from rfl, show Fin.castSucc (2 : Fin 7) = (2 : Fin 8) from rfl, show Fin.castSucc (3 : Fin 7) = (3 : Fin 8) from rfl, show Fin.castSucc (4 : Fin 7) = (4 : Fin 8) from rfl, show Fin.castSucc (5 : Fin 7) = (5 : Fin 8) from rfl, show Fin.castSucc (6 : Fin 7) = (6 : Fin 8) from rfl]
  ring


theorem bit31 (acc : BitVec 32) : (bval (bv acc 31) : ℤ) = if acc.toNat < 2 ^ 31 then 0 else 1 := by
  have h := acc.isLt
  simp only [bv, bval, BitVec.getLsbD, Nat.testBit_eq_decide_div_mod_eq, Fin.isValue]
  simp only [Fin.coe_ofNat_eq_mod, Nat.reduceMod, Nat.reducePow] at h ⊢
  split_ifs with h1 h2 h2 <;> simp_all <;> omega

/-- The accumulator sign-extended to 33 bits is its s32 value plus `2^33` times its sign. -/
theorem acc_sum (acc : BitVec 32) :
    ((∑ i : Fin 32, 2 ^ i.val * bval (bv acc i) : ℕ) : ℤ) + 2 ^ 32 * bval (bv acc 31) =
      acc.toInt + 2 ^ 33 * bval (bv acc 31) := by
  rw [← toNat_eq_sum, BitVec.toInt_eq_toNat_cond, bit31]
  split_ifs with h1 h2 h2 <;> push_cast <;> omega

theorem colSum_emod (acc : BitVec 32) (a b : Fin 32 → BitVec 8) :
    (stepColSum (bv acc) (fun k => bv (a k)) (fun k => bv (b k)) : ℤ) % 2 ^ 33 =
      (acc.toInt + ∑ k, (a k).toInt * (b k).toInt) % 2 ^ 33 := by
  have hc : (((32 * bwConst % 2 ^ 33).toNat : ℕ) : ℤ) = 32 * bwConst % 2 ^ 33 :=
    Int.toNat_of_nonneg (Int.emod_nonneg _ (by norm_num))
  have hp : ∀ k, ((∑ i : Fin 8, ∑ j : Fin 8, 2 ^ (i.val + j.val) * bval (bwBit (bv (a k)) (bv (b k)) i j) : ℕ) : ℤ) =
      (a k).toInt * (b k).toInt - bwConst := fun k => by rw [← bw_sum]; ring
  have e : (stepColSum (bv acc) (fun k => bv (a k)) (fun k => bv (b k)) : ℤ) =
      (acc.toInt + ∑ k, (a k).toInt * (b k).toInt) +
        2 ^ 33 * (bval (bv acc 31) - 32 * bwConst / 2 ^ 33) := by
    unfold stepColSum
    simp only [Nat.cast_add]
    rw [hc]
    push_cast
    have hs : ∑ k, ((∑ i : Fin 8, ∑ j : Fin 8, 2 ^ (i.val + j.val) * bval (bwBit (bv (a k)) (bv (b k)) i j) : ℕ) : ℤ) =
        ∑ k, ((a k).toInt * (b k).toInt - bwConst) := Finset.sum_congr rfl fun k _ => hp k
    push_cast at hs
    have ha := acc_sum acc
    push_cast at ha
    rw [hs, Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin]
    have hd := Int.emod_add_mul_ediv (32 * bwConst) (2 ^ 33)
    simp only [nsmul_eq_mul, Nat.cast_ofNat]
    simp only [Int.reducePow] at hd ⊢
    linear_combination ha + hd
  rw [e, Int.add_mul_emod_self_left]

theorem prod_bounds (x y : BitVec 8) : -16256 ≤ x.toInt * y.toInt ∧ x.toInt * y.toInt ≤ 16384 := by
  have h1 := BitVec.le_toInt x
  have h2 := BitVec.toInt_lt (x := x)
  have h3 := BitVec.le_toInt y
  have h4 := BitVec.toInt_lt (x := y)
  norm_num at h1 h2 h3 h4
  constructor <;> nlinarith

theorem dot_bounds (a b : Fin 32 → BitVec 8) :
    -520192 ≤ ∑ k, (a k).toInt * (b k).toInt ∧ ∑ k, (a k).toInt * (b k).toInt ≤ 524288 := by
  constructor
  · have := Finset.sum_le_sum (s := Finset.univ) fun k _ => (prod_bounds (a k) (b k)).1
    simpa using this
  · have := Finset.sum_le_sum (s := Finset.univ) fun k _ => (prod_bounds (a k) (b k)).2
    simpa using this

theorem satMax : ∀ i : Fin 32, (BitVec.ofInt 32 (2 ^ 31 - 1)).getLsbD i.val = decide (i.val < 31) := by decide
theorem satMin : ∀ i : Fin 32, (BitVec.ofInt 32 (-2 ^ 31)).getLsbD i.val = decide (i.val = 31) := by decide

/-- **`BlackwellS8ImmaDot32_v2`'s lowering computes `BlackwellS8ImmaDot32_v1`**, given `compress_columns`' contract. -/
theorem immaS8Bits_bv (acc : BitVec 32) (a b : Fin 32 → BitVec 8) :
    immaS8Bits (bv acc) (fun k => bv (a k)) (fun k => bv (b k)) = bv (immaS8 acc a b) := by
  set T := acc.toInt + ∑ k, (a k).toInt * (b k).toInt with hT
  have hcmp : ∀ i : Fin 33, compressCols 33 (stepColSum (bv acc) (fun k => bv (a k)) (fun k => bv (b k))) i =
      (T % 2 ^ 33).toNat.testBit i.val := fun i => by
    unfold compressCols
    have h33 : (stepColSum (bv acc) (fun k => bv (a k)) (fun k => bv (b k))) % 2 ^ 33 = (T % 2 ^ 33).toNat := by
      have := colSum_emod acc a b
      omega
    have := Nat.testBit_mod_two_pow (stepColSum (bv acc) (fun k => bv (a k)) (fun k => bv (b k))) 33 i.val
    rw [h33] at this
    rw [this]
    simp [i.isLt]
  have hacc1 := BitVec.le_toInt acc
  have hacc2 := BitVec.toInt_lt (x := acc)
  norm_num at hacc1 hacc2
  obtain ⟨hs1, hs2⟩ := dot_bounds a b
  have hm := Int.emod_nonneg T (show (2 : ℤ) ^ 33 ≠ 0 by norm_num)
  have hm2 := Int.emod_lt_of_pos T (show (0 : ℤ) < 2 ^ 33 by norm_num)
  have hu : ((T % 2 ^ 33).toNat : ℤ) = T % 2 ^ 33 := Int.toNat_of_nonneg hm
  have hb : ∀ j, (T % 2 ^ 33).toNat.testBit j = decide ((T % 2 ^ 33).toNat / 2 ^ j % 2 = 1) := fun j =>
    Nat.testBit_eq_decide_div_mod_eq
  have h32 : ∀ i : Fin 32, i.val < 32 := fun i => i.isLt
  funext i
  simp only [immaS8Bits, hcmp, bv, immaS8, Fin.isValue]
  rcases lt_or_ge T (2 ^ 31) with hTlt | hTge
  · rcases lt_or_ge T (-2 ^ 31) with hTlo | hTlo
    · have hs : satS32 T = -2 ^ 31 := by unfold satS32; omega
      have e31 : (T % 2 ^ 33).toNat.testBit 31 = false := by rw [hb]; simp; omega
      have e32 : (T % 2 ^ 33).toNat.testBit 32 = true := by rw [hb]; simp; omega
      rw [hs, satMin i]
      simp only [Fin.coe_ofNat_eq_mod, Nat.reduceMod, e31, e32]
      split_ifs with h <;> simp <;> omega
    · have hs : satS32 T = T := by unfold satS32; omega
      have e : (T % 2 ^ 33).toNat.testBit 31 = (T % 2 ^ 33).toNat.testBit 32 := by
        rw [hb, hb]
        rcases le_or_gt 0 T with h0 | h0
        · have h1 : (T % 2 ^ 33).toNat / 2 ^ 31 % 2 = 0 := by omega
          have h2 : (T % 2 ^ 33).toNat / 2 ^ 32 % 2 = 0 := by omega
          rw [h1, h2]
        · have h1 : (T % 2 ^ 33).toNat / 2 ^ 31 % 2 = 1 := by omega
          have h2 : (T % 2 ^ 33).toNat / 2 ^ 32 % 2 = 1 := by omega
          rw [h1, h2]
      have hr : (BitVec.ofInt 32 T).getLsbD i.val = (T % 2 ^ 33).toNat.testBit i.val := by
        rw [BitVec.getLsbD, BitVec.toNat_ofInt]
        have : (T % ((2 ^ 32 : ℕ) : ℤ)).toNat = (T % 2 ^ 33).toNat % 2 ^ 32 := by omega
        rw [this, Nat.testBit_mod_two_pow]
        simp [i.isLt]
      rw [hs, hr]
      split_ifs with h
      · simp only [Fin.coe_ofNat_eq_mod, Nat.reduceMod, e, Bool.xor_self, Bool.false_and, Bool.xor_false]
      · have : i.val = 31 := by omega
        simp only [Fin.coe_ofNat_eq_mod, Nat.reduceMod, this, e]
  · have hs : satS32 T = 2 ^ 31 - 1 := by unfold satS32; omega
    have e31 : (T % 2 ^ 33).toNat.testBit 31 = true := by rw [hb]; simp; omega
    have e32 : (T % 2 ^ 33).toNat.testBit 32 = false := by rw [hb]; simp; omega
    rw [hs, satMax i]
    simp only [Fin.coe_ofNat_eq_mod, Nat.reduceMod, e31, e32]
    split_ifs with h <;> simp <;> omega

theorem ofBits_bv {n : ℕ} (x : BitVec n) : ofBits (bv x) = x := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  rw [ofBits, BitVec.getLsbD_cast, BitVec.getLsbD_ofBoolListLE]
  simp [List.getD_eq_getElem?_getD, hi, bv, BitVec.getLsbD_eq_getElem]

theorem bitsOf_eq_bv : bitsOf = bv (n := 32) := rfl

theorem muxBit_eq (s a b : Bool) : muxBit s a b = mux s a b := by cases s <;> cases a <;> cases b <;> rfl

theorem mask32Bits_bv (w : BitVec 32) (live : Bool) : mask32Bits (bv w) live = bv (mask32 w live) := by
  funext i
  cases live <;> simp [mask32Bits, mask32, bv]

theorem zyCellBits_bv (zr : BitVec 32) (yr : BitVec 16) (z : BitVec 32) (y : BitVec 16) :
    zyCellBits (bv zr) (bv yr) (bv z) (bv y) = zyCell zr yr z y := by
  apply Bool.eq_iff_iff.2
  rw [zyCellBits, tree_andBit, zyCell]
  simp only [List.mem_append, List.mem_ofFn, Bool.and_eq_true, decide_eq_true_eq, bv]
  constructor
  · intro h
    constructor
    · refine BitVec.eq_of_getLsbD_eq fun i hi => ?_
      have := h _ (Or.inl ⟨⟨i, hi⟩, rfl⟩)
      revert this
      cases zr.getLsbD i <;> cases z.getLsbD i <;> simp
    · refine BitVec.eq_of_getLsbD_eq fun i hi => ?_
      have := h _ (Or.inr ⟨⟨i, hi⟩, rfl⟩)
      revert this
      cases yr.getLsbD i <;> cases y.getLsbD i <;> simp
  · rintro ⟨rfl, rfl⟩ b (⟨_, rfl⟩ | ⟨_, rfl⟩) <;> simp

theorem gemmStepsBits_bv {K : ℕ} (x w : Fin K → BitVec 8) :
    ∀ n, gemmStepsBits (fun k => bv (x k)) (fun k => bv (w k)) n = bv (gemmAccSteps x w n)
  | 0 => by funext i; simp [gemmStepsBits, gemmAccSteps, bv]
  | n + 1 => by
      rw [gemmStepsBits, gemmAccSteps, gemmStepsBits_bv x w n, ← immaS8Bits_bv]
      congr 1
      · funext j
        unfold codeAt
        split_ifs <;> first | rfl | (funext i; simp [bv])
      · funext j
        unfold codeAt
        split_ifs <;> first | rfl | (funext i; simp [bv])

theorem gemmAccS8Bits_bv {K : ℕ} (x w : Fin K → BitVec 8) :
    gemmAccS8Bits (fun k => bv (x k)) (fun k => bv (w k)) = bv (gemmAccS8 x w) :=
  gemmStepsBits_bv x w _

theorem foldl_pairs_map {α β : Type} (f : Bool → α → α → α) (g : Bool → β → β → β) (φ : α → β)
    (h : ∀ s a b, φ (f s a b) = g s (φ a) (φ b)) :
    ∀ (sel : List Bool) (cs : List α),
      sel.foldl (fun cs s => pairs (g s) cs) (cs.map φ) = (sel.foldl (fun cs s => pairs (f s) cs) cs).map φ
  | [], _ => rfl
  | s :: sel, cs => by
      simp only [List.foldl_cons]
      rw [pairs_map (f s) (g s) φ (h s), foldl_pairs_map f g φ h sel]

theorem pickBy_map {α β : Type} (f : Bool → α → α → α) (g : Bool → β → β → β) (φ : α → β)
    (h : ∀ s a b, φ (f s a b) = g s (φ a) (φ b)) (d : α) (d' : β) (hd : φ d = d') (sel : List Bool) (cs : List α) :
    pickBy g d' sel (cs.map φ) = φ (pickBy f d sel cs) := by
  unfold pickBy
  rw [foldl_pairs_map f g φ h]
  cases (sel.foldl (fun cs s => pairs (f s) cs) cs) <;> simp [hd]

theorem pick_eq_pickBy {α : Type} (d : α) (sel : List Bool) (cs : List α) : pick d sel cs = pickBy mux d sel cs := rfl

theorem muxCand_bv (s : Bool) (p q : BitVec 32 × BitVec 16) :
    muxCand s (bv p.1, bv p.2) (bv q.1, bv q.2) = (bv (mux s p q).1, bv (mux s p q).2) := by
  cases s <;> simp [muxCand, muxBit, mux, bv] <;> constructor <;> funext i <;> cases p.1.getLsbD i <;>
    cases q.1.getLsbD i <;> cases p.2.getLsbD i <;> cases q.2.getLsbD i <;> rfl


theorem liftDq_bv (dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16) (z sa sw : BitVec 32) :
    liftDq dqW (bv z) (bv sa) (bv sw) = bv (dqW z sa sw) := by
  simp only [liftDq, ofBits_bv]

theorem liftChain_bv {K nw : ℕ} {P : Type}
    (chain : P → Fin 32 → (Fin 4 → Fin K → BitVec 8) → (Fin 8 → Fin K → BitVec 8) → Fin nw → BitVec 32)
    (p : P) (l : Fin 32) (a : Fin 4 → Fin K → BitVec 8) (b : Fin 8 → Fin K → BitVec 8) (w : Fin nw) :
    liftChain chain p l (fun s k => bv (a s k)) (fun t k => bv (b t k)) w = bv (chain p l a b w) := by
  simp only [liftChain, ofBits_bv]

theorem zyCheckBits_bv {R C : ℕ} (dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16)
    (zr : Fin R → Fin C → BitVec 32) (yr : Fin R → Fin C → BitVec 16) (z : Fin R → Fin C → BitVec 32)
    (sa : Fin R → BitVec 32) (sw : Fin C → BitVec 32) :
    zyCheckBits (liftDq dqW) (fun i j => bv (zr i j)) (fun i j => bv (yr i j)) (fun i j => bv (z i j))
      (fun i => bv (sa i)) (fun j => bv (sw j)) = zyCheck dqW zr yr z sa sw := by
  simp only [zyCheckBits, zyCheck, liftDq_bv, zyCellBits_bv]

/-- **The lane's body on bits computes it on words**: on the bits of any lane inputs, at the dequant's and the chain's
word views on bits, `laneBits` is `laneOk`. -/
theorem laneBits_bv {K nw : ℕ} {P : Type}
    (chain : P → Fin 32 → (Fin 4 → Fin K → BitVec 8) → (Fin 8 → Fin K → BitVec 8) → Fin nw → BitVec 32)
    (dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16) (x : LaneIn K nw P) :
    laneBits (liftChain chain) (liftDq dqW) x.bits = laneOk chain dqW x := by
  have hpick : ∀ s t, pickBy muxCand (fun _ => false, fun _ => false) x.sel
      (List.ofFn fun ln : Fin 32 => (bv (x.zr (laneRow ln s) (laneCol ln t)), bv (x.yr (laneRow ln s) (laneCol ln t)))) =
      ((fun p : BitVec 32 × BitVec 16 => (bv p.1, bv p.2)) (pick (0, 0) x.sel
        (List.ofFn fun ln : Fin 32 => (x.zr (laneRow ln s) (laneCol ln t), x.yr (laneRow ln s) (laneCol ln t))))) := by
    intro s t
    rw [pick_eq_pickBy]
    have := pickBy_map mux muxCand (fun p : BitVec 32 × BitVec 16 => (bv p.1, bv p.2))
      (fun s p q => (muxCand_bv s p q).symm) (0, 0) (fun _ => false, fun _ => false)
      (by simp only [Prod.mk.injEq]; constructor <;> funext i <;> simp [bv]) x.sel
      (List.ofFn fun ln : Fin 32 => (x.zr (laneRow ln s) (laneCol ln t), x.yr (laneRow ln s) (laneCol ln t)))
    rw [List.map_ofFn] at this
    exact this
  simp only [LaneIn.sel] at hpick
  have hsel : ∀ w, liftChain chain x.pub ⟨bitsVal (List.ofFn fun t : Fin 5 => x.lane (Fin.castLE (by omega) t)),
      by simpa using bitsVal_lt (List.ofFn fun t : Fin 5 => x.lane (Fin.castLE (by omega) t))⟩
      (fun s k => bv (x.a s k)) (fun t k => bv (x.b t k)) w = bv (chain x.pub x.l x.a x.b w) := fun w =>
    liftChain_bv chain _ _ _ _ w
  have heq : ∀ a b : BitVec 32, eq32Bits (bv a) (bv b) = eq32 a b := eq32Bits_bitsOf
  simp only [laneBits, laneOk, muxBit_eq, LaneIn.bits, LaneIn.sel, hpick, hsel, gemmAccS8Bits_bv, mask32Bits_bv, heq]
  rw [← zyCheckBits_bv]

end Pouw.H1Zy
