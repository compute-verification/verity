import Definitions.Pouw.Assumptions.Pinned
import Mathlib.Data.ZMod.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Abel
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum

/-!
# (e) Usefulness and int32 exactness

The int32 facts go through `ZMod (2^32)`: `wrap32` is the identity there, so every int32 computation of `H_ref` is
the exact computation mod `2^32`, where the decode identity holds in any commutative ring; two int32 values that
agree mod `2^32` are equal.
-/

namespace Pouw.Proofs

open Finset

/-! ## Matrices over a commutative ring -/

section Ring

variable {A : Type} [CommRing A]

/-- The product over inner dimension `k`, over any commutative ring (`mm` is the case `ℤ`). -/
def mmR (k : ℕ) (X Y : ℕ → ℕ → A) : ℕ → ℕ → A := fun i j => ∑ l ∈ range k, X i l * Y l j

theorem mmR_add_left (k : ℕ) (X X' Y : ℕ → ℕ → A) : mmR k (X + X') Y = mmR k X Y + mmR k X' Y := by
  funext i j; simp [mmR, add_mul, Finset.sum_add_distrib]

theorem mmR_add_right (k : ℕ) (X Y Y' : ℕ → ℕ → A) : mmR k X (Y + Y') = mmR k X Y + mmR k X Y' := by
  funext i j; simp [mmR, mul_add, Finset.sum_add_distrib]

theorem mmR_assoc (a b : ℕ) (X Y Z : ℕ → ℕ → A) : mmR a (mmR b X Y) Z = mmR b X (mmR a Y Z) := by
  funext i j
  simp only [mmR, Finset.sum_mul, Finset.mul_sum]
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun _ _ => Finset.sum_congr rfl fun _ _ => mul_assoc _ _ _

/-- The decode identity in any commutative ring. -/
theorem decode_ring (k r : ℕ) (X B EL ER FL FR : ℕ → ℕ → A) :
    mmR k (X + mmR r EL ER) (B + mmR r FL FR) - mmR r (mmR k X FL) FR - mmR r EL (mmR k ER (B + mmR r FL FR))
      = mmR k X B := by
  rw [mmR_add_left, mmR_add_right k X B, mmR_assoc r k X FL FR, mmR_assoc k r EL ER]
  abel

end Ring

/-- **(e1) Usefulness over the integers.** -/
theorem usefulness : Pinned.Usefulness := fun k r A B EL ER FL FR => decode_ring k r A B EL ER FL FR

/-! ## int32 -/

theorem wrap32_mem (x : ℤ) : InInt32 (wrap32 x) := by
  unfold wrap32 InInt32; omega

theorem wrap32_of_mem {x : ℤ} (h : InInt32 x) : wrap32 x = x := by
  unfold wrap32 InInt32 at *; omega

/-- The ring of int32 values mod `2^32`. -/
abbrev Z32 : Type := ZMod (2 ^ 32)

theorem wrap32_cast (x : ℤ) : ((wrap32 x : ℤ) : Z32) = (x : Z32) := by
  rw [ZMod.intCast_eq_intCast_iff']
  unfold wrap32
  push_cast
  omega

theorem eq_of_cast_eq {x y : ℤ} (hx : InInt32 x) (hy : InInt32 y) (h : (x : Z32) = (y : Z32)) : x = y := by
  rw [ZMod.intCast_eq_intCast_iff'] at h
  push_cast at h
  unfold InInt32 at hx hy
  omega

theorem accW_cast (f : ℕ → ℤ) (n : ℕ) : ((accW f n : ℤ) : Z32) = ∑ l ∈ range n, ((f l : ℤ) : Z32) := by
  induction n with
  | zero => simp [accW]
  | succ n ih => rw [accW, wrap32_cast, Int.cast_add, ih, Finset.sum_range_succ]

/-- Entrywise cast of an integer matrix to `Z32`. -/
def castM (X : Mat) : ℕ → ℕ → Z32 := fun i j => ((X i j : ℤ) : Z32)

theorem castM_mmW (k : ℕ) (X Y : Mat) : castM (mmW k X Y) = mmR k (castM X) (castM Y) := by
  funext i j
  simp only [castM, mmW, mmR, accW_cast, Int.cast_mul]

theorem castM_mm (k : ℕ) (X Y : Mat) : castM (mm k X Y) = mmR k (castM X) (castM Y) := by
  funext i j
  simp only [castM, mm, mmR, Int.cast_sum, Int.cast_mul]

theorem cast_mmW_apply (k : ℕ) (X Y : Mat) (i j : ℕ) :
    ((mmW k X Y i j : ℤ) : Z32) = mmR k (castM X) (castM Y) i j :=
  congrFun (congrFun (castM_mmW k X Y) i) j

theorem cast_mm_apply (k : ℕ) (X Y : Mat) (i j : ℕ) :
    ((mm k X Y i j : ℤ) : Z32) = mmR k (castM X) (castM Y) i j :=
  congrFun (congrFun (castM_mm k X Y) i) j

theorem castM_wrapAdd (X Z : Mat) : castM (fun i l => wrap32 (X i l + Z i l)) = castM X + castM Z := by
  funext i l
  simp only [castM, wrap32_cast, Int.cast_add, Pi.add_apply]

theorem accW_eq (f : ℕ → ℤ) (n : ℕ) : accW f n = wrap32 (∑ l ∈ range n, f l) := by
  refine eq_of_cast_eq ?_ (wrap32_mem _) ?_
  · cases n with
    | zero => simp [accW, InInt32]
    | succ n => exact wrap32_mem _
  · rw [accW_cast, wrap32_cast, Int.cast_sum]

/-- The int32 accumulator is exact when the exact sum is in int32. -/
theorem mmW_eq_of_mem {k : ℕ} {X Y : Mat} {i j : ℕ} (h : InInt32 (mm k X Y i j)) : mmW k X Y i j = mm k X Y i j := by
  unfold mmW
  rw [accW_eq]
  exact wrap32_of_mem h

/-! ## Range bounds -/

theorem sum_bounds (k : ℕ) (f : ℕ → ℤ) (lo hi : ℤ) (h : ∀ l < k, lo ≤ f l ∧ f l ≤ hi) :
    (k : ℤ) * lo ≤ ∑ l ∈ range k, f l ∧ ∑ l ∈ range k, f l ≤ (k : ℤ) * hi := by
  constructor
  · have := Finset.card_nsmul_le_sum (range k) f lo fun l hl => (h l (Finset.mem_range.1 hl)).1
    simpa [nsmul_eq_mul] using this
  · have := Finset.sum_le_card_nsmul (range k) f hi fun l hl => (h l (Finset.mem_range.1 hl)).2
    simpa [nsmul_eq_mul] using this

theorem mul_bounds_noised {a b : ℤ} (ha : -127 ≤ a ∧ a ≤ 126) (hb : -127 ≤ b ∧ b ≤ 126) :
    -16002 ≤ a * b ∧ a * b ≤ 16129 := by
  obtain ⟨ha1, ha2⟩ := ha; obtain ⟨hb1, hb2⟩ := hb
  constructor <;> nlinarith [mul_nonneg (sub_nonneg.2 ha1) (sub_nonneg.2 hb1),
    mul_nonneg (sub_nonneg.2 ha2) (sub_nonneg.2 hb2), mul_nonneg (sub_nonneg.2 ha1) (sub_nonneg.2 hb2),
    mul_nonneg (sub_nonneg.2 ha2) (sub_nonneg.2 hb1)]

theorem mul_bounds_int8 {a b : ℤ} (ha : InInt8 a) (hb : InInt8 b) : -16256 ≤ a * b ∧ a * b ≤ 16384 := by
  obtain ⟨ha1, ha2⟩ := ha; obtain ⟨hb1, hb2⟩ := hb
  constructor <;> nlinarith [mul_nonneg (sub_nonneg.2 ha1) (sub_nonneg.2 hb1),
    mul_nonneg (sub_nonneg.2 ha2) (sub_nonneg.2 hb2), mul_nonneg (sub_nonneg.2 ha1) (sub_nonneg.2 hb2),
    mul_nonneg (sub_nonneg.2 ha2) (sub_nonneg.2 hb1)]

theorem mul_bounds_int7 {a b : ℤ} (ha : InInt7 a) (hb : InInt7 b) : -4032 ≤ a * b ∧ a * b ≤ 4096 := by
  obtain ⟨ha1, ha2⟩ := ha; obtain ⟨hb1, hb2⟩ := hb
  constructor <;> nlinarith [mul_nonneg (sub_nonneg.2 ha1) (sub_nonneg.2 hb1),
    mul_nonneg (sub_nonneg.2 ha2) (sub_nonneg.2 hb2), mul_nonneg (sub_nonneg.2 ha1) (sub_nonneg.2 hb2),
    mul_nonneg (sub_nonneg.2 ha2) (sub_nonneg.2 hb1)]

theorem noised_sum_mem (k : ℕ) (a b : ℕ → ℤ) (hk : k ≤ 133144) (ha : ∀ l < k, -127 ≤ a l ∧ a l ≤ 126)
    (hb : ∀ l < k, -127 ≤ b l ∧ b l ≤ 126) : InInt32 (∑ l ∈ range k, a l * b l) := by
  have := sum_bounds k (fun l => a l * b l) (-16002) 16129 fun l hl => mul_bounds_noised (ha l hl) (hb l hl)
  unfold InInt32
  have hk' : (k : ℤ) ≤ 133144 := by exact_mod_cast hk
  constructor <;> nlinarith [this.1, this.2]

theorem int7_sum_mem (k : ℕ) (a b : ℕ → ℤ) (hk : k ≤ 2 ^ 19 - 1) (ha : ∀ l < k, InInt7 (a l))
    (hb : ∀ l < k, InInt7 (b l)) : InInt32 (∑ l ∈ range k, a l * b l) := by
  have := sum_bounds k (fun l => a l * b l) (-4032) 4096 fun l hl => mul_bounds_int7 (ha l hl) (hb l hl)
  unfold InInt32
  have hk' : (k : ℤ) ≤ 524287 := by exact_mod_cast (show k ≤ 524287 by omega)
  constructor <;> nlinarith [this.1, this.2]

/-- **(e2)** int32 exactness of the noised transcript for `k ≤ 133144`. -/
theorem int32Transcript : Pinned.Int32Transcript := by
  intro k A B i j hk hA hB
  have hmem : InInt32 (mm k A B i j) := noised_sum_mem k (fun l => A i l) (fun l => B l j) hk hA hB
  exact ⟨mmW_eq_of_mem hmem, hmem⟩

/-- **(e3)** The exact `k` bound for noised operands. -/
theorem noisedKBound : Pinned.NoisedKBound := by
  refine ⟨fun k a b hk ha hb => noised_sum_mem k a b hk ha hb, ?_⟩
  simp only [Finset.sum_const, Finset.card_range, nsmul_eq_mul, InInt32]
  norm_num

/-- **(e3′)** The problem statement's int8 bound. -/
theorem int8KBound : Pinned.Int8KBound := by
  refine ⟨fun k a b hk ha hb => ?_, ?_⟩
  · have := sum_bounds k (fun l => a l * b l) (-16256) 16384 fun l hl => mul_bounds_int8 (ha l hl) (hb l hl)
    unfold InInt32
    have hk' : (k : ℤ) ≤ 131071 := by exact_mod_cast (show k ≤ 131071 by omega)
    constructor <;> nlinarith [this.1, this.2]
  · simp only [Finset.sum_const, Finset.card_range, nsmul_eq_mul, InInt32]
    norm_num

/-- **(e4)** `H_ref`'s int32 decode is exact for int7 operands and `k ≤ 2^19 − 1`, whatever the factors. -/
theorem int32Decode : Pinned.Int32Decode := by
  intro k r A B EL ER FL FR i j hk hA hB
  refine eq_of_cast_eq (wrap32_mem _) (int7_sum_mem k (fun l => A i l) (fun l => B l j) hk hA hB) ?_
  have key := congrFun (congrFun (decode_ring k r (castM A) (castM B) (castM EL) (castM ER) (castM FL)
    (castM FR)) i) j
  simp only [wrap32_cast, Int.cast_sub, cast_mmW_apply, cast_mm_apply, castM_mmW, castM_wrapAdd]
  exact key

/-! ## D1's honest reference is complete -/

section D1

variable {Q R S : Type} (P : D1.Params Q R S)

theorem blockEnd_le (d k t : ℕ) : D1.blockEnd d k t ≤ k := min_le_right _ _

theorem refAct_eq (hN : P.NoiseRange) {H : Q → R} {s : S} {U : Workload} {u : ℕ} {A : Mat}
    (hA : ActOK (U.layout.shape u) A) {i l : ℕ} (hi : i < (U.layout.shape u).m) (hl : l < (U.layout.shape u).k) :
    P.refAct H s U u A i l = P.noisedAct H s U u A i l ∧ -127 ≤ P.refAct H s U u A i l ∧
      P.refAct H s U u A i l ≤ 126 := by
  have hn : InNoise (mm P.r (P.actF H s U u A).1 (P.actF H s U u A).2 i l) := hN.1 _ _ _ i hi l hl
  have ha := hA i hi l hl
  unfold InNoise at hn; unfold InInt7 at ha
  have hW : mmW P.r (P.actF H s U u A).1 (P.actF H s U u A).2 i l
      = mm P.r (P.actF H s U u A).1 (P.actF H s U u A).2 i l :=
    mmW_eq_of_mem (by unfold InInt32; omega)
  have h : P.refAct H s U u A i l = A i l + mm P.r (P.actF H s U u A).1 (P.actF H s U u A).2 i l := by
    unfold D1.Params.refAct; rw [hW]; exact wrap32_of_mem (by unfold InInt32; omega)
  refine ⟨h.trans rfl, ?_, ?_⟩ <;> rw [h] <;> omega

theorem refWt_eq (hN : P.NoiseRange) {H : Q → R} {s : S} {U : Workload} {u : ℕ}
    (hB : ∀ l < (U.layout.shape u).k, ∀ j < (U.layout.shape u).n, InInt7 (U.weight (U.layout.wid u) l j))
    {l j : ℕ} (hl : l < (U.layout.shape u).k) (hj : j < (U.layout.shape u).n) :
    P.refWt H s U u l j = P.noisedWt H s U u l j ∧ -127 ≤ P.refWt H s U u l j ∧ P.refWt H s U u l j ≤ 126 := by
  have hn : InNoise (mm P.r (P.wtF H s U u).1 (P.wtF H s U u).2 l j) := hN.2 _ _ l hl j hj
  have hb := hB l hl j hj
  unfold InNoise at hn; unfold InInt7 at hb
  have hW : mmW P.r (P.wtF H s U u).1 (P.wtF H s U u).2 l j = mm P.r (P.wtF H s U u).1 (P.wtF H s U u).2 l j :=
    mmW_eq_of_mem (by unfold InInt32; omega)
  have h : P.refWt H s U u l j = U.weight (U.layout.wid u) l j + mm P.r (P.wtF H s U u).1 (P.wtF H s U u).2 l j := by
    unfold D1.Params.refWt; rw [hW]; exact wrap32_of_mem (by unfold InInt32; omega)
  refine ⟨h.trans rfl, ?_, ?_⟩ <;> rw [h] <;> omega

/-- **(e5)** D1's `H_ref` is complete on the default domain when noise is in `[−63, 63]`. -/
theorem d1Complete : Pinned.D1Complete := by
  intro Q R S P hN U hU u hu H s A hA
  have hD := hU.2.1 u hu
  have hB := hU.2.2 u hu
  have hk : (U.layout.shape u).k ≤ 2 ^ 16 := hD.2.2.2.1
  refine ⟨fun x hx => ?_, fun i hi j hj => ?_⟩
  · obtain ⟨t, i, j⟩ := x
    simp only [D1.Params.toProtocol, D1.Params.checkIdx, Finset.mem_product, Finset.mem_range] at hx
    obtain ⟨-, hi, hj⟩ := hx
    show mmW _ (P.refAct H s U u A) (P.refWt H s U u) i j = mm _ (P.noisedAct H s U u A) (P.noisedWt H s U u) i j
    have he := blockEnd_le P.d (U.layout.shape u).k t
    set e := D1.blockEnd P.d (U.layout.shape u).k t
    have hex := (int32Transcript e (P.refAct H s U u A) (P.refWt H s U u) i j (by omega)
      (fun l hl => (refAct_eq P hN hA hi (by omega)).2)
      (fun l hl => (refWt_eq P hN hB (by omega) hj).2)).1
    rw [hex]
    refine Finset.sum_congr rfl fun l hl => ?_
    have hl' : l < (U.layout.shape u).k := by rw [Finset.mem_range] at hl; omega
    rw [(refAct_eq P hN hA hi hl').1, (refWt_eq P hN hB hl' hj).1]
  · exact int32Decode _ P.r A (U.weight (U.layout.wid u)) (P.actF H s U u A).1 (P.actF H s U u A).2
      (P.wtF H s U u).1 (P.wtF H s U u).2 i j (by omega) (fun l hl => hA i hi l hl) (fun l hl => hB l hl j hj)

end D1

end Pouw.Proofs
