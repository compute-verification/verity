import Definitions.Core.Accel.Principles
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Linarith

/-!
# The accelerator model's principles, proved

The integer path is exact on the s32 range, saturating or not (`p1Int`). P2's halves that need no capture: an MMA's
output reads only its own row, column and accumulator (`p2Local`; a sparse MMA's, its kept pairs, `p2LocalSp`), its
outputs are distinct products' destinations (`p2Distinct`), an opcode writes no word but its outputs (`p2Frame`), and
every opcode takes at least a lane's share of an issue slot (`p2Priced`). Each card refines its coarse `Device` (`H200.refines`, `Sm120.refines`). The integer
path's dots have the product form (`intProductForm`, the integer half of `CapturedProductForm`; the float half, the
cards' `CapturedProductForm` and P1 on their captured float atoms are in `Float.lean`).
-/

open scoped NNRat

namespace Verity.Accel

theorem p1Int : P1Int := by
  intro fa fb sat c a b _ s h1 h2
  have hs : (if sat then max (-2 ^ 31) (min (2 ^ 31 - 1) s) else s) = s := by split <;> omega
  simp only [intDot]
  rw [hs, BitVec.toInt_ofInt]
  exact Int.bmod_eq_of_le h1 h2

theorem p2Local : P2Local := by
  intro dot fa fb s v v' o hA hB hC
  simp only [mmaEval, hA, hB, hC]

theorem p2LocalSp : P2LocalSp := by
  intro dot fa fb s v v' o hA hB hC
  simp only [mmaSpEval, hA, hB, hC]

theorem p2Distinct : P2Distinct := by
  intro s o o' _ _ hq hr
  rw [← Nat.div_add_mod o s.n, ← Nat.div_add_mod o' s.n, hq, hr]

theorem p2Priced : P2Priced := by
  have h : (1 / 32 : ℚ≥0) ≤ 1 := by rw [← NNRat.coe_le_coe]; norm_num
  intro weight scale hi o
  cases o with
  | mma fa fb _ sh sa sb => simpa [opUse, port, rd] using h
  | mmaSp fa fb _ sh sa sb => simpa [opUse, port, rd] using h
  | lane g => simp [opUse]
  | uni g => simpa [opUse] using h
  | copy s t n => cases s <;> cases t <;> simp [opUse, port, rd]
  | bcast s n k =>
    have h1 : (1 : ℚ) ≤ max (k : ℚ) 1 := le_max_right _ _
    have hk : (1 / 32 : ℚ≥0) ≤ ((max k 1 : ℕ) : ℚ≥0) / 32 := by
      rw [← NNRat.coe_le_coe]
      push_cast
      linarith
    cases s <;> simpa [opUse, port, rd] using hk
  | atom s _ => cases s <;> simp [opUse, port, rd]
  | red s t n _ => cases s <;> cases t <;> simp [opUse, port, rd]
  | dmma sh => simpa [opUse] using h

theorem p2Frame : P2Frame := by
  intro ι m o src dst pred M l hl
  simp only [Machine.execOp]
  split
  · rfl
  · suffices ∀ (L : List (Fin (m.nOut o))) (M' : m.Mem), (L.foldl (fun M' i => (dst i).writeIn (m.outStore o i) M M'
        (m.eval o (fun i => (src i).readIn (m.inStore o i) M) i)) M') l = M' l from this _ M
    intro L
    induction L with
    | nil => intro M'; rfl
    | cons i L ih =>
      intro M'
      rw [List.foldl_cons, ih]
      simp only [Machine.Operand.writeIn]
      have := hl i
      cases hx : Machine.Operand.locIn (m.outStore o i) (dst i) M with
      | none => rfl
      | some l' => exact Function.update_of_ne (fun e => this (by rw [hx, e])) _ _

theorem H200.refines : H200.card.Refines := by
  refine ⟨?_, ?_, ?_, rfl⟩ <;> simp only [H200.card, H200.ρ] <;> rw [← NNRat.coe_le_coe] <;>
    push_cast <;> norm_num

theorem Sm120.refines : Sm120.card.Refines := by
  refine ⟨?_, ?_, ?_, rfl⟩ <;> simp only [Sm120.card, Sm120.ρ] <;> rw [← NNRat.coe_le_coe] <;>
    push_cast <;> norm_num

/-- The integer path's dot has the product form: the products' signs and magnitudes, summed into the accumulator. -/
theorem intProductForm (fa fb : Fmt) (sat : Bool) (k : ℕ) (h : intPair fa fb = true) :
    ProductForm fa fb k (intDot fa fb sat) := by
  refine ⟨fun c ms _ _ =>
    let s : ℤ := c.toInt + (ms.map fun m => if m.2.2.1 then -(m.2.2.2.1 : ℤ) else (m.2.2.2.1 : ℤ)).sum
    BitVec.ofInt 32 (if sat then max (-2 ^ 31) (min (2 ^ 31 - 1) s) else s), ?_⟩
  intro c a b ha hb
  have hka : fa.rowLen k = k := by cases fa <;> cases fb <;> simp_all [intPair, Fmt.rowLen]
  have hkb : fb.rowLen k = k := by cases fa <;> cases fb <;> simp_all [intPair, Fmt.rowLen]
  rw [hka] at ha
  rw [hkb] at hb
  rw [List.take_of_length_le (le_of_eq ha), List.take_of_length_le (le_of_eq hb)]
  have hp : ∀ x y, (if (mulOut fa fb x y).2.2.1 = true then -((mulOut fa fb x y).2.2.2.1 : ℤ)
      else ((mulOut fa fb x y).2.2.2.1 : ℤ)) = fa.intVal x * fb.intVal y := by
    intro x y
    simp only [mulOut, h, ite_true]
    by_cases hn : fa.intVal x * fb.intVal y < 0
    · simp only [hn, decide_true, ite_true]
      rw [Int.ofNat_natAbs_of_nonpos hn.le, neg_neg]
    · simp only [hn, decide_false, Bool.false_eq_true, ite_false]
      exact Int.natAbs_of_nonneg (not_lt.mp hn)
  simp only [List.map_zipWith, hp, intDot]

end Verity.Accel
