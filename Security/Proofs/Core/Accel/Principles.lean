import Definitions.Core.Accel.Principles
import Mathlib.Tactic.NormNum

/-!
# The accelerator model's principles, proved

The integer path is exact on the s32 range, saturating or not (`p1Int`). P2's halves that need no capture: an MMA's
output reads only its own row, column and accumulator (`p2Local`), its outputs are distinct products' destinations
(`p2Distinct`), an opcode writes no word but its outputs (`p2Frame`), and every opcode takes at least a lane's share of an issue slot (`p2Priced`). Each card refines its coarse `Device` (`H200.refines`, `Sm120.refines`).
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

theorem p2Distinct : P2Distinct := by
  intro s o o' _ _ hq hr
  rw [← Nat.div_add_mod o s.n, ← Nat.div_add_mod o' s.n, hq, hr]

theorem p2Priced : P2Priced := by
  have h : (1 / 32 : ℚ≥0) ≤ 1 := by rw [← NNRat.coe_le_coe]; norm_num
  intro weight scale o
  cases o with
  | mma fa fb _ sh sa sb => simpa [opUse, port] using h
  | lane g => simp [opUse]
  | uni g => simpa [opUse] using h
  | copy s t n => cases s <;> cases t <;> simp [opUse, port]
  | atom s _ => cases s <;> simp [opUse, port]

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

end Verity.Accel
