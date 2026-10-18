import Proofs.Accounting.TS

/-!
# The headlines' luck from the receipt

`luck_of_receipt_half`: (I7) at `7/8` for a play whose ideal game is a receipt game's, its openings pinned, its rounds'
operands degree 2 on cube bits and its receipt luck at `½`: `OperandLuckCube` gives the operand luck at ¾, and
`LuckOfReceipt` (I7) at `¾ + ¼·½`.
-/

namespace Accounting.SecurityProofs.TS.Headline

open Accounting.TS Accounting.Assumptions.TS Accounting.SecurityProofs.TS Pouw.RelativeSecurity

/-- (I7) at `7/8` from the receipt split: the operand luck at ¾ on cube bits and the receipt luck at `½`. -/
theorem luck_of_receipt_half {α ι : Type} [Fintype α] [Fintype ι] [DecidableEq ι] (R : Type) [CommRing R] {S : ℕ}
    (g : RoundGame (α × Finset ι) S) (G : ReceiptGame (α × Finset ι) S) (hG : g = G.game) (hPin : G.Pinned)
    (hCube : G.CubeOperands R) (hRL : ReceiptLuckAt (1 / 2) G) : LuckAt (7 / 8) g := by
  have h := LuckOfReceipt G (3 / 4) (1 / 2) hPin (OperandLuckCube R G hCube) hRL (by norm_num)
  have e : (3 / 4 : ℝ) + (1 - 3 / 4) * (1 / 2) = 7 / 8 := by norm_num
  rw [e, ← hG] at h
  exact h

end Accounting.SecurityProofs.TS.Headline
