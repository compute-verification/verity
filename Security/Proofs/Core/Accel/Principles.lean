import Definitions.Core.Accel.Principles
import Mathlib.Tactic.NormNum

/-!
# The accelerator model's principles, proved

P2's three halves that need no capture: an MMA's output reads only its own row, column and accumulator (`p2Local`), its
outputs are distinct products' destinations (`p2Distinct`), and every opcode takes at least a lane's share of an issue
slot (`p2Priced`).
-/

open scoped NNRat

namespace Verity.Accel

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

end Verity.Accel
