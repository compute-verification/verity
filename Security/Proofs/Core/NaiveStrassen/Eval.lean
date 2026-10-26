import Proofs.Core.NaiveStrassen.Sound

/-!
# The kernel evaluation (NaiveStrassen)

`checkAll = true` is a pure kernel check: `decide +kernel` has the kernel compute the tables and evaluate the check,
with GMP arithmetic on the packed rows. No compiled code, `native_decide` or `#eval` is involved, and no axiom beyond
the kernel's own.
-/

namespace Verity.Accel.NaiveStrassen.Packed

open Verity.Accel.NaiveStrassen

/-- The check passes. A pure kernel check (`decide +kernel`). -/
theorem checkAll_eq_true : checkAll = true := by decide +kernel

theorem dom_rtx6000Sheet : Dom rtx6000Sheet := fun _ hf _ _ h => atomMin_le hf h

/-- Theorem 1 on the RTX PRO 6000's data sheet, to depth `thmL`. -/
theorem naiveBeatsStrassen_rtx6000Sheet : NaiveBeatsStrassen rtx6000Sheet thmL := fun _ hl _ hv hf =>
  (bound checkAll_eq_true dom_rtx6000Sheet (Nat.le_trans hl (by decide)) hv).1 hl hf

end Verity.Accel.NaiveStrassen.Packed
