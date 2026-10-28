import Proofs.Core.NaiveStrassen.Size

/-!
# The kernel evaluation (NaiveStrassen)

`checkAll = true` is a pure kernel check: `decide +kernel` has the kernel compute the tables and evaluate the check,
with GMP arithmetic on the packed rows. No compiled code, `native_decide` or `#eval` is involved, and no axiom beyond
the kernel's own. At depth 13 it takes about 80 s and 25 GiB.
-/

namespace Verity.Accel.NaiveStrassen.Packed

open Verity.Accel.NaiveStrassen

/-- The check passes. A pure kernel check (`decide +kernel`). -/
theorem checkAll_eq_true : checkAll = true := by decide +kernel

/-- Theorem 1 on the four instances to depth `thmL = 12`, from the one table. -/
theorem naiveBeatsStrassen_all :
    NaiveBeatsStrassen h200Sheet thmL ∧ NaiveBeatsStrassen h200Measured thmL ∧
      NaiveBeatsStrassen rtx6000Sheet thmL ∧ NaiveBeatsStrassen rtx6000Measured thmL :=
  ⟨beats_of_check checkAll_eq_true dom_h200Sheet, beats_of_check checkAll_eq_true dom_h200Measured,
    beats_of_check checkAll_eq_true dom_rtx6000Sheet, beats_of_check checkAll_eq_true dom_rtx6000Measured⟩

/-- The size form on the four instances at `c = 4096`, from the one table. -/
theorem naiveBeatsStrassenUpTo_all :
    NaiveBeatsStrassenUpTo h200Sheet 4096 ∧ NaiveBeatsStrassenUpTo h200Measured 4096 ∧
      NaiveBeatsStrassenUpTo rtx6000Sheet 4096 ∧ NaiveBeatsStrassenUpTo rtx6000Measured 4096 :=
  ⟨upTo_of_check checkAll_eq_true dom_h200Sheet (by decide),
    upTo_of_check checkAll_eq_true dom_h200Measured (by decide),
    upTo_of_check checkAll_eq_true dom_rtx6000Sheet (by decide),
    upTo_of_check checkAll_eq_true dom_rtx6000Measured (by decide)⟩

end Verity.Accel.NaiveStrassen.Packed
