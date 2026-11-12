module

public import Proofs.Core.NaiveStrassen.Check
public import Proofs.Core.NaiveStrassen.Size

/-!
# The kernel evaluation (NaiveStrassen)

`checkAll = true` is a pure kernel check: `decide +kernel` has the kernel compute the tables and evaluate the check,
with GMP arithmetic on the packed rows. No compiled code, `native_decide` or `#eval` is involved, and no axiom beyond
the kernel's own. At depth 13 it takes about 80 s and 25 GiB.
-/

@[expose] public section

namespace Verity.Accel.NaiveStrassen.Packed

open Verity.Accel.NaiveStrassen

/-- The check passes. A pure kernel check (`decide +kernel`). -/
theorem checkAll_eq_true : checkAll = true := by decide +kernel

/-- Theorem 1 to depth `thmL = 12` on every table the sheet dominates, from the one table. -/
theorem naiveBeatsStrassen_of_dom {fs : List Atom} (hD : Dom fs) : NaiveBeatsStrassen fs thmL :=
  beats_of_check checkAll_eq_true hD

/-- The size form at `c = 4096` on every table the sheet dominates, from the one table. -/
theorem naiveBeatsStrassenUpTo_of_dom {fs : List Atom} (hD : Dom fs) : NaiveBeatsStrassenUpTo fs 4096 :=
  upTo_of_check checkAll_eq_true hD (weight_pos_of_dom hD)

end Verity.Accel.NaiveStrassen.Packed
