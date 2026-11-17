module

public import Proofs.Core.NaiveStrassen.Check
public import Proofs.Core.NaiveStrassen.Size

/-! The kernel's evaluation of NaiveStrassen's check, with no compiled code, `native_decide` or `#eval`. -/

@[expose] public section

namespace Verity.Accel.NaiveStrassen.Packed

open Verity.Accel.NaiveStrassen

theorem checkAll_eq_true : checkAll = true := by decide +kernel

theorem naiveBeatsStrassen_of_dom {fs : List Atom} (hD : Dom fs) : NaiveBeatsStrassen fs thmL :=
  beats_of_check checkAll_eq_true hD

theorem naiveBeatsStrassenUpTo_of_dom {fs : List Atom} (hD : Dom fs) : NaiveBeatsStrassenUpTo fs 4096 :=
  upTo_of_check checkAll_eq_true hD (weight_pos_of_dom hD)

end Verity.Accel.NaiveStrassen.Packed
