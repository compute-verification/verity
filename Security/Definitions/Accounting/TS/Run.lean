import Mathlib

/-!
# A run (trusted)

A program's run on the device, per unit of honest work: its instruction counts per class `K`, normalized per SMSP (its
device totals of each class divided by the SMSPs), its cycles per SMSP and its SM clock. Its time is cycles over clock.
A cheater's labelled run (`Anatomy`) and the plays and periods built on it read it.
-/

namespace Accounting.TS

/-- A run of a program on the device, per unit of honest work: its instruction counts `n` (normalized per SMSP), its
cycles `cycles` per SMSP and its SM clock `clock`. -/
structure Run (K : Type) where
  n : K → ℝ
  cycles : ℝ
  clock : ℝ

variable {K : Type} [Fintype K]

/-- A run's time: its cycles over its clock. -/
noncomputable def Run.time (P : Run K) : ℝ := P.cycles / P.clock

end Accounting.TS
