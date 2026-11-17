module

public import Definitions.Core.DeviceFigures
public import Mathlib.Algebra.Field.Rat
import Mathlib.Algebra.Order.Field.Rat
import Mathlib.Algebra.Order.Ring.NNRat
import Mathlib.Data.Finset.Attr

/-! The figures a bound reads, and registration's acceptance of a device's figures against its estimate. -/

@[expose] public section

namespace Verity

open scoped NNRat

theorem OperandClass.lo_le_hi (c : OperandClass) : c.lo ≤ c.hi := by
  cases c <;> decide

theorem OperandClass.range_s8 (c : OperandClass) : -128 ≤ c.lo ∧ c.hi ≤ 127 := by
  cases c <;> decide

theorem withinTolerance_self (τ e : ℚ≥0) : WithinTolerance τ e e :=
  ⟨mul_le_of_le_one_left zero_le tsub_le_self, le_mul_of_one_le_left zero_le le_self_add⟩

theorem WithinTolerance.max_right {τ e x : ℚ≥0} (h : WithinTolerance τ e x) : WithinTolerance τ e (max x e) :=
  ⟨(withinTolerance_self τ e).1.trans (le_max_right _ _), max_le h.2 (withinTolerance_self τ e).2⟩

theorem WithinTolerance.of_max {τ e x : ℚ≥0} (h : WithinTolerance τ e x) : WithinTolerance τ (max x e) x := by
  refine ⟨?_, (le_max_left _ _).trans (withinTolerance_self τ _).2⟩
  rcases le_total x e with hxe | hex
  · rw [max_eq_right hxe]; exact h.1
  · rw [max_eq_left hex]; exact (withinTolerance_self τ x).1

namespace DeviceFigures

theorem CapacityLE.refl (a : DeviceFigures) : a.CapacityLE a :=
  ⟨le_rfl, fun _ => le_rfl⟩

theorem CapacityLE.trans {a b c : DeviceFigures} (hab : a.CapacityLE b) (hbc : b.CapacityLE c) : a.CapacityLE c :=
  ⟨hab.1.trans hbc.1, fun o => (hab.2 o).trans (hbc.2 o)⟩

theorem capacityLE_bounding (est : DeviceEstimate) (d : DeviceFigures) : d.CapacityLE (d.bounding est) :=
  ⟨le_max_left _ _, fun _ => le_max_left _ _⟩

theorem estimate_capacityLE_bounding (est : DeviceEstimate) (d : DeviceFigures) :
    est.figures.CapacityLE (d.bounding est) :=
  ⟨le_max_right _ _, fun _ => le_max_right _ _⟩

theorem accepted_self (est : DeviceEstimate) (h : ∀ c, 0 < est.figures.peakRate c) : est.figures.Accepted est :=
  ⟨rfl, withinTolerance_self _ _, fun _ => withinTolerance_self _ _, h⟩

theorem Accepted.bounding {est : DeviceEstimate} {d : DeviceFigures} (h : d.Accepted est) :
    (d.bounding est).Accepted est :=
  ⟨h.1, h.2.1.max_right, fun c => (h.2.2.1 c).max_right, fun c => (h.2.2.2 c).trans_le (le_max_left _ _)⟩

theorem Accepted.within_bounding {est : DeviceEstimate} {d : DeviceFigures} (h : d.Accepted est) :
    d.Accepted ⟨d.bounding est, est.tolerance⟩ :=
  ⟨rfl, h.2.1.of_max, fun c => (h.2.2.1 c).of_max, h.2.2.2⟩

/-- A toy estimate: clock 100 and every peak rate 10, at a tolerance of a twentieth. -/
def toyEstimate : DeviceEstimate where
  figures := { sku := "toy", clock := 100, peakRate := fun _ => 10 }
  tolerance := { capacity := 1 / 20 }

-- `Rat.blt` is irreducible to the elaborator's `decide`; the kernel evaluates it.
example : toyEstimate.figures.Accepted toyEstimate := by decide +kernel

example : ¬ ({ toyEstimate.figures with clock := 90 } : DeviceFigures).Accepted toyEstimate := by decide +kernel

example : ¬ ({ toyEstimate.figures with clock := 110 } : DeviceFigures).Accepted toyEstimate := by decide +kernel

-- At a tolerance of 1 a zero rate is within tolerance, and registration still refuses it.
example : ¬ ({ toyEstimate.figures with peakRate := fun _ => 0 } : DeviceFigures).Accepted
    { toyEstimate with tolerance := { capacity := 1 } } := by decide +kernel

end DeviceFigures

end Verity
