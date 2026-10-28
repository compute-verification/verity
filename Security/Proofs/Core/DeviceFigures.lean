import Definitions.Core.DeviceFigures

/-!
# The figures a bound reads

`DeviceFigures.bounding` has at least the capacity of the stated figures and of the estimate's
(`DeviceFigures.capacityLE_bounding`, `DeviceFigures.estimate_capacityLE_bounding`). An accepted device's figures are
within tolerance of `bounding`'s (`DeviceFigures.Accepted.within_bounding`), and `bounding`'s are themselves accepted
against the estimate (`DeviceFigures.Accepted.bounding`), so the bound reads at most the tolerance more capacity than the
estimate. Registration accepts the estimate's own figures (`DeviceFigures.accepted_self`), so `Accepted` is not vacuous,
and it refuses a toy device stated ten percent slow or ten percent fast.
-/

namespace Verity

open scoped NNRat

theorem withinTolerance_self (τ e : ℚ≥0) : WithinTolerance τ e e :=
  ⟨mul_le_of_le_one_left zero_le tsub_le_self, le_mul_of_one_le_left zero_le le_self_add⟩

theorem WithinTolerance.max_right {τ e x : ℚ≥0} (h : WithinTolerance τ e x) : WithinTolerance τ e (max x e) :=
  ⟨(withinTolerance_self τ e).1.trans (le_max_right _ _), max_le h.2 (withinTolerance_self τ e).2⟩

theorem WithinTolerance.min_right {τ e x : ℚ≥0} (h : WithinTolerance τ e x) : WithinTolerance τ e (min x e) :=
  ⟨le_min h.1 (withinTolerance_self τ e).1, (min_le_right _ _).trans (withinTolerance_self τ e).2⟩

theorem WithinTolerance.of_max {τ e x : ℚ≥0} (h : WithinTolerance τ e x) : WithinTolerance τ (max x e) x := by
  refine ⟨?_, (le_max_left _ _).trans (withinTolerance_self τ _).2⟩
  rcases le_total x e with hxe | hex
  · rw [max_eq_right hxe]; exact h.1
  · rw [max_eq_left hex]; exact (withinTolerance_self τ x).1

theorem WithinTolerance.of_min {τ e x : ℚ≥0} (h : WithinTolerance τ e x) : WithinTolerance τ (min x e) x := by
  refine ⟨(withinTolerance_self τ _).1.trans (min_le_left _ _), ?_⟩
  rcases le_total x e with hxe | hex
  · rw [min_eq_left hxe]; exact (withinTolerance_self τ x).2
  · rw [min_eq_right hex]; exact h.2

namespace DeviceFigures

theorem CapacityLE.refl (a : DeviceFigures) : a.CapacityLE a :=
  ⟨le_rfl, fun _ => le_rfl, fun _ => le_rfl, le_rfl⟩

theorem CapacityLE.trans {a b c : DeviceFigures} (hab : a.CapacityLE b) (hbc : b.CapacityLE c) : a.CapacityLE c :=
  ⟨hab.1.trans hbc.1, fun o => (hab.2.1 o).trans (hbc.2.1 o), fun o => (hbc.2.2.1 o).trans (hab.2.2.1 o),
    hab.2.2.2.trans hbc.2.2.2⟩

/-- The figures the bound reads have at least the stated figures' capacity. -/
theorem capacityLE_bounding (est : DeviceEstimate) (d : DeviceFigures) : d.CapacityLE (d.bounding est) :=
  ⟨le_max_left _ _, fun _ => le_max_left _ _, fun _ => min_le_left _ _, le_max_left _ _⟩

/-- The figures the bound reads have at least the estimate's capacity. -/
theorem estimate_capacityLE_bounding (est : DeviceEstimate) (d : DeviceFigures) :
    est.figures.CapacityLE (d.bounding est) :=
  ⟨le_max_right _ _, fun _ => le_max_right _ _, fun _ => min_le_right _ _, le_max_right _ _⟩

/-- Registration accepts the estimate's own figures. -/
theorem accepted_self (est : DeviceEstimate) : est.figures.Accepted est :=
  ⟨rfl, withinTolerance_self _ _, fun _ => withinTolerance_self _ _, fun _ => withinTolerance_self _ _,
    withinTolerance_self _ _⟩

/-- The figures the bound reads for an accepted device are themselves within tolerance of the estimate's. -/
theorem Accepted.bounding {est : DeviceEstimate} {d : DeviceFigures} (h : d.Accepted est) :
    (d.bounding est).Accepted est :=
  ⟨h.1, h.2.1.max_right, fun c => (h.2.2.1 c).max_right, fun c => (h.2.2.2.1 c).min_right, h.2.2.2.2.max_right⟩

/-- An accepted device's figures are within tolerance of the figures the bound reads. -/
theorem Accepted.within_bounding {est : DeviceEstimate} {d : DeviceFigures} (h : d.Accepted est) :
    d.Accepted ⟨d.bounding est, est.tolerance⟩ :=
  ⟨rfl, h.2.1.of_max, fun c => (h.2.2.1 c).of_max, fun c => (h.2.2.2.1 c).of_min, h.2.2.2.2.of_max⟩

/-- A toy estimate: clock 100, every peak rate 10, every energy floor 1 and power cap 50, at tolerances of a twentieth
and a tenth. -/
def toyEstimate : DeviceEstimate where
  figures := { sku := "toy", clock := 100, peakRate := fun _ => 10, energyFloor := fun _ => 1, powerCap := 50 }
  tolerance := { capacity := 1 / 20, floor := 1 / 10 }

-- `Rat.blt` is irreducible to the elaborator's `decide`; the kernel evaluates it.
example : toyEstimate.figures.Accepted toyEstimate := by decide +kernel

example : ¬ ({ toyEstimate.figures with clock := 90 } : DeviceFigures).Accepted toyEstimate := by decide +kernel

example : ¬ ({ toyEstimate.figures with clock := 110 } : DeviceFigures).Accepted toyEstimate := by decide +kernel

end DeviceFigures

end Verity
