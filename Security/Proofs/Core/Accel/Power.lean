import Definitions.Core.Guarantees.Power
import Definitions.Core.Accel.H200Power
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Rat.Cast.CharZero
import Mathlib.Tactic.Linarith

/-!
# The power bound

Each phase draws its declared work's floor plus every other floor (`EnergyFloor`, `DeclaredEnergy`); the phases' energy
sums to at most what the meter logged (`MeterReads`), and an accepted window's reading is at most its budget. So when the
reading the verifier was given is the one the meter logged, the any-operand tensor floors of the window's MACs fit in the
budget less every other floor.
-/

open scoped NNRat

namespace Verity.SecurityProofs

open Verity.Accel Verity.Assumptions

theorem PowerBound (pw : Power) : Verity.Guarantees.PowerBound pw := by
  intro w budget hacc hmeter hfloor hdecl
  by_cases h : w.reading = w.logged
  · left
    have hphase : ∀ i, ((pw.mac.at (w.phase i).f * (w.phase i).macs : ℚ≥0) : ℝ) +
        ((pw.static.at (w.phase i).f * (w.phase i).t + pw.macDecl.at (w.phase i).f * (w.phase i).decl +
          pw.comp.at (w.phase i).f * (w.phase i).comps : ℚ≥0) : ℝ) ≤ (w.phase i).e := by
      intro i
      have h1 := hfloor i
      have h2 := hdecl i
      simp only [EnergyFloor, DeclaredEnergy] at h1 h2
      push_cast at h1 h2 ⊢
      linarith
    have hsum := Finset.sum_le_sum fun i (_ : i ∈ Finset.univ) => hphase i
    rw [Finset.sum_add_distrib] at hsum
    simp only [MeterReads] at hmeter
    simp only [Window.Accepted] at hacc
    linarith
  · exact Or.inr h

/-- The bound at the H200 SXM's floors. -/
theorem PowerBoundH200 : Verity.Guarantees.PowerBound H200.power := PowerBound H200.power

end Verity.SecurityProofs
