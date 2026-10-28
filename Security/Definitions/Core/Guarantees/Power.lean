import Definitions.Core.Assumptions.Power

/-!
# What a window's energy can hold

On a card with energy floors `pw`, the tensor work beyond the declared MACs that a window the verifier accepts can hold,
each tensor unit at its floor at its clock, is at most the window's energy budget less the declared work's floor
energy (`PowerBound`). This bounds energy only: the clocks are the run's own, and a bound on the work in units, which
needs the card's rate at each clock too, is not stated here.
-/

open scoped NNRat

namespace Verity.Guarantees

open Verity.Accel Verity.Assumptions

/-- **The hidden tensor work an accepted window can hold, by energy**, on a card with energy floors `pw`: for every
window `w` the verifier accepts at the energy budget `budget` (W), where the meter logs at least what the phases draw
(`MeterReads`), each phase draws at least the card's floors at its clock (`EnergyFloor`) and its declared MACs at least
their operands' floor (`DeclaredEnergy`), the tensor units beyond the declared MACs, each at its floor at its phase's
clock, draw at most the budget over the window less the declared work's floor energy (the static power over each
phase, and each declared MAC and each compression at its floor at its phase's clock), or the verifier took a reading
the meter didn't log: a verifying signature on a reading missing from the meter's log. -/
def PowerBound (pw : Power) : Prop :=
  ∀ (w : Window) (budget : ℝ), w.Accepted budget → MeterReads w → (∀ i, EnergyFloor pw (w.phase i)) →
    (∀ i, DeclaredEnergy pw (w.phase i)) →
    ∑ i, ((pw.mac.at (w.phase i).f * (w.phase i).macs : ℚ≥0) : ℝ) ≤
        budget * w.T - ∑ i, ((pw.static.at (w.phase i).f * (w.phase i).t +
          pw.macDecl.at (w.phase i).f * (w.phase i).decl + pw.comp.at (w.phase i).f * (w.phase i).comps : ℚ≥0) : ℝ) ∨
      w.reading ≠ w.logged

end Verity.Guarantees
