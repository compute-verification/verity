import Definitions.Core.Accel.Power

/-!
# The power premises

The three premises a window's energy bound rests on (`Guarantees.PowerBound`), one per thing a measurement or an
attack can falsify, with no overlap:

* `MeterReads`, the auditor's device: the meter on the machine's power feed logs at least what the window's phases
  draw;
* `EnergyFloor`, the card's device model: a phase draws at least the card's floors at its clock;
* `DeclaredEnergy`, the declared work's: its MACs draw at least their own operands' floor at their clock.
-/

open scoped NNRat

namespace Verity.Assumptions

/-- **`auditor-meter(feed)`**: the auditor's meter on the machine's power feed logs at least the energy the window's
phases draw. The meter is an auditor-trusted device, like a network certifier (`TrustedHost`): it sits on the feed into
the machine, upstream of every card and of everything else the machine powers, on hardware and with a key the prover
doesn't control, and it signs what it logs. The card's own power reading and its power cap are the prover's and are
never read. -/
def MeterReads (w : Accel.Window) : Prop := ∑ i, (w.phase i).e ≤ w.logged

/-- **`energy-floor(card)`**: a phase draws at least its declared MACs' part plus the card's floors at its clock: the
static power over its length, and each tensor unit beyond the declared MACs and each SHA-512 compression at its floor.
The card's device model, falsified by a measurement: a stretch of the card at a clock that draws less. Each floor is
the least the card was read to draw: a tensor unit's on zero operands, a compression's on the cheapest message read,
the static power with the SMs resident and idle. -/
def EnergyFloor (pw : Accel.Power) (p : Accel.Phase) : Prop :=
  p.eDecl + ((pw.static.at p.f * p.t + pw.mac.at p.f * p.macs + pw.comp.at p.f * p.comps : ℚ≥0) : ℝ) ≤ p.e

/-- **`declared-energy(card)`**: a phase's declared MACs draw at least the declared operands' floor each, at its clock.
The declared work's premise: its operands are uniformly random (the noised product), and it is falsified by a way to
compute that product on the card that draws less than `s8` MMAs on uniformly random operands; a costlier way meets
it. -/
def DeclaredEnergy (pw : Accel.Power) (p : Accel.Phase) : Prop :=
  ((pw.macDecl.at p.f * p.decl : ℚ≥0) : ℝ) ≤ p.eDecl

end Verity.Assumptions
