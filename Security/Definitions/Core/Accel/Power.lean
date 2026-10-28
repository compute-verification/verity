import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Basic.Real.Basic
import Mathlib.Data.NNRat.Defs

/-!
# A card's energy (trusted)

The power side of the accelerator model: a card's energy floors against its SM clock (`Power`), and a window of a run as
the auditor's meter on the machine's power feed sees it (`Window`).

The SM clock is a variable of the run. The H200 holds 1830 MHz while its tensor pipe runs and 1980 MHz for ALU or hash
work alone, and runs lower under its power cap or a lock. Each floor is a table of rows against that clock (`Rows`): a
row `(f, e)` says the floor is at least `e` at every clock from `f` up, so the floor at a clock is the largest row at or
below it, and `0` below every row (`Rows.at`). The rows are measured; what makes them floors is the premise that takes
them (`Assumptions.EnergyFloor`, `Assumptions.DeclaredEnergy`).

A window is cut into phases, each a stretch of one card at one clock (`Phase`): the work it does (its declared MACs,
the tensor units beyond them, its SHA-512 compressions) and the energy it draws, all of it (`e`) and its declared MACs'
part (`eDecl`). The verifier reads the meter's signed reading for the window and refuses a window whose reading passes
its energy budget, a power fixed before the period, over the window's length (`Window.Accepted`).
-/

open scoped NNRat

namespace Verity.Accel

/-- **A floor against the SM clock**: rows `(f, e)`, the clock `f` in Hz, each a floor of `e` from `f` up. -/
abbrev Rows : Type := List (ℚ≥0 × ℚ≥0)

/-- The floor at the clock `f`: the largest row at or below `f`, and `0` below every row. -/
def Rows.at (rows : Rows) (f : ℚ≥0) : ℚ≥0 :=
  ((rows.filter fun r => r.1 ≤ f).map Prod.snd).foldr max 0

/-- **A card's energy floors**, each against the SM clock. -/
structure Power where
  /-- the card's static power, W: what it draws with its SMs resident and idle -/
  static : Rows
  /-- a tensor unit's energy, J, in any format and on any operands -/
  mac : Rows
  /-- a declared MAC's energy, J: an `s8` MAC on the declared work's operands, uniformly random -/
  macDecl : Rows
  /-- a SHA-512 compression's energy, J, on any message -/
  comp : Rows

/-- **A phase of a run**: a stretch of one card at one SM clock, the work it does and the energy it draws. -/
structure Phase where
  /-- the SM clock, Hz -/
  f : ℚ≥0
  /-- its length, s -/
  t : ℚ≥0
  /-- the declared MACs it does -/
  decl : ℚ≥0
  /-- the tensor units it uses beyond its declared MACs' (`Card.use`) -/
  macs : ℚ≥0
  /-- the SHA-512 compressions it computes -/
  comps : ℚ≥0
  /-- the energy the card draws in it, J -/
  e : ℝ
  /-- the part of `e` its declared MACs draw, J -/
  eDecl : ℝ

/-- **A window of a run** under the auditor's meter: its length, its phases (every card's under the meter, each card's
one after another), the energy the meter logged for it and signed, and the reading the verifier took from the meter's
signed record. -/
structure Window where
  /-- its length, s -/
  T : ℚ≥0
  /-- the number of phases -/
  n : ℕ
  /-- its phases -/
  phase : Fin n → Phase
  /-- the energy the meter logged for the window and signed, J -/
  logged : ℝ
  /-- the energy the verifier read from the meter's signed record, J -/
  reading : ℝ

/-- **The verifier's check on a window**: the meter's reading is within the energy budget, a power `budget` (W) fixed
before the period, over the window's length. A window past it is refused. -/
def Window.Accepted (budget : ℝ) (w : Window) : Prop := w.reading ≤ budget * w.T

end Verity.Accel
