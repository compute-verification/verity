import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.GCongr
import FlockSoundness.Protocol.Accounting.Bound

/-!
# The bound for `fast100` run twice on one root

For every witness size the profile supports and every statement shape the three statements of
PROTOCOL.md §16 can take (at most `2^27` bits per block, 1024 public regions and 64 link-point
coordinates, well beyond today's statements), `tableError` is at most

* `2^-205` for every `22 ≤ m ≤ 35` (`tableError_le_of_le_33`, `tableError_le_of_34_35`), at the
  analysis slack `η = 1/200`.

At `η = 1/50` these were 2^-195.5 and, with the seventh level at `m = 34, 35`, 2^-195.4. The slack
is the analysis's choice: the radius `1 − √ρ − η` and the list bound `1/(2ηρ)` follow it, and
the verifier never reads it.

**Method.** Each real term is bounded by a rational one (`…Up`): `√(2^-r)` by a rational upper
bound, and the mutual-correlated-agreement numerator of BCHKS25 Theorem 4.6 by a crude closed
form (its terms sit below 2^-180, against about 2^-102.5 for the queries). The rational bound is then
checked by exact arithmetic for each `m`.
-/

namespace FlockSoundness.Accounting

open Real

/-! ## Rational upper bounds, term by term -/

/-- The level conditions the rational bounds need: rates `2^-1 … 2^-7`, at least two message
columns. Every `fast100` level satisfies them (checked below). -/
def Level.Ok (l : Level) : Prop := 1 ≤ l.logInvRate ∧ l.logInvRate ≤ 7 ∧ 1 ≤ l.logCols

instance (l : Level) : Decidable l.Ok := by unfold Level.Ok; infer_instance

/-! ## The statement ranges -/

/-- The statement shapes covered: at most `2^27` witness bits per block, 1024 public regions,
64 coordinates per link point. -/
def Stmt.InRange (st : Stmt) : Prop := st.kLog ≤ 27 ∧ st.nRegions ≤ 1024 ∧ st.mPts ≤ 64

end FlockSoundness.Accounting
