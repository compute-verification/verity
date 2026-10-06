import Definitions.Flock.Accounting.Bound
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.GCongr

/-! The definitions of `Proofs.Flock.Soundness.Accounting.Fast100` that C-Flock's guarantees read, or that a
definition here uses. The lemmas about them stay there. -/

namespace FlockSoundness.Accounting

open Real

/-- The level conditions the rational bounds need: rates `2^-1 … 2^-7`, at least two message
columns. Every `fast100` level satisfies them (checked below). -/
def Level.Ok (l : Level) : Prop := 1 ≤ l.logInvRate ∧ l.logInvRate ≤ 7 ∧ 1 ≤ l.logCols

instance (l : Level) : Decidable l.Ok := by unfold Level.Ok; infer_instance

/-- The statement shapes covered: at most `2^27` witness bits per block, 1024 public regions,
64 coordinates per link point. -/
def Stmt.InRange (st : Stmt) : Prop := st.kLog ≤ 27 ∧ st.nRegions ≤ 1024 ∧ st.mPts ≤ 64

end FlockSoundness.Accounting
