module

public import Proofs.Flock.Soundness.Discharge.Hidden.Binding
public import Proofs.Flock.Soundness.Discharge.Hidden.Committed
public import Proofs.Flock.Soundness.Discharge.Hidden.Defs
public import Proofs.Flock.Soundness.Discharge.Hidden.Public
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Proofs.Flock.Soundness.Discharge.Hidden.Commit
import Proofs.Flock.Soundness.Discharge.Hidden.Delta
import Proofs.Flock.Soundness.Discharge.Hidden.Spec

/-! The soundness of a hidden-output statement (`Stmt.setupHidden`): `Defs.lean`'s frozen statements, one theorem each. -/

@[expose] public section

namespace FlockSoundness.Discharge.Hidden

theorem setupHidden_spec : SetupHiddenSpec := fun h => setupHidden_spec' h

theorem hidden_delta : HiddenDelta := hidden_delta'

theorem hidden_layout : HiddenLayout := hidden_layout'

theorem public_exposes : PublicExposes := public_exposes'

theorem outputs_committed : OutputsCommitted := outputs_committed'

theorem commitPair_spec : CommitPairSpec := commitPair_spec'

theorem outputs_binding : OutputsBinding := outputs_binding_of outputs_committed

end FlockSoundness.Discharge.Hidden
