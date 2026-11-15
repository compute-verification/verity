module

public import Proofs.Flock.Claims.DeployCode

/-!
# A table's own error at a statement

`tableErrJ`: `tableErr` at the statement's table `j`, `0` past its tables, beside `ε_table`, which is these summed
over the statement's tables. It is here, on the deployed records' numbers alone, so that a statement at one table of a
session (`Flock.Guarantees.MaskedKSTable`) reads it without the proof that the record meets it (`ZkKSTable`).
-/

@[expose] public section

namespace FlockDeploy.KS

open Flock

namespace Gen

variable [S : ZkSetup]

/-- **A table's own error at a statement**: `tableErr` at the statement's table `j`, `0` past its tables. `ε_table`
is these summed over the statement's tables. -/
noncomputable def tableErrJ (zk : Bool) (x : Claims.Msg) (j : ℕ) : ℝ :=
  ((tables zk x)[j]?).elim 0 fun p => tableErr zk p.1 p.2

end Gen

/-- **A table's own error at a statement**: `tableErr` at the statement's table `j`, `0` past its tables. `ε_table`
is these summed over the statement's tables. -/
noncomputable def tableErrJ (zk : Bool) (x : Claims.Msg) (j : ℕ) : ℝ :=
  ((tables zk x)[j]?).elim 0 fun p => tableErr zk p.1 p.2

end FlockDeploy.KS
