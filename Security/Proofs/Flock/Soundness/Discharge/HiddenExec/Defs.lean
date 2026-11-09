module

public import Aesop.BuiltinRules
public import Flock.HmRow
public import Proofs.Flock.Soundness.Discharge.Exec
import Mathlib.Algebra.Order.Floor.Semifield
import Mathlib.CategoryTheory.Category.Init
import Mathlib.Combinatorics.Matroid.Init
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.Tactic.Positivity.Finset
import VCVio.Prelude.Core

/-!
# The statement flock-verify sets up for one table, under either tag

`Stmt.setupTables` sets up each table with its `setup`: `Stmt.setupHidden` under a tag with hidden outputs
(`Tags.hiddenOutputs`: `verity/flock-circuit`, the default `--statement`, and `verity/flock-circuit/types`), and
`Stmt.setupH` otherwise (`verity/flock-circuit@967b8d06` and the older tags). `setupOne` is that `setup`, reading the flag
off the tags it is given; `setupTables` gives each table its session's tags with only `identity` changed
(`Exec.tagsAt`), so the flag is the session's.
-/

@[expose] public section

namespace FlockSoundness.Discharge.HiddenExec

open Flock
open Lean (Json)

/-- **The statement flock-verify sets up for one table** (`Stmt.setupTables`' `setup`): `Stmt.setupHidden` under a tag
with hidden outputs, `Stmt.setupH` otherwise. -/
def setupOne (tags : Tags) (cf pf : ByteArray) (coins : Coins) (tables : List (String × ByteArray))
    (draw partition : Option Json) (program : Option ByteArray) : V Stmt :=
  if tags.hiddenOutputs then Stmt.setupHidden tags cf pf coins tables draw partition program
  else Stmt.setupH tags cf pf coins tables draw partition program

end FlockSoundness.Discharge.HiddenExec
