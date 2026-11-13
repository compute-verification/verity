module

public import Proofs.Flock.SpotFlock.Code
public import Proofs.Flock.SessionLoop.Concrete.Stmt
public import Proofs.Properties.Flock.Nets
import Flock.JsonParse
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.List.Control
import all Init.Data.Range.Basic
import all Init.Data.Repr
import all Init.Data.Ord.Basic
import all Init.Data.Ord.String
import all Init.Data.String.Defs
import all Init.Data.String.Basic
import all Init.Data.String.Decode
import all Init.Data.String.PosRaw
import all Init.Data.String.Bootstrap
import all Init.Data.String.Extra
import all Init.Data.String.Legacy
import all Lean.Data.Json.Basic
import all Lean.Data.Json.FromToJson.Basic
import all Std.Data.TreeMap.Raw.Basic
import all Std.Data.DTreeMap.Raw.Basic
import all Std.Data.DTreeMap.Internal.Operations
import all Std.Data.DTreeMap.Internal.Queries
import all Std.Data.DTreeMap.Internal.Balancing
import all Std.Data.DTreeMap.Internal.Def

/-!
# Kernel probes for a literal statement at SPOT[Flock]'s two-stage parser

A scratch module: the three evaluations a literal registration and statement at `SpotFlock.parse` would rest on, each
left to the kernel.

* `jsonMsg_null`: the four bytes of `null` read as JSON, so the parser takes them at every registration.
* `decode_empty`: the empty descriptor does not decode.
* `zkOk_concrete`: the concrete session statement `FlockDeploy.concreteStmt` is one `SpotFlock.Held` takes as a
  template's reference statement.
-/

@[expose] public section

namespace ScanFlattensTry

theorem jsonMsg_null : (SpotFlock.jsonMsg [110, 117, 108, 108]).isSome = true := by decide +kernel

theorem decode_empty : (Flock.Program.decode ⟨#[]⟩).toOption.isSome = false := by decide +kernel

theorem zkOk_concrete : SpotFlock.zkOk FlockDeploy.concreteStmt = true := by
  rw [SpotFlock.zkOk, FlockDeploy.concreteStmt_setupOf Properties.Flock.Nets.fixedNetsBuild]
  decide +kernel

end ScanFlattensTry
