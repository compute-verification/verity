import Proofs.Flock.Claims.Recursion.VStarCross
import Proofs.Flock.Claims.Complete.Walk

/-!
# Flock's admission of plain leaves

The one fact the recursion's deployed stop takes of its inner statement (`ZKStop_meets_ready`), and nothing proves:
some concrete inner statement passes the checked reader, is small enough for Flock's completeness, and is in Flock's
scope and true. `DeployChecked.lean` proves what follows from it.
-/

namespace RecDeploy

open Claims (Msg)
open Lean (Json)

/-- **Flock admits a plain-leaves statement of the inner shape** (what #2411 and #2468 make possible): some inner
statement passes every check the checked reader makes of it (it decodes, its partition reads, the inner setup at the
recursion's inner tags accepts it, `RecVStar.innerOf`, and Flock's plain verifier sets it up, its statement check
passes and it carries a draw, `innerOk`), is under `2^56` bytes with fewer than `2^128` zerocheck outer coins, has
tables whose regions fit (`FlockComplete.RegionsFit`) and an honest record that fits (`FlockComplete.RecordFitsAt`)
(so Flock's completeness error is below 1 there, `FlockComplete.δc_sha512_lt_one`), is in Flock's scope, where its
soundness headline holds (`Claims.Flock.Scope`), and is true (`Claims.Flock.R`). Every conjunct is a fact about one
concrete inner statement and its witness: #2468's inner session is one, as its run `r20261018-053343-99cf` measured
(Lean `verify` without `--zk` accepts it, its record holding the full stratified draw), and none is proved here. Only
an inner statement whose public file holds each instance's own rows can be one: Flock's statement check refuses a
shared-row public file at plain leaves (`ProvedScope.sharedRowsOk` passes one only at `hm96-sha512/v1` leaves under
`--registered F`), which `RecVStar.innerOf` sets up. -/
def PlainLeavesAdmitted (rel : Flock.Readable.Parsed → Flock.Readable.Relation) : Prop :=
  ∃ (inner : Msg) (ri : FlockDeploy.Raw) (partition : Option Json) (dims : Flock.RecVStar.Inner)
    (w : Claims.Flock.Wit) (s : FlockDeploy.Ins) (spec : Flock.SessionSpec) (sts : Array Flock.Stmt),
    FlockDeploy.decode inner = some ri ∧ (ri.partition.mapM (FlockDeploy.jsonOf "partition")).toOption = some partition ∧
      Flock.RecVStar.innerOf ri.circuit ri.publicFile partition ri.program = .ok dims ∧
      FlockDeploy.setupOf false inner = some (s, spec, sts) ∧ s.read.check = .ok () ∧ s.read.draw.isSome = true ∧
      inner.length < 2 ^ 56 ∧ (FlockComplete.outerSlots inner).length < 2 ^ 128 ∧
      (∀ st ∈ sts, FlockComplete.RegionsFit st) ∧ FlockComplete.RecordFitsAt FlockDeploy.sha512 inner spec sts ∧
      Claims.Flock.Scope (FlockDeploy.deployPlain rel) inner ∧ Claims.Flock.R (FlockDeploy.deployPlain rel).parse inner w

end RecDeploy
