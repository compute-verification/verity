module

public import Flock.Tags
public import Flock.Zk
public import Proofs.Flock.Soundness.Discharge.Exec.Event
public import Proofs.Flock.Soundness.Discharge.ZkHidden.DefsJ

/-!
# The inputs `EndToEndRegistered`'s witnesses are stated at

`verify --zk --registered`'s inputs at the templated statement (`verity/flock-circuit/tmpl`'s tags), one table, on the
coin tree `verify --zk` samples. `EndToEndRegistered_honest` concludes `TagsOkZHJ` there, and
`EndToEndRegistered_nontrivial` exhibits there a read that is not checked.
-/

@[expose] public section

namespace FlockSoundness.Discharge.ZkReg.Witness

open Exec

/-- **`verify --zk --registered`'s inputs at the templated statement**: `verity/flock-circuit/tmpl`'s tags, one table, no
session tables, on the coin tree `verify --zk` samples. The files are empty: no premise the witnesses conclude reads
them. -/
def ins : Inputs :=
  { tags := Flock.Tags.circuitTmpl, circuitFile := .empty, publicFile := .empty,
    coins := .tree (Flock.Zk.coinSpec (Flock.Tags.circuitTmpl.tableNames 1)), count := 1, tables := [],
    partition := none, program := none, work := none, stratified := none, closure := none }

theorem ins_tagsOk : ZkHidden.TagsOkZHJ ins := ⟨rfl, rfl, rfl⟩

theorem ins_count : ins.count ≤ 1 := by decide

end FlockSoundness.Discharge.ZkReg.Witness
