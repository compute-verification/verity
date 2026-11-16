module

public import Proofs.Flock.Claims.Complete.ZkTables.RecordSetup

@[expose] public section
open Flock

example : ∀ t ∈ Tags.all ++ Tags.templated, t.hm96Rows = true → t.typed = false ∨ t.hiddenOutputs = true := by
  decide +kernel
