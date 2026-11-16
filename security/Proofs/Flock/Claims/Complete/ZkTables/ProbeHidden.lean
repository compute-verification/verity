module

public import Proofs.Flock.Claims.Complete.ZkTables.RecordSetup

@[expose] public section
open Flock

example : ∀ t ∈ Tags.all ++ Tags.templated, t.hm96Rows = true →
    t.hiddenOutputs = true ∨ t.name = "verity/flock-circuit+plain-leaves" := by decide +kernel

example : ∃ t ∈ Tags.all ++ Tags.templated, t.hm96Rows = true ∧ t.hiddenOutputs = false := by decide +kernel
