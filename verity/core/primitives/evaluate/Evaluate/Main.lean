/- The `verity-evaluate` executable: `Evaluate.evaluateFn`, `Evaluate.libraryTableFn`, `Evaluate.registeredTableFn`,
`Evaluate.gateFn` and `Evaluate.partitionFn` behind the difftest harness's line protocol. -/
module

public import Evaluate.Serve

@[expose] public section

def main : IO Unit :=
  Difftest.serve (Difftest.table [("evaluate", Evaluate.evaluateFn), ("library_table", Evaluate.libraryTableFn),
    ("registered_table", Evaluate.registeredTableFn), ("gate", Evaluate.gateFn), ("partition", Evaluate.partitionFn)])
