/- The `verity-evaluate` executable: `Evaluate.evaluateFn` and `Evaluate.libraryTableFn` behind the difftest harness's line
protocol. -/
import Evaluate.Serve

def main : IO Unit :=
  Difftest.serve (Difftest.table [("evaluate", Evaluate.evaluateFn), ("library_table", Evaluate.libraryTableFn)])
