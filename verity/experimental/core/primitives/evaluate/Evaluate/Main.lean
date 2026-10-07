/- The `verity-evaluate` executable: `Evaluate.evaluateFn` behind the difftest harness's line protocol. -/
import Evaluate.Serve

def main : IO Unit := Difftest.serve (Difftest.table [("evaluate", Evaluate.evaluateFn)])
