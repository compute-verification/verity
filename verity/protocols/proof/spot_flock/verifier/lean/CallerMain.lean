module

public import Difftest
public import Spot.OneStage
public import VStarRecursive.Caller

/-!
# `spot-flock-caller`: SPOT's caller's check behind `Difftest.serve`

One function, `caller`: `{"params", "record"}`, the one-stage audit's parameters as `Spot.OneStage.paramsParsed` reads
them and a registration record, to `{"codes"}`, the caller's codes (`VStar.Recursive.Caller.check`), whatever SPOT's
check makes of the record. SPOT's verifier (`one_stage.verifier.check`) asks it beside its Python twin
(`one_stage.caller.check`) once SPOT's check passes, as `Caller.gate` does; SPOT's own package can't load the caller's
check, so it has its own executable.
-/

@[expose] public section

namespace VStar.Recursive.CallerMain

open Lean (Json)
open Spot

/-- `{"params", "record"}` to the caller's codes. -/
def caller (j : Json) : Flock.V Json := do
  let p ← OneStage.paramsParsed (← Core.canonicalJson (← Flock.canon (getD j "params")).toUTF8)
  return Json.mkObj [("codes", Json.arr ((Caller.check (getD j "record") p.verifier).map Json.str))]

end VStar.Recursive.CallerMain

/-- Answer requests from stdin until it ends. -/
def main : IO Unit := Difftest.serve (Difftest.table [("caller", VStar.Recursive.CallerMain.caller)])
