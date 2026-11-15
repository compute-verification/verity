module

public import Lean.Data.Json
public import Flock.JsonParse
public import Spot.PlanCheck
public import Spot.TwoStageCheck

/-!
# `sampled-proofs-plancheck`: the planner's designs, checked

One JSON object a line on stdin, one answer a line on stdout (`verity.core.protocols.spot.plancheck`). Every
number is a string, a natural number or a rational `"num/den"`, so none passes through a float:

* `{"kind": "leak", "units", "width", "q": "num/den", "count", "delta"}` answers `{"bound": "num/den"}`, what
  `PlanCheck.leakBound` certifies, or `{"bound": null}` when it refuses;
* `{"kind": "work", "strata": [["work", "num"], ...], "den", "target", "delta"}` answers `{"ok": bool}`,
  `PlanCheck.checkWork`;
* `{"kind": "two", "form": "sets" | "patterns" | "capped", "q", "theta", "delta", "S", "p", "classes": [{"count",
  "cells": [{"k", "parts": [["count", "bits"], ...]}, ...], "reads": [[["count", "bits"], ...], ...], "N"}, ...]}`,
  optionally with
  `"search": {"span", "n", "p"}`, answers `{"bits": "num/den", "theta": "num/den"}`, what `TwoStageCheck.checkTwoStage`
  certifies at `theta`, or at the `θ` `TwoStageCheck.searchTheta` finds from it (`{"bits": null}` when it refuses);

and a line it can't read answers `{"error": "..."}`.
-/

@[expose] public section

namespace Spot.PlanCheckMain

open Lean (Json)
open Spot.PlanCheck

def ratStr (q : Rat) : String := s!"{q.num}/{q.den}"

def nat (s : String) : Except String Nat :=
  match s.toNat? with
  | some n => .ok n
  | none => .error s!"{s} is not a natural number"

/-- `"num/den"` or `"num"`, as its numerator and denominator as written. -/
def pair (s : String) : Except String (Int × Nat) :=
  match s.splitOn "/" with
  | [a] => match a.toInt? with
    | some n => .ok (n, 1)
    | none => .error s!"{s} is not a rational"
  | [a, b] => match a.toInt?, b.toNat? with
    | some n, some d => if d = 0 then .error s!"{s} has denominator 0" else .ok (n, d)
    | _, _ => .error s!"{s} is not a rational"
  | _ => .error s!"{s} is not a rational"

def rat (s : String) : Except String Rat := do
  let (n, d) ← pair s
  return (n : Rat) / (d : Rat)

def str (j : Json) (k : String) : Except String String := j.getObjValAs? String k

/-- `[["count", "bits"], ...]`. -/
def parts (j : Json) : Except String (List (Nat × Nat)) := do
  let rows : Array (Array String) ← Lean.fromJson? j
  rows.toList.mapM fun r => match r.toList with
    | [c, b] => return (← nat c, ← nat b)
    | _ => throw "a part is [count, bits]"

def twoClass (j : Json) : Except String TwoStageCheck.ReplayClass := do
  let cells ← (← j.getObjValAs? (Array Json) "cells").toList.mapM fun c => do
    return ({ k := ← nat (← str c "k"), parts := ← parts (← c.getObjVal? "parts") } : FactorUpper.Cell)
  let reads ← (← j.getObjValAs? (Array Json) "reads").toList.mapM parts
  return { count := ← nat (← str j "count"), cells, reads, N := ← nat (← str j "N") }

def two (j : Json) : Except String Json := do
  let form ← match ← str j "form" with
    | "sets" => pure TwoStageCheck.Form.sets
    | "patterns" => pure .patterns
    | "capped" => pure .capped
    | f => throw s!"no form {f}"
  let classes ← (← j.getObjValAs? (Array Json) "classes").toList.mapM twoClass
  let D : TwoStageCheck.Point := { form, q := ← rat (← str j "q"), θ := ← rat (← str j "theta"),
                                    δ := ← rat (← str j "delta"), S := ← nat (← str j "S"), classes,
                                    p := ← nat (← str j "p") }
  let θ ← match j.getObjVal? "search" with
    | .ok s => do
      pure (TwoStageCheck.searchTheta D D.θ (← nat (← str s "span")) (← nat (← str s "n")) (← nat (← str s "p")))
    | .error _ => pure D.θ
  return Json.mkObj [("bits", match TwoStageCheck.checkTwoStage { D with θ := θ } with
      | some v => Json.str (ratStr v)
      | none => Json.null), ("theta", Json.str (ratStr θ))]

def answer (j : Json) : Except String Json := do
  match ← str j "kind" with
  | "leak" =>
    let (num, den) ← pair (← str j "q")
    if num < 0 then throw "q is negative"
    let d : Leak := { units := ← nat (← str j "units"), width := ← nat (← str j "width"), num := num.toNat, den,
                      count := ← nat (← str j "count"), delta := ← rat (← str j "delta") }
    return Json.mkObj [("bound", match leakBound d with
      | some v => Json.str (ratStr v)
      | none => Json.null)]
  | "work" =>
    let rows ← j.getObjValAs? (Array (Array String)) "strata"
    let strata ← rows.toList.mapM fun r => match r.toList with
      | [w, n] => return ({ work := ← rat w, num := ← nat n } : Rate)
      | _ => throw "a stratum is [work, num]"
    let d : Work := { strata, den := ← nat (← str j "den"), target := ← rat (← str j "target"),
                      delta := ← rat (← str j "delta") }
    return Json.mkObj [("ok", Json.bool (checkWork d))]
  | "two" => two j
  | k => throw s!"no kind {k}"

def respond (line : String) : String :=
  match Flock.JsonParse.parse line >>= answer with
  | .ok j => j.compress
  | .error e => (Json.mkObj [("error", Json.str e)]).compress

def main : IO Unit := do
  let stdin ← IO.getStdin
  let stdout ← IO.getStdout
  repeat
    let line ← stdin.getLine
    if line.isEmpty then break
    let l := line.trimAsciiEnd.toString
    if !l.isEmpty then
      stdout.putStrLn (respond l)
      stdout.flush

end Spot.PlanCheckMain

def main : IO Unit := Spot.PlanCheckMain.main
