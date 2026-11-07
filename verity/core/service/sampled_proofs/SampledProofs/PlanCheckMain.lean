module

public import Lean.Data.Json
public import Flock.JsonParse
public import SampledProofs.PlanCheck

/-!
# `sampled-proofs-plancheck`: the planner's designs, checked

One JSON object a line on stdin, one answer a line on stdout (`verity.core.protocols.sampled_proofs.plancheck`). Every
number is a string, a natural number or a rational `"num/den"`, so none passes through a float:

* `{"kind": "leak", "units", "width", "q": "num/den", "count", "delta"}` answers `{"bound": "num/den"}`, what
  `PlanCheck.leakBound` certifies, or `{"bound": null}` when it refuses;
* `{"kind": "work", "strata": [["work", "num"], ...], "den", "target", "delta"}` answers `{"ok": bool}`,
  `PlanCheck.checkWork`;

and a line it can't read answers `{"error": "..."}`.
-/

@[expose] public section

namespace SampledProofs.PlanCheckMain

open Lean (Json)
open SampledProofs.PlanCheck

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

end SampledProofs.PlanCheckMain

def main : IO Unit := SampledProofs.PlanCheckMain.main
