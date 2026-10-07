import Evaluate.Walk
import Difftest

/-!
# The difftest function

`verity-evaluate` serves `evaluate` through the difftest harness's line protocol (`Difftest.serve`,
`verity/experimental/core/difftest`): a request `{"fn": "evaluate", "input": {"program": <descriptor>, "inputs": [[word,
...], ...]}}` is answered `{"output": [word, ...]}` or `{"reject": "<reason>"}`. `program` is the descriptor as a JSON object,
or its bytes as a JSON string; `inputs` holds one list of non-negative integer words per parameter of the program function,
each in leaf order (`Evaluate.evaluate`). An `input` not of this form is rejected too.
-/

namespace Evaluate

open Lean (Json)

private def word? : Json → Option Nat
  | .num n => if n.exponent == 0 && n.mantissa ≥ 0 then some n.mantissa.toNat else none
  | _ => none

/-- The `inputs` of a request: a list of lists of non-negative integers. -/
def inputsOfJson (j : Json) : Flock.V (Array (Array Nat)) := do
  let .arr ps := j | throw "inputs: not a list of lists of words"
  ps.mapM fun p => match p with
    | .arr ws => ws.mapM fun w => match word? w with
      | some v => pure v
      | none => throw s!"inputs: {w.compress} is not a non-negative integer word"
    | _ => throw "inputs: not a list of lists of words"

/-- `evaluate` on a request's `input`: the root's output words, or the refusal. -/
def evaluateFn (input : Json) : Except String Json := do
  let bytes ← match input.getObjVal? "program" with
    | .ok (.str s) => pure s.toUTF8
    | .ok p@(.obj _) => pure p.compress.toUTF8
    | _ => throw "input: program is a descriptor object or its bytes as a string"
  let .ok inputs := input.getObjVal? "inputs" | throw "input: no inputs"
  let out ← evaluate bytes (← inputsOfJson inputs)
  return Json.arr (out.map fun v => Lean.toJson v)

end Evaluate
