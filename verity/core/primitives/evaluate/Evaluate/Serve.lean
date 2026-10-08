import Evaluate.Walk
import Difftest

/-!
# The difftest function

`verity-evaluate` serves `evaluate` through the difftest harness's line protocol (`Difftest.serve`,
`verity/core/difftest`): a request `{"fn": "evaluate", "input": {"program": <descriptor>, "inputs": [[word,
...], ...]}}` is answered `{"output": [word, ...]}` or `{"reject": "<reason>"}`. `program` is the descriptor as a JSON object,
or its bytes as a JSON string; `inputs` holds one argument per parameter of the program function, a list of non-negative
integer words in leaf order or `{"table": <name>}` for a registered table the evaluator computes (`Evaluate.Arg`), which keeps
a table of 2^28 words out of the request (`Evaluate.evaluate`). An `input` not of this form is rejected too.
-/

namespace Evaluate

open Lean (Json)

private def word? : Json → Option Nat
  | .num n => if n.exponent == 0 && n.mantissa ≥ 0 then some n.mantissa.toNat else none
  | _ => none

/-- The `inputs` of a request: a list of arguments, each a list of non-negative integers or `{"table": <name>}`. -/
def inputsOfJson (j : Json) : Flock.V (Array Arg) := do
  let .arr ps := j | throw "inputs: not a list of arguments"
  ps.mapM fun p => match p with
    | .arr ws => Arg.words <$> ws.mapM fun w => match word? w with
      | some v => pure v
      | none => throw s!"inputs: {w.compress} is not a non-negative integer word"
    | .obj kvs => match kvs.toArray with
      | #[("table", .str name)] => pure (Arg.table name)
      | _ => throw "inputs: an object argument is {\"table\": <name>}"
    | _ => throw "inputs: not a list of arguments"

/-- `evaluate` on a request's `input`: the root's output words, or the refusal. -/
def evaluateFn (input : Json) : Except String Json := do
  let bytes ← match input.getObjVal? "program" with
    | .ok (.str s) => pure s.toUTF8
    | .ok p@(.obj _) => pure p.compress.toUTF8
    | _ => throw "input: program is a descriptor object or its bytes as a string"
  let .ok inputs := input.getObjVal? "inputs" | throw "input: no inputs"
  let out ← evaluate bytes (← inputsOfJson inputs)
  return Json.arr (out.map fun v => Lean.toJson v)

/-- `library_table` on a request's `input`, `{"name": <table>}`: the SHA-512 (hex) of the library table the MUFU primitives
read (`Evaluate.Mufu.tableSha512`), or the refusal. -/
def libraryTableFn (input : Json) : Except String Json := do
  let .ok name := input.getObjValAs? String "name" | throw "input: no table name"
  let some digest := Mufu.tableSha512 name | throw s!"table {name}: not a table the MUFU primitives read"
  return Json.str digest

/-- `registered_table` on a request's `input`, `{"name": <table>}`: the SHA-256 (hex) of a registered table's entries
(`Evaluate.Act.tableSha256`), or the refusal. -/
def registeredTableFn (input : Json) : Except String Json := do
  let .ok name := input.getObjValAs? String "name" | throw "input: no table name"
  let some digest := Act.tableSha256 name | throw s!"table {name}: not a registered table"
  return Json.str digest

end Evaluate
