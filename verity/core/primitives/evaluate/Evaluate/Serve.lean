import Evaluate.Walk
import Flock.Gate
import Flock.Evaluated
import Difftest

/-!
# The difftest function

`verity-evaluate` serves `evaluate` through the difftest harness's line protocol (`Difftest.serve`,
`verity/core/difftest`): a request `{"fn": "evaluate", "input": {"program": <descriptor>, "inputs": [[word,
...], ...]}}` is answered `{"output": [word, ...]}` or `{"reject": "<reason>"}`. `program` is the descriptor as a JSON object,
or its bytes as a JSON string; `inputs` holds one argument per parameter of the program function, a list of non-negative
integer words in leaf order or `{"table": <name>}` for a registered table the evaluator computes (`Evaluate.Arg`), which keeps
a table of 2^28 words out of the request (`Evaluate.evaluate`). An `input` not of this form is rejected too.

It also serves two readings of the decoded Program, each the Lean declaration a Python class of `verity.ml.circuits` mirrors:
`gate` (`Flock.Program.Gate`, `layout.Gate`) and `partition` (`Flock.Extract.Partition`, `partition_object.Partition`).
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

private def programBytes (input : Json) : Except String ByteArray :=
  match input.getObjVal? "program" with
    | .ok (.str s) => pure s.toUTF8
    | .ok p@(.obj _) => pure p.compress.toUTF8
    | _ => throw "input: program is a descriptor object or its bytes as a string"

/-- `evaluate` on a request's `input`: the root's output words, or the refusal. -/
def evaluateFn (input : Json) : Except String Json := do
  let bytes ← programBytes input
  let .ok inputs := input.getObjVal? "inputs" | throw "input: no inputs"
  let out ← evaluate bytes (← inputsOfJson inputs)
  return Json.arr (out.map fun v => Lean.toJson v)

open Flock.Program in
/-- An activation from the innermost up to the root, each `[definition key, offset]`, a child's followed by its node and
member in its parent's body. -/
def scopeJson (P : Prog) : Scope → List Json
  | .top d off _ => [Json.arr #[Json.str (P.defs[d]!).key, Lean.toJson off]]
  | .child parent d off node member =>
    Json.arr #[Json.str (P.defs[d]!).key, Lean.toJson off, Lean.toJson node, Lean.toJson member] :: scopeJson P parent

open Flock.Program in
/-- `gate` on a request's `input`, `{"program": <descriptor>, "gate": g}`: gate `g` of the Program's canonical layout
(`Flock.Program.Gate`, `Prog.gateAt`) as `{"index", "prim", "operands", "scope"}`, its primitive by definitions key and its
activation by `scopeJson`, or the refusal. -/
def gateFn (input : Json) : Except String Json := do
  let P ← decode (← programBytes input)
  let .ok g := input.getObjValAs? Nat "gate" | throw "input: gate is a gate index"
  let gt : Gate ← P.gateAt g
  return Json.mkObj [("index", Lean.toJson gt.index), ("prim", Json.str (P.defs[gt.prim]!).key),
    ("operands", Lean.toJson gt.operands), ("scope", Json.arr (scopeJson P gt.scope).toArray)]

open Flock.Program in
/-- `partition` on a request's `input`, `{"program": <descriptor>, "X": x}` with `x` in 1..2^16: the Program's `Q_word{X}`
evaluation (`Flock.Extract.Partition`, `Flock.Extract.evaluate`) as `{"population", "calls"}`, each Call
`[root-body index, definition key, units, owner]`, or the refusal. -/
def partitionFn (input : Json) : Except String Json := do
  let P ← decode (← programBytes input)
  let .ok x := input.getObjValAs? Nat "X" | throw "input: X is an integer in 1..65536"
  if x == 0 || x > 65536 then throw s!"input: X is an integer in 1..65536, not {x}"
  let part : Flock.Extract.Partition ← Flock.Extract.evaluate P x
  let calls := part.map fun (k, d, c) =>
    Json.arr #[Lean.toJson k, Json.str (P.defs[d]!).key, Lean.toJson c.units, Lean.toJson c.owner]
  return Json.mkObj [("population", Lean.toJson (part.foldl (fun n (_, _, c) => n + c.units) 0)), ("calls", Json.arr calls)]

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
