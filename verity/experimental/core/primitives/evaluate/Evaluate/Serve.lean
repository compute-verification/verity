import Evaluate.Walk

/-!
# The line protocol

`verity-evaluate` reads one JSON request per line on stdin and writes one JSON answer per line on stdout:

    {"fn": "evaluate", "input": {"program": <descriptor>, "inputs": [[word, ...], ...]}}
      -> {"output": [word, ...]}  or  {"reject": "<reason>"}

`program` is the descriptor as a JSON object, or its bytes as a JSON string; `inputs` holds one list of non-negative integer
words per parameter of the program function, each in leaf order (`Evaluate.evaluate`). A request that is not of this form
is answered with a `reject` too, so every line gets exactly one answer.
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

/-- One request, answered. -/
def answer (j : Json) : Flock.V (Array Nat) := do
  match j.getObjValAs? String "fn" with
  | .ok "evaluate" => pure ()
  | .ok f => throw s!"unknown fn '{f}' (this executable answers evaluate)"
  | .error _ => throw "request: no fn"
  let .ok input := j.getObjVal? "input" | throw "request: no input"
  let bytes ← match input.getObjVal? "program" with
    | .ok (.str s) => pure s.toUTF8
    | .ok p@(.obj _) => pure p.compress.toUTF8
    | _ => throw "input: program is a descriptor object or its bytes as a string"
  let .ok inputs := input.getObjVal? "inputs" | throw "input: no inputs"
  evaluate bytes (← inputsOfJson inputs)

/-- The answer line to a request line. -/
def handle (line : String) : String :=
  let res := match Json.parse line with
    | .ok j => answer j
    | .error e => throw s!"request: not JSON: {e}"
  match res with
  | .ok out => (Json.mkObj [("output", Json.arr (out.map fun v => Lean.toJson v))]).compress
  | .error e => (Json.mkObj [("reject", Json.str e)]).compress

/-- Answer every non-empty line of stdin, in order, until it ends. -/
def serve : IO Unit := do
  let stdin ← IO.getStdin
  let stdout ← IO.getStdout
  repeat
    let line ← stdin.getLine
    if line.isEmpty then break
    let line := line.trimAsciiEnd.toString
    if line.isEmpty then continue
    stdout.putStrLn (handle line)
    stdout.flush

end Evaluate
