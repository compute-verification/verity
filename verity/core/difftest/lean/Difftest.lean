import Lean.Data.Json
import Flock.Canon

/-!
# The difftest server: a port's Lean functions behind a line protocol

A port's difftest executable is `def main : IO Unit := Difftest.serve (Difftest.table [("name", fn), …])`. It reads one
request per line on stdin, `{"fn": NAME, "input": JSON}`, until end of input, and writes one line per request on stdout:

* `{"output": JSON}` when `fn` returns `.ok`, the output in canonical JSON (`Flock.canon`: compact, keys sorted,
  integers only);
* `{"reject": REASON}` when `fn` returns `.error REASON`: the port refuses the input, as the Python reference must;
* `{"error": MESSAGE}` for a fault of the protocol rather than a verdict of the port: a line that is not a JSON object
  with a string `fn` and an `input`, an unknown `fn`, or an output that has no canonical form (a non-integer number).

Every line written is canonical JSON. The Python half (`verity_difftest`) keeps one process open for a whole run, so a
port's functions must be pure: nothing else may be written to stdout (use stderr).
-/

namespace Difftest

open Lean (Json)

/-- A ported function: its output, or why it rejects the input. -/
abbrev Fn := Json → Except String Json

/-- The dispatch of a list of named functions; the first of a repeated name wins. -/
def table (fns : List (String × Fn)) : String → Option Fn := fun name => fns.lookup name

private def line (key : String) (value : String) : String := "{" ++ Flock.canonStr key ++ ":" ++ value ++ "}"

/-- The response line to one request line (`Json.parse` takes its trailing newline as whitespace). -/
def respond (dispatch : String → Option Fn) (request : String) : String :=
  let fault := fun (msg : String) => line "error" (Flock.canonStr msg)
  match Json.parse request with
  | .error e => fault s!"request: not JSON: {e}"
  | .ok req =>
    match req.getObjValAs? String "fn", req.getObjVal? "input" with
    | .error _, _ => fault "request: no string fn"
    | _, .error _ => fault "request: no input"
    | .ok name, .ok input =>
      match dispatch name with
      | none => fault s!"unknown fn: {name}"
      | some f =>
        match f input with
        | .error reason => line "reject" (Flock.canonStr reason)
        | .ok out =>
          match Flock.canon out with
          | .ok text => line "output" text
          | .error e => fault s!"output of {name}: {e}"

/-- Answer requests from stdin until it ends, flushing each response. -/
def serve (dispatch : String → Option Fn) : IO Unit := do
  let stdin ← IO.getStdin
  let stdout ← IO.getStdout
  repeat
    let request ← stdin.getLine
    if request.isEmpty then break
    stdout.putStrLn (respond dispatch request)
    stdout.flush

end Difftest
