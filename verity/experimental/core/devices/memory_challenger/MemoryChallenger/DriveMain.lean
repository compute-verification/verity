import MemoryChallenger.MemoryWorld

/-! `memory-challenger-drive`: `MemoryChallenger.MemoryWorld.driveFn` behind the difftest's line protocol. -/

open Lean (Json)

private def line (key value : String) : String := "{" ++ Flock.canonStr key ++ ":" ++ value ++ "}"

def main : IO Unit := do
  let stdin ← IO.getStdin
  let stdout ← IO.getStdout
  repeat
    let request ← stdin.getLine
    if request.isEmpty then break
    let response ← match Json.parse request with
      | .error e => pure (line "error" (Flock.canonStr s!"request: not JSON: {e}"))
      | .ok req =>
        match req.getObjValAs? String "fn", req.getObjVal? "input" with
        | .ok "drive", .ok input =>
          match ← MemoryChallenger.MemoryWorld.driveFn input with
          | .error reason => pure (line "reject" (Flock.canonStr reason))
          | .ok out =>
            match Flock.canon out with
            | .ok text => pure (line "output" text)
            | .error e => pure (line "error" (Flock.canonStr s!"output: {e}"))
        | _, _ => pure (line "error" "\"request: not a drive request\"")
    stdout.putStrLn response
    stdout.flush
