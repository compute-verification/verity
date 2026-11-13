module

public import NetworkCheck.Period

/-!
The `network-period-files-given` executable, the period's difftest's alone: `NetworkCheck.Period.protocol` with the
scan's circuit files' finding taken as each case gives it, in place of the one `protocol` reads from the caller's blobs.

`serve` reads a case a line, `Core.Protocol.case`'s with the finding under `files`: a string, why the scan's circuit
files are not its templates', or absent when they are. The parameters, commitment and evidence are decoded as `protocol`
decodes them, and the check is `protocol`'s own, `NetworkCheck.Period.checkGiven`, at that finding: this module defines
no check. No protocol composes it and no command but the difftest's runs it, since it accepts a period whose files
nothing has read. Audited under `NetworkCheck`.
-/

@[expose] public section

namespace NetworkCheck.PeriodFilesGiven

open Lean (Json)
open Flock (V)

/-- `NetworkCheck.Period.protocol` with its files' finding `files` as given. -/
def withFinding (files : Option String) : Core.Protocol :=
  { NetworkCheck.Period.protocol.toProtocol with
    check := fun (n : NetworkCheck.Period.Named) c e => NetworkCheck.Period.checkGiven files n.params c e }

/-- One case's finding (`files`, absent: none) and the case without it. -/
def findingOf (line : String) : V (Option String × Json) := do
  let j ← Core.Canonical.parse line
  let .obj kvs := j | throw "the case is not an object"
  let files : Option String ← match j.getObjVal? "files" with
    | .error _ => pure none
    | .ok (.str why) => pure (some why)
    | .ok _ => throw "the case's files is not a string"
  pure (files, Json.mkObj (kvs.toArray.filter (·.1 != "files")).toList)

def answerOf (line : String) : Core.Answer :=
  match findingOf line with
  | .ok (files, rest) => (withFinding files).ofCase (Core.caseInputs rest)
  | .error e => (withFinding none).ofCase (.error e)

partial def serveLoop (stdin : IO.FS.Stream) : IO UInt32 := do
  let line ← stdin.getLine
  if line.isEmpty then return 0
  let line := (line.dropEndWhile (· == '\n')).toString
  match (answerOf line).canonical with
  | .ok out =>
    let o ← IO.getStdout
    o.putStrLn out
    o.flush
    serveLoop stdin
  | .error e =>
    IO.eprintln s!"network-period-files-given: the outcome is not canonical JSON: {e}"
    return 70

end NetworkCheck.PeriodFilesGiven

def main (args : List String) : IO UInt32 := do
  match args with
  | ["serve"] => NetworkCheck.PeriodFilesGiven.serveLoop (← IO.getStdin)
  | _ =>
    IO.eprintln "usage: network-period-files-given serve"
    return 64
