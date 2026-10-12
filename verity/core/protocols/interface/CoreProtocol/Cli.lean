/-
The command line every protocol's executable has, so that one harness drives them all (circuits' differential tests,
a composition, a service):

  <exe> id                                   prints the protocol id
  <exe> check PARAMS COMMITMENT EVIDENCE     reads three files and prints the outcome's canonical JSON line
  <exe> serve                                reads cases from stdin, one canonical JSON object per line,
                                             `{"commitment": hex, "evidence": hex, "params": <params JSON>}`, and
                                             writes one outcome line per case, flushed, until end of input

`check` exits 0 on accept, 1 on reject and 2 on undecided. Every command exits 64 on a usage error and 70 when an
outcome can't be written canonically. In `serve` a malformed case line is answered as a rejection, with no
commitment.
-/
import CoreProtocol.Protocol

namespace Core

open Lean (Json)
open Flock (V)

private def exitOf : Verdict → UInt32
  | .accept => 0
  | .reject => 1
  | .undecided => 2

private def emit (a : Answer) : IO (Option Verdict) := do
  match a.canonical with
  | .ok line =>
    let out ← IO.getStdout
    out.putStrLn line
    out.flush
    pure (some a.verdict)
  | .error e =>
    IO.eprintln s!"{a.protocol}: the outcome is not canonical JSON: {e}"
    pure none

/-- One `serve` case: the parameters as JSON (re-encoded canonically for `answer`), commitment and evidence as hex. -/
def Protocol.case (P : Protocol) (line : String) : Answer :=
  let malformed (e : String) : Answer :=
    { protocol := P.id, verdict := .reject, reason := some s!"case: {e}", digest := none, published := none, profile := none }
  let parsed : V (ByteArray × ByteArray × ByteArray) := do
    let j ← canonicalJson line.toUTF8
    let params ← Flock.canon (← Flock.J.get j "params" "the case")
    let c ← Flock.unhex (← Flock.J.str j "commitment" "the case")
    let e ← Flock.unhex (← Flock.J.str j "evidence" "the case")
    pure (params.toUTF8, c, e)
  match parsed with
  | .ok (p, c, e) => P.answer p c e
  | .error e => malformed e

partial def serveLoop (P : Protocol) (stdin : IO.FS.Stream) : IO UInt32 := do
  let line ← stdin.getLine
  if line.isEmpty then return 0
  let line := (line.dropEndWhile (· == '\n')).toString
  match ← emit (P.case line) with
  | some _ => serveLoop P stdin
  | none => return 70

def Protocol.main (P : Protocol) (args : List String) : IO UInt32 := do
  match args with
  | ["id"] => IO.println P.id; return 0
  | ["check", params, commitment, evidence] =>
    let a := P.answer (← IO.FS.readBinFile params) (← IO.FS.readBinFile commitment) (← IO.FS.readBinFile evidence)
    match ← emit a with
    | some v => return exitOf v
    | none => return 70
  | ["serve"] => serveLoop P (← IO.getStdin)
  | _ =>
    IO.eprintln s!"usage: {P.id} id | check PARAMS COMMITMENT EVIDENCE | serve"
    return 64

end Core
