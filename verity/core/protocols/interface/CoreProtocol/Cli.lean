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
import CoreProtocol.Protocol.Canonical

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

/-- One `serve` case's inputs from its JSON: the parameters (re-encoded canonically for `answer`), commitment and
evidence as hex. The case is read in place (`Canonical`), so a case of any size gets the answer `canonicalJson` and
`Flock.unhex` give one small enough for them. -/
def caseInputs (j : Json) : V (ByteArray × ByteArray × ByteArray) := do
  let params ← Flock.canon (← Flock.J.get j "params" "the case")
  let c ← Canonical.unhex (← Flock.J.str j "commitment" "the case")
  let e ← Canonical.unhex (← Flock.J.str j "evidence" "the case")
  pure (params.toUTF8, c, e)

/-- The answer on a case's inputs, a malformed case's a rejection with no commitment. -/
def Protocol.ofCase (P : Protocol) : V (ByteArray × ByteArray × ByteArray) → Answer
  | .ok (p, c, e) => P.answer p c e
  | .error e =>
    { protocol := P.id, verdict := .reject, reason := some s!"case: {e}", digest := none, published := none, profile := none }

/-- One `serve` case: the parameters as JSON (re-encoded canonically for `answer`), commitment and evidence as hex. -/
def Protocol.case (P : Protocol) (line : String) : Answer := P.ofCase (Canonical.parse line >>= caseInputs)

/-- `line`'s first `stop` bytes without the newlines that end them. -/
def trimNewlines (line : String) : (fuel stop : Nat) → Nat
  | 0, stop => stop
  | fuel + 1, stop => if 0 < stop && Canonical.byte line (stop - 1) == 10 then trimNewlines line fuel (stop - 1) else stop

/-- `Protocol.case` on each line without its newline. The line is parsed before anything is decoded from it, so it is
dropped before the evidence's hex is: a case's text is never held beside its decoded evidence. -/
partial def serveLoop (P : Protocol) (stdin : IO.FS.Stream) : IO UInt32 := do
  let line ← stdin.getLine
  if line.isEmpty then return 0
  let j := Canonical.parseUpTo line (trimNewlines line line.utf8ByteSize line.utf8ByteSize)
  match ← emit (P.ofCase (j >>= caseInputs)) with
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

/-! `Canonical`'s reader against `canonicalJson`, which a case read to its end gets the answer of. -/

private def sameAnswer (a b : V Json) : Bool :=
  match a, b with
  | .ok x, .ok y => (Flock.canon x).toOption == (Flock.canon y).toOption
  | .error x, .error y => x == y
  | _, _ => false

private def TEXTS : List String := [
  "{}", "[]", "\"\"", "0", "-1", "-0", "01", "12345678901234567890123456789", "1.0", "1e2", "-", "true", "false", "null",
  "tru", "nul", "{\"a\":1}", "{\"a\":1,\"b\":[1,2,{\"c\":null}]}", "{\"b\":1,\"a\":2}", "{\"a\":1,\"a\":2}",
  "{\"a\": 1}", " {}", "{} ", "{}\n", "[1,]", "[,1]", "[1,2", "{\"a\":}", "{\"a\"}", "{\"a\":1,}", "{\"\":0}",
  "[true,false,null,\"x\",[],{}]", "[1.5]", "[[[[1]]]]", "\"\\u0001\"", "\"\\u000b\"", "\"\\u001f\"", "\"\\u001F\"",
  "\"\\u0008\"", "\"\\b\"", "\"\\f\"", "\"\\n\"", "\"\\r\"", "\"\\t\"", "\"\\u0009\"", "\"\\/\"", "\"\\u0041\"",
  "\"\\\"\\\\\"", "\"tab\there\"", "\"é\"", "\"\\u00e9\"", "\"😀\"", "\"\\ud83d\\ude00\"", "\"\\u0000\"",
  "\"\\u0020\"", "\"\\x\"", "\"\\", "\"abc", "{\"b\":1,\"é\":2}", "{\"é\":1,\"z\":2}", "{\"a\":{\"b\":[\"c\",1]},\"b\":0}",
  "[\"a\\nb\",\"\\\"q\\\"\"]", "\u007f", "[\"\u007f\"]", "\"\\u007f\"", ""]

#guard TEXTS.all fun t => sameAnswer (Canonical.parse t) (canonicalJson t.toUTF8)
#guard TEXTS.all fun t => sameAnswer (Canonical.parseUpTo (t ++ "\n\n") t.utf8ByteSize) (canonicalJson t.toUTF8)
#guard TEXTS.all fun t => trimNewlines (t ++ "\n\n") (t.utf8ByteSize + 2) (t.utf8ByteSize + 2) ==
  t.utf8ByteSize - (if t.endsWith "\n" then 1 else 0)

end Core
