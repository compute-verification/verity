import NetworkCertifier.Calls
import Lean.Data.Json

/-!
# The calls' difftest: the certifier program's calls against `calls.Calls`, on random windows

`scripts/calls_vectors.py` generates windows and runs `verity.core.protocols.certifier.calls.Calls` on them.
`NetworkCertifier/CallsDifftestMain.lean` reads both files and fails unless `evaluate`, which computes the same results
from the program's definitions (`NetworkCertifier.Calls`), gives every one of them. The Lean audit runs it (`runs` in
`lean-audit.json`): compiled evaluation, a test and not a proof.

An instance is a window: `{"cap", "fwd_macs", "release_bits", "entropy", "events"}`, the certifier's entropy as hex,
and each event `["a", forward, call, root, rows, cols, K, bn]` (a call's A root and its shape), `["b", r, frames]` (a
boundary link's bucket at allocation `r`, its real frames' content bits), `["w", forward, root]` (a forward's words
root) or `["c"]` (the window's close), each root as hex. `evaluate` gives
* `answers`: each event's answer, `{"release": hex}` (the release's bytes), `"no coins"`, `"counted"`,
  `{"verdict": name}`, `{"record": forwards}` (the close's record, its forwards as `summary` has them) or `"refusal"`;
* `summary`: the window's forwards in id order, `{"forward", "calls", "words_root", "verdict"}` with each call
  `{"call", "rows", "cols", "K", "bn", "a_root"}`, as `Calls.summary` gives them without the releases' digests, which the
  answers hold the bytes of.
-/

namespace NetworkCertifier.Calls.CallsDifftest

open Lean

/-- A lowercase hex digit's value. -/
def hexDigit (c : Char) : Except String Nat :=
  if '0' ≤ c ∧ c ≤ '9' then pure (c.toNat - '0'.toNat)
  else if 'a' ≤ c ∧ c ≤ 'f' then pure (c.toNat - 'a'.toNat + 10)
  else throw s!"not a lowercase hex digit: {c}"

/-- The bytes of hex digits, after `acc`, which holds the bytes read so far reversed. -/
def bytesOfChars : List Char → List Nat → Except String (List Nat)
  | [], acc => pure acc.reverse
  | a :: b :: cs, acc => do bytesOfChars cs (((← hexDigit a) * 16 + (← hexDigit b)) :: acc)
  | [_], _ => throw "hex of odd length"

/-- The bytes of lowercase hex. -/
def bytesOfHex (s : String) : Except String (List Nat) := bytesOfChars s.toList []

/-- A hex digit. -/
def hexChar (d : Nat) : Char := Char.ofNat (if d < 10 then 48 + d else 87 + d)

/-- Bytes as lowercase hex. -/
def hexOf (bytes : List Nat) : String :=
  String.ofList (bytes.flatMap fun x => [hexChar (x / 16), hexChar (x % 16)])

/-- An event as read. -/
def eventOfJson (j : Json) : Except String Event := do
  match ← fromJson? (α := List Json) j with
  | [.str "a", forward, call, root, rows, cols, k, bn] =>
    pure (.aRoot (← fromJson? forward) (← fromJson? call) (← bytesOfHex (← fromJson? root))
      ⟨← fromJson? rows, ← fromJson? cols, ← fromJson? k, ← fromJson? bn⟩)
  | [.str "b", r, frames] => pure (.bucket (← fromJson? r) (← fromJson? frames))
  | [.str "w", forward, root] => pure (.wordsRoot (← fromJson? forward) (← bytesOfHex (← fromJson? root)))
  | [.str "c"] => pure .close
  | _ => throw s!"not an event: {j.compress}"

/-- An instance as read. -/
structure Raw where
  cap : Cap
  releaseBits : Nat
  entropy : List Nat
  events : List Event

def rawOfJson (j : Json) : Except String Raw := do
  pure {
    cap := ⟨← j.getObjValAs? Nat "cap", ← j.getObjValAs? Nat "fwd_macs"⟩
    releaseBits := ← j.getObjValAs? Nat "release_bits"
    entropy := ← bytesOfHex (← j.getObjValAs? String "entropy")
    events := ← (← j.getObjValAs? (List Json) "events").mapM eventOfJson }

/-- A call as `Calls.summary` gives it, without the release's digest. -/
def callJson (c : Call) : Json :=
  Json.mkObj [("call", toJson c.id), ("rows", toJson c.shape.rows), ("cols", toJson c.shape.cols),
    ("K", toJson c.shape.K), ("bn", toJson c.shape.bn), ("a_root", Json.str (hexOf c.aRoot))]

/-- A forward as `Calls.summary` gives it, without its calls' releases' digests: its id, calls, words root and verdict
(none while it is open). -/
def forwardJson (id : Nat) (calls : List Call) (wordsRoot : Option (List Nat)) (verdict : Option ForwardVerdict) :
    Json :=
  Json.mkObj [("forward", toJson id), ("calls", Json.arr (calls.map callJson).toArray),
    ("words_root", match wordsRoot with | some r => Json.str (hexOf r) | none => Json.null),
    ("verdict", match verdict with | some v => Json.str v.name | none => Json.null)]

/-- An answer as `calls_vectors.py` writes it. -/
def answerJson : CallAnswer → Json
  | .release bytes => Json.mkObj [("release", Json.str (hexOf bytes))]
  | .counted => Json.str "counted"
  | .verdict v => Json.mkObj [("verdict", Json.str v.name)]
  | .record es => Json.mkObj [("record",
      Json.arr (es.map fun e => forwardJson e.id e.calls e.wordsRoot (some e.verdict)).toArray)]
  | .refusal => Json.str "refusal"
  | .noCoins => Json.str "no coins"

/-- What the program's definitions give on an instance, in the shape of the Python results. -/
def evaluate (r : Raw) : Except String Json := do
  let (answers, st) := run r.cap r.releaseBits (State.init r.entropy) r.events
  pure <| Json.mkObj [("answers", Json.arr (answers.map answerJson).toArray),
    ("summary", Json.arr (st.forwards.map fun f => forwardJson f.id f.calls f.wordsRoot f.verdict).toArray)]

/-- The results' fields, in the order they are compared. -/
def fields : List String := ["answers", "summary"]

/-- How the Lean side's results for instance line `i` differ from Python's, one line per field that differs. -/
def differences (i : Nat) (inst python : String) : List String :=
  match Json.parse inst >>= rawOfJson >>= evaluate, Json.parse python with
  | .error e, _ => [s!"instance {i}: the Lean side rejects it: {e}"]
  | _, .error e => [s!"instance {i}: Python's results don't parse: {e}"]
  | .ok lean, .ok py => fields.filterMap fun k =>
    let a := lean.getObjValD k
    let b := py.getObjValD k
    if a == b then none else some s!"instance {i}: {k}: Lean {(a.compress.take 400)}, Python {(b.compress.take 400)}"

/-- `lake env lean --run NetworkCertifier/CallsDifftestMain.lean INSTANCES PYTHON N`: every instance line of `INSTANCES`
evaluated, each compared with the same line of `PYTHON`. Fails unless both files have `N` lines and none differs. -/
def main (args : List String) : IO UInt32 := do
  let [insts, py, n] := args | IO.eprintln "usage: NetworkCertifier/CallsDifftestMain.lean INSTANCES PYTHON N"; return 2
  let lines (p : String) : IO (Array String) := do
    return ((← IO.FS.readFile p).splitOn "\n").filter (! ·.isEmpty) |>.toArray
  let a ← lines insts
  let b ← lines py
  if n.toNat? != some a.size ∨ a.size != b.size then
    IO.eprintln s!"calls difftest: {a.size} instances and {b.size} results, not {n}"
    return 1
  let tasks := (a.zip b).toList.zipIdx.map fun ((x, y), i) => Task.spawn fun _ => differences i x y
  let found := tasks.flatMap Task.get
  for d in found.take 50 do IO.eprintln d
  IO.println s!"calls difftest: {a.size} instances compared, {found.length} differences"
  return if found.isEmpty then 0 else 1

end NetworkCertifier.Calls.CallsDifftest
