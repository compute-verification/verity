import Lean.Data.Json
import Flock.RecLinks

/-!
`flock-rec-links`'s command: `Flock.RecLinks.check` on each case of a vectors file (`verity/rec-links-vectors/v1`, written
by `verity_flock.rec_links.vectors`), which `test_rec_links` holds to the Python mirror.

    flock-rec-links VECTORS     each case's verdict, one line: `NAME ok L` or `NAME refused CODE`

A case holds the shape (`units`, `ins`, `outs`), every unit's input and output rows' words (`in_rows`, `out_rows`, by unit
then row) and the table's rows in hex (`rows`). The code is the refusal's text before its first colon.
-/

namespace Flock.RecLinksCmd

open Lean (Json)

def VECTORS_FORMAT : String := "verity/rec-links-vectors/v1"

def nibble (c : Char) : Option UInt8 :=
  if '0' ≤ c && c ≤ '9' then some (c.toNat - 48).toUInt8
  else if 'a' ≤ c && c ≤ 'f' then some (c.toNat - 87).toUInt8
  else none

def unhex (s : String) : Except String ByteArray := do
  let cs := s.toList
  if cs.length % 2 != 0 then throw "a hex row of odd length"
  let rec go : List Char → ByteArray → Except String ByteArray
    | a :: b :: rest, acc => match nibble a, nibble b with
      | some x, some y => go rest (acc.push (16 * x + y))
      | _, _ => throw "a row that is not lowercase hex"
    | _, acc => pure acc
  go cs ByteArray.empty

def verdict (c : Json) : Except String String := do
  let name ← c.getObjValAs? String "name"
  let sh : RecLinks.Shape :=
    { units := ← c.getObjValAs? Nat "units", ins := ← c.getObjValAs? (Array Nat) "ins",
      outs := ← c.getObjValAs? (Array Nat) "outs" }
  let ins ← c.getObjValAs? (Array (Array (Array Nat))) "in_rows"
  let outs ← c.getObjValAs? (Array (Array (Array Nat))) "out_rows"
  let rows ← (← c.getObjValAs? (Array String) "rows").mapM unhex
  let at_ := fun (xs : Array (Array (Array Nat))) (u p : Nat) => (xs[u]?.bind (·[p]?)).getD #[]
  return match RecLinks.check sh (at_ ins) (at_ outs) rows with
    | .ok n => s!"{name} ok {n}"
    | .error e => s!"{name} refused {(e.splitOn ":").head!}"

def cmd (args : List String) : IO UInt32 := do
  let [path] := args | IO.eprintln "usage: flock-rec-links VECTORS"; return 2
  let text ← IO.FS.readFile path
  let out : Except String (Array String) := do
    let j ← Json.parse text
    if (j.getObjValAs? String "format").toOption != some VECTORS_FORMAT then throw s!"not a {VECTORS_FORMAT} file"
    (← j.getObjValAs? (Array Json) "cases").mapM verdict
  match out with
  | .ok lines => lines.forM IO.println; return 0
  | .error e => IO.eprintln s!"flock-rec-links: {e}"; return 1

end Flock.RecLinksCmd
