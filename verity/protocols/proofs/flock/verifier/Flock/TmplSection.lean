/-
A templated statement's template section (`Tags.isTmpl`): right after META, a line `TEMPLATE n`, then the template's `n`
`flock-ir-tmpl/v2` lines, then the `CIRCUIT` sections. `HmRow.tmplLines` is what `HmRow.parse` passes over before the
sections, and `HmOut.tmplOf` reads the template off the file's head alone.
-/
module

public import Flock.Bytes
public import Flock.Net
public import Flock.SplitNL
public import Flock.Tags

@[expose] public section

namespace Flock

/-- The lines of a templated statement's (`Tags.isTmpl`) template section, right after META, which `HmRow.parse` passes
over: its header `TEMPLATE n` and the template's `n` lines (`HmOut.tmplOf` reads them). Any other statement's file has
none, its sections starting right after META. -/
def HmRow.tmplLines (tags : Tags) (lines : Array String) : Nat :=
  let h := tokens (lines.getD 2 "")
  if tags.isTmpl && h.size == 2 && h[0]! == "TEMPLATE" then (h[1]!.toNat?.map (· + 1)).getD 0 else 0

/-- A templated file's template: the lines of its template section, right after META (the file's third line `TEMPLATE n`,
then the template's `n` lines, which `HmRow.parse` passes over: `HmRow.tmplLines`), read off the file's bytes up to the
section's end alone, as `HmOut.metaOf` reads META. -/
def HmOut.tmplOf (bytes : ByteArray) : V (Array String) := do
  let bad := "circuit: no template section (TEMPLATE n, then the template's n lines) after META"
  let some n1 := findByte bytes 10 | throw bad
  let some n2 := findByte bytes 10 (n1 + 1) | throw bad
  let some n3 := findByte bytes 10 (n2 + 1) | throw bad
  let some head := String.fromUTF8? (bytes.extract (n2 + 1) n3) | throw "circuit: not UTF-8"
  let h := tokens head
  let some k := (if h.size == 2 && h[0]! == "TEMPLATE" then h[1]!.toNat? else none) | throw bad
  if k == 0 then return #[]
  let mut e := n3
  for _ in [0:k] do
    let some e' := findByte bytes 10 (e + 1) | throw "circuit: the template section is truncated"
    e := e'
  let some text := String.fromUTF8? (bytes.extract (n3 + 1) e) | throw "circuit: not UTF-8"
  return (text.splitOn "\n").toArray

end Flock
