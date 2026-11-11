module

public import Flock.HmRow
public import Flock.ReadNet.Template
public import Flock.Zk

/-!
`flock-read-tmpl`'s command: the table read's template (`Flock.ReadNet.readTmpl`), and the check that a circuit file holds
it, which is what the read's binding (`readUnitDrawn_check`) takes of the file it reads.

    flock-read-tmpl V H                 `readTmpl V H` as `flock-ir-tmpl/v2` text, once `Tmpl.parse` reads it back and
                                        `ReadNet.check V H` accepts it
    flock-read-tmpl check V H CIRCUIT   `ok NAME` (exit 0) when CIRCUIT parses as `flock-verify verify --zk` parses a
                                        `verity/flock-circuit/tmpl` statement (`HmOut.parseTmpl`), its unit's template
                                        passes `ReadNet.check V H` and its outputs are its unit's; else the first of those
                                        that fails (exit 1)

Exit 2 is a usage error or an unreadable file.
-/

@[expose] public section

namespace Flock.ReadTmplCmd

def usage : String := "usage: flock-read-tmpl V H | flock-read-tmpl check V H CIRCUIT"

/-- `readTmpl V H` as text, once it reads back and `ReadNet.check V H` accepts it. -/
def template (V H : Nat) : Except String String := do
  let text := ReadNet.render (ReadNet.readTmpl V H)
  let t ← (Tmpl.parse ((text.splitOn "\n").filter (· ≠ "")).toArray).mapError (s!"the template doesn't parse: {·}")
  unless ReadNet.check V H t do throw s!"the check refuses V = {V}, H = {H}"
  return text

/-- **The read's premises on a circuit file**: `HmOut.parseTmpl` accepts it under `verify --zk`'s tags for a templated
statement, it carries its unit's template, `ReadNet.check V H` accepts that template, and its outputs are its unit's
(`outNet = unitNet`). The template's name, or the first that fails. -/
def premises (V H : Nat) (bytes : ByteArray) : Except String String := do
  -- a session's table count reaches only the identity's hooks, which parsing doesn't read
  let tags := Zk.tags Tags.circuitTmpl (Zk.maskLogOf bytes) 1
  let c ← (HmOut.parseTmpl tags bytes).mapError (s!"it doesn't parse as {Tags.circuitTmpl.statement}: {·}")
  let some tu := c.tmpl | throw "it carries no template for its unit"
  unless ReadNet.check V H tu.t do throw s!"the check refuses its template {tu.t.name} at V = {V}, H = {H}"
  unless c.outNet == c.unitNet do throw s!"its outputs are net {c.outNet}'s, not its unit's ({c.unitNet})"
  return tu.t.name

def cmd (args : List String) : IO UInt32 := do
  match args with
  | ["check", v, h, path] =>
    let (some V, some H) := (v.toNat?, h.toNat?) | IO.eprintln usage; return 2
    let bytes ← match ← (IO.FS.readBinFile path).toBaseIO with
      | .ok b => pure b
      | .error e => IO.eprintln s!"flock-read-tmpl: {path}: {e}"; return 2
    match premises V H bytes with
    | .ok name => IO.println s!"ok {name}"; return 0
    | .error e => IO.eprintln s!"flock-read-tmpl: {path}: {e}"; return 1
  | [v, h] =>
    let (some V, some H) := (v.toNat?, h.toNat?) | IO.eprintln usage; return 2
    match template V H with
    | .ok text => IO.print text; return 0
    | .error e => IO.eprintln s!"flock-read-tmpl: {e}"; return 1
  | _ => IO.eprintln usage; return 2

end Flock.ReadTmplCmd
