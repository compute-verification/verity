/- The `flock-read-tmpl` executable: `flock-read-tmpl V H` prints the table read's template
(`Flock.ReadNet.readTmpl V H`) as `flock-ir-tmpl/v2` text, after reading the text back and checking it with
`Flock.ReadNet.check`. -/
module

public import Flock.ReadNet.Template

@[expose] public section

def main (args : List String) : IO UInt32 := do
  match args.map String.toNat? with
  | [some V, some H] =>
    let text := Flock.ReadNet.render (Flock.ReadNet.readTmpl V H)
    match Flock.Tmpl.parse ((text.splitOn "\n").filter (· ≠ "")).toArray with
    | .error e =>
      IO.eprintln s!"flock-read-tmpl: the template doesn't parse: {e}"
      return 1
    | .ok t =>
      if !Flock.ReadNet.check V H t then
        IO.eprintln s!"flock-read-tmpl: the check refuses V = {V}, H = {H}"
        return 1
      IO.print text
      return 0
  | _ =>
    IO.eprintln "usage: flock-read-tmpl V H"
    return 2
