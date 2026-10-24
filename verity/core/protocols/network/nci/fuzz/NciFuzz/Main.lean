import NciFuzz.Check

/-!
`lake exe nci-desc-fuzz IN OUT`: `IN` has one JSON object per line, `{"id": …, "program": TEXT}` (any other fields are
copied through; `"force": true` checks `ReadsExact` on it even where `Desc.walkable` refuses it), and `OUT` gets
`NciCheck.Fuzz.check`'s result on each, with its `id`, one line each, in order, and the program's text where a
statement fails.
-/

open Lean (Json)

def failed (r : Json) : Bool :=
  ["s1", "s2", "s3", "agree"].any fun s => !((r.getObjValD s).getObjValD "fail").isNull

def checkLine (line : String) : String :=
  match Json.parse line with
  | .error e => (Json.mkObj [("pre", Json.str "line"), ("why", Json.str e)]).compress
  | .ok j =>
    let text := match j.getObjValD "program" with
      | .str s => s
      | _ => ""
    let r := NciCheck.Fuzz.check {} text (force := j.getObjValD "force" == Json.bool true)
    let tag := match j with
      | .obj kvs => kvs.toList.filter fun kv => kv.1 != "program" || failed r
      | _ => []
    (Json.mkObj (tag ++ match r with
      | .obj kvs => kvs.toList
      | _ => [])).compress

def main (args : List String) : IO UInt32 := do
  let [inp, out] := args | IO.eprintln "usage: nci-desc-fuzz IN OUT"; return 2
  let lines := ((← IO.FS.readFile inp).splitOn "\n").filter (!·.isEmpty) |>.toArray
  let h ← IO.FS.Handle.mk out .write
  let block := 256
  let mut i := 0
  while i < lines.size do
    let tasks := (lines.extract i (i + block)).map fun l => Task.spawn fun _ => checkLine l
    for t in tasks do h.putStrLn t.get
    h.flush
    i := i + block
  IO.println s!"nci-desc-fuzz: {lines.size} programs checked"
  return 0
