import NciFuzz.Check

/-!
`lake exe nci-desc-fuzz IN OUT`: `IN` has one JSON object per line, `{"id": …, "program": TEXT}` (any other fields are
copied through; `"force": true` checks `ReadsExact` on it even where `Desc.walkable` refuses it, and `"instances": […]`,
each `{"theta", "owner", "gamma", "alpha"}`, are checked by `agree` beside its own four), and `OUT` gets
`NciCheck.Fuzz.check`'s result on each, with its `id`, one line each, in order, and the program's text and instances
where a statement fails.
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
    let given := (j.getObjValD "instances").getArr?.toOption.getD #[]
    let r := NciCheck.Fuzz.check {} text (force := j.getObjValD "force" == Json.bool true) given
    let tag := match j with
      | .obj kvs => kvs.toList.filter fun kv => (kv.1 != "program" && kv.1 != "instances") || failed r
      | _ => []
    (Json.mkObj (tag ++ match r with
      | .obj kvs => kvs.toList
      | _ => [])).compress

def main (args : List String) : IO UInt32 := do
  let [inp, out] := args | IO.eprintln "usage: nci-desc-fuzz IN OUT"; return 2
  let hin ← IO.FS.Handle.mk inp .read
  let h ← IO.FS.Handle.mk out .write
  let block := 256
  let mut n := 0
  let mut eof := false
  while !eof do
    let mut lines : Array String := #[]
    while !eof && lines.size < block do
      let l ← hin.getLine
      if l.isEmpty then
        eof := true
      else if l != "\n" then
        lines := lines.push (if l.endsWith "\n" then (l.dropEnd 1).toString else l)
    let tasks := lines.map fun l => Task.spawn fun _ => checkLine l
    for t in tasks do h.putStrLn t.get
    h.flush
    n := n + lines.size
  IO.println s!"nci-desc-fuzz: {n} programs checked"
  return 0
