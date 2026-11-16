import Lean

/-!
# The import graph of some modules, from their `.olean` headers

`lean --run Graph.lean OUT M₁ M₂ …` (an argument `@F` stands for F's lines) writes to `OUT`
`{"search": [dir, …], "graph": {M: {"olean": path, "imports": [I, …]}}}`: the search path, and for the `Mᵢ` and every
module they import, transitively, its `.olean` as the search path finds it and the imports its header names, every one
of them, as `Facts.lean`'s `graph` gives them. It reads each header and nothing else: no module is imported, so it takes seconds where `Facts.lean` loads the
environment. `check.py`'s pipelined replay keys a batch with it (`replay_keys`); a graph that differs from `Facts.lean`'s
gives other keys, which the final replay never finds, so it can cost reuse and never soundness.
-/

open Lean

unsafe def main (args : List String) : IO UInt32 := do
  initSearchPath (← findSysroot)
  let mut names : Array String := #[]
  for a in args do
    if a.startsWith "@" then
      for l in ← IO.FS.lines (a.drop 1).toString do
        unless l.isEmpty do names := names.push l
    else
      names := names.push a
  let some out := names[0]? | throw <| IO.userError "usage: Graph.lean OUT MODULE..."
  let mut todo := (names.extract 1 names.size).toList.map String.toName
  let mut seen : NameSet := {}
  let mut graph : Array (String × Json) := #[]
  while true do
    let m :: next := todo | break
    todo := next
    if seen.contains m then continue
    seen := seen.insert m
    let olean ← findOLean m
    unless (← olean.pathExists) do
      -- name the module and the cause so the seal step's log says why, not an opaque read failure
      IO.eprintln s!"Graph.lean: no `.olean` for {m} (looked at {olean}); build it, or list it under `exempt`"
      (← IO.getStderr).flush
      IO.Process.exit 1
    -- the region stays mapped, as `Replay.lean` keeps its own: the names below live in it
    let (header, _) ← CompactedRegion.read (α := ModuleData) olean #[]
    let imports := header.imports.map (·.module)
    graph := graph.push (m.toString, Json.mkObj [("olean", toJson olean.toString),
      ("imports", toJson (imports.map (·.toString)))])
    todo := imports.toList ++ todo
  let search := (← searchPathRef.get).map (·.toString)
  IO.FS.writeFile out (Json.mkObj [("search", toJson search), ("graph", Json.mkObj graph.toList)]).compress
  return 0
