import InferenceOnlyVerifier.Record
open InferenceOnlyVerifier InferenceOnlyVerifier.Record
open Flock.Program

/-- scratch (lean/WIRED): each stage of `Record.read` timed on one case, the edges collected into an Array (the same
`partWires` per part), the wires counted by root node, `pairwiseDisjoint` timed on prefixes, and the stages after
`Spot.Wires.check` timed with its disjointness test bypassed. Never pushed to a shared name. -/
def edgesArr (P : Prog) (rs : Array Flock.TemplateQuery.TRange) (W : Spot.Wires.Wiring) :
    Except String (Array (Nat × Nat × Nat × List Spot.Wires.Wire)) := do
  let fu := Spot.Edges.firstUnits rs
  let nodes := (P.defs[P.root]!).body.nodes
  let mut out := #[]
  for k in [0:nodes.size] do
    if Spot.Edges.isInputNode P k then continue
    let node := nodes[k]!
    for i in [0:node.args.size] do
      for (q, s0, qi) in Spot.Edges.merged node.args[i]!.seq do
        match Spot.Edges.partWires P fu W k i q s0 with
        | .ok ws => out := out.push (k, i, qi, ws)
        | .error e => throw s!"{e} at node {k} argument {i} part {qi}"
  return out

def main (args : List String) : IO UInt32 := do
  let [path] := args | IO.eprintln "usage: wired-time CASE"; return 2
  let t0 ← IO.monoMsNow
  let say (s : String) : IO Unit := do
    IO.println s!"[{((← IO.monoMsNow) - t0) / 1000} s] {s}"; (← IO.getStdout).flush
  let .ok c := Lean.Json.parse (← IO.FS.readFile path) | IO.eprintln "not JSON"; return 2
  let j := c.getObjValD "instance"; let tr := c.getObjValD "trusted"
  say "read the case"
  let f ← match front j tr with
    | .ok f => pure f
    | .error e => say s!"front: REFUSED {e}"; return 1
  say s!"front (parse, decode, program checks, record, trusted, templates, own-rows): ok, {f.ts.length} templates, {(f.P.defs[f.P.root]!).body.nodes.size} root nodes"
  let rs ← match Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg f.P) (f.ts.map (·.tid)).toArray with
    | .ok rs => pure rs
    | .error _ => say "ranges: REFUSED"; return 1
  let cuts ← match Spot.RecordText.cutsOf f.doc with
    | .ok c => pure c
    | .error e => say s!"cuts: REFUSED {e}"; return 1
  let W0 : Spot.Wires.Wiring :=
    { templates := f.ts.map fun t => (t.tid, t.wired), cuts := cuts, wires := [], order := some (Spot.RecordText.orderOf rs) }
  say s!"ranges and cuts: {rs.size} ranges"
  let parts ← match edgesArr f.P rs W0 with
    | .ok a => pure a
    | .error e => say s!"edges: REFUSED {e}"; return 1
  let total := parts.foldl (fun n (_, _, _, ws) => n + ws.length) 0
  say s!"edges (Array, same partWires): {total} wires from {parts.size} parts"
  let perPart := parts.foldl (fun (m : Std.HashMap Nat Nat) (_, _, _, ws) => m.insert ws.length ((m.getD ws.length 0) + 1)) {}
  say s!"parts by wires each: {(perPart.toList.mergeSort (fun a b => a.1 ≤ b.1)).take 12}"
  let byNode := parts.foldl (fun (m : Std.HashMap Nat Nat) (k, _, _, ws) => m.insert k ((m.getD k 0) + ws.length)) {}
  let top := (byNode.toList.mergeSort (fun a b => a.2 ≥ b.2)).take 8
  say s!"root nodes with wires: {byNode.size}; the most: {top.map fun (k, n) => s!"node {k} ({(f.P.defs[(f.P.defs[f.P.root]!).body.nodes[k]!.fn]!).id}, {(f.P.defs[f.P.root]!).body.nodes[k]!.members} members): {n}"}"
  let biggest := (parts.toList.mergeSort (fun a b => a.2.2.2.length ≥ b.2.2.2.length)).take 3
  for (k, i, qi, ws) in biggest do
    say s!"part node {k} argument {i} part {qi}: {ws.length} wires; first two: {repr (ws.take 2)}"
  let wires := parts.toList.flatMap (·.2.2.2)
  let W := { W0 with wires := wires }
  say s!"wires flattened: {wires.length}"
  say s!"check.cutsOk: {Spot.Wires.cutsOk W}"
  say s!"check.firstWire: {repr (Spot.Wires.firstWire W (Spot.Wires.wireCode W))}"
  let timed (what : String) (f : Unit → Bool) : IO Bool := do
    let a ← IO.monoMsNow
    let r ← IO.lazyPure f
    let b ← IO.monoMsNow
    say s!"{what}: {r}, {b - a} ms"; return r
  for n in [2000, 4000, 8000, 16000, 32000] do
    let ws := wires.take n
    let o ← timed s!"pairwiseDisjoint (every pair) on the first {n} wires" fun _ => Spot.Wires.pairwiseDisjoint ws
    let q ← timed s!"pairwiseDisjointSorted on the first {n} wires" fun _ => Spot.Wires.pairwiseDisjointSorted ws
    say s!"  same verdict: {o == q}"
  let _ ← timed s!"pairwiseDisjointSorted on all {wires.length} wires" fun _ => Spot.Wires.pairwiseDisjointSorted wires
  let a0 ← IO.monoMsNow
  let shipped ← IO.lazyPure fun _ => Spot.Edges.edgesOf f.P rs W0
  say s!"edgesOf (the trial's Array accumulator): {(← IO.monoMsNow) - a0} ms, same wires as the probe's: {match shipped with | .ok ws => ws == wires | .error _ => false}"
  let a1 ← IO.monoMsNow
  let chk ← IO.lazyPure fun _ => Spot.Wires.check W
  say s!"Spot.Wires.check (the trial's): {(← IO.monoMsNow) - a1} ms, {match chk with | .ok _ => "ok" | .error r => s!"{r.code.name} at {r.wire}"}"
  let s : Spot.Wires.Summary := ⟨wires, wires.map (Spot.Wires.rowWords W)⟩
  say s!"summary built (disjointness bypassed; order not checked)"
  say s!"unwhole (leaf-widths): {(unwhole f.P f.ts f.leaves s).getD "none"}"
  let x ← match Spot.RecordText.inputsOf f.ts W f.doc with
    | .ok x => pure x
    | .error e => say s!"inputsOf: REFUSED {e}"; return 1
  say s!"inputsOf: ok"
  let theta := fun (p : String) => ((thetaOf f.T f.doc x).lookup p).getD false
  say s!"sharedHeld: {sharedHeld f.P f.ts W x.decls theta}"
  say s!"formatsHeld: {formatsHeld f.T f.doc x.params}"
  match Spot.RecordText.gridsOf f.ts f.doc with
  | .ok (S, grids) =>
    say s!"gridsOf: S = {S}, {grids.length} templates"
    let gridsAt := fun (tid : String) => (grids.lookup tid).getD []
    let inside := (wires.map fun w => sameSlot S (gridsAt w.reader) (gridsAt w.writer) w).toArray
    say s!"inside (sameSlot per wire): {inside.size}"
  | .error e => say s!"gridsOf: REFUSED {e}"
  return 0
