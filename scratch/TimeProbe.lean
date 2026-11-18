import InferenceOnlyVerifier.Record
open InferenceOnlyVerifier InferenceOnlyVerifier.Record
open Flock.Program

/-- scratch (lean/WIRED): each stage of `Record.read` timed on one case, `Spot.Wires.orderedWire` timed against the
trial's `orderedWireAt` on prefixes of the wires, and the stages after `Spot.Wires.check`. Never pushed to a shared
name. -/
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
  let e0 ← IO.monoMsNow
  let wires ← match (← IO.lazyPure fun _ => Spot.Edges.edgesOf f.P rs W0) with
    | .ok ws => pure ws
    | .error e => say s!"edges: REFUSED {e}"; return 1
  say s!"edgesOf (proofs' head): {wires.length} wires, {(← IO.monoMsNow) - e0} ms"
  let W := { W0 with wires := wires }
  say s!"check.cutsOk: {Spot.Wires.cutsOk W}"
  let fw := Spot.Wires.firstWire W (Spot.Wires.wireCode W)
  say s!"check.firstWire: {repr fw}"
  if let some ⟨_, some i⟩ := fw then
   if let some w := wires[i]? then
    let rw := ((W.templates.lookup w.reader).bind fun (t : Spot.Wires.Template) => (t.rows.lookup w.port).map (·.words)).getD 0
    let ow := ((W.templates.lookup w.writer).bind fun (t : Spot.Wires.Template) => t.outs.lookup w.out).getD 0
    say s!"  wire {i}: reader {w.reader} port {w.port} (row {rw} words, cut {W.cutsOf w.reader w.port} ways), writer {w.writer} out {w.out} ({ow} words), offset {w.offset}, opairs {repr w.opairs}, low {Circuits.Wire.lowOf w.opairs}, high {Circuits.Wire.highOf w.opairs}, start {w.start}, radix {w.radix}, base {w.base}, step {w.step}, cut0 {w.cut0}, cuts {w.cuts}"
    let same := (wires.filter fun v => v.reader == w.reader && v.port == w.port && v.writer == w.writer).length
    say s!"  wires of that reader port from that writer: {same}; refused wires in all: {(wires.filter fun v => (Spot.Wires.wireCode W v).isSome).length}"
  let timed (what : String) (f : Unit → Bool) : IO Bool := do
    let a ← IO.monoMsNow
    let r ← IO.lazyPure f
    let b ← IO.monoMsNow
    say s!"{what}: {r}, {b - a} ms"; return r
  let _ ← timed s!"pairwiseDisjointSorted on all {wires.length} wires" fun _ => Spot.Wires.pairwiseDisjointSorted wires
  let o0 ← IO.monoMsNow
  let o ← match (← IO.lazyPure fun _ => Spot.Wires.orderOf W) with
    | .ok o => pure o
    | .error r => say s!"orderOf: REFUSED {r.code.name}"; return 1
  say s!"orderOf: {o.length} ranges, {(← IO.monoMsNow) - o0} ms"
  let i0 ← IO.monoMsNow
  let ix ← IO.lazyPure fun _ => Spot.Wires.orderIndex o
  say s!"orderIndex: {ix.size} templates, {(← IO.monoMsNow) - i0} ms"
  for n in [1000, 2000, 4000, 8000] do
    let ws := wires.take n
    let p ← timed s!"orderedWire (shipped) on the first {n} wires" fun _ => ws.all (Spot.Wires.orderedWire o)
    let q ← timed s!"orderedWireAt (trial) on the first {n} wires" fun _ => ws.all (Spot.Wires.orderedWireAt ix)
    let same := ws.all fun w => Spot.Wires.orderedWire o w == Spot.Wires.orderedWireAt ix w
    say s!"  same verdict: {p == q}, same on every wire: {same}"
  let _ ← timed s!"orderedWireAt (trial) on all {wires.length} wires" fun _ => wires.all (Spot.Wires.orderedWireAt ix)
  let c0 ← IO.monoMsNow
  let oc ← IO.lazyPure fun _ => Spot.Wires.orderCodeAt W
  say s!"orderCodeAt (trial, from orderOf): {(← IO.monoMsNow) - c0} ms, {match oc with | none => "none" | some r => s!"{r.code.name} at {r.wire}"}"
  let a1 ← IO.monoMsNow
  let chk ← IO.lazyPure fun _ => Spot.Wires.check W
  say s!"Spot.Wires.check (the trial's): {(← IO.monoMsNow) - a1} ms, {match chk with | .ok _ => "ok" | .error r => s!"{r.code.name} at {r.wire}"}"
  let s : Spot.Wires.Summary := ⟨wires, wires.map (Spot.Wires.rowWords W)⟩
  say s!"summary built "
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
