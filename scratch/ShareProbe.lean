import InferenceOnlyVerifier.Record
open InferenceOnlyVerifier InferenceOnlyVerifier.Record
open Flock.Program Spot.RecordText

/-- scratch (lean/WIRED): `sharedDecide` row by row on one case: which test refuses each shared run's row (no
declaration covers it, it isn't a registered read, the plan's shape test, or `heldFrom`), the declaration that covers
it, and θ of its parameter. Never pushed to a shared name. -/
def main (args : List String) : IO UInt32 := do
  let [path] := args | IO.eprintln "usage: wired-share CASE"; return 2
  let say (s : String) : IO Unit := do IO.println s; (← IO.getStdout).flush
  let .ok c := Lean.Json.parse (← IO.FS.readFile path) | IO.eprintln "not JSON"; return 2
  let j := c.getObjValD "instance"; let tr := c.getObjValD "trusted"
  let f ← match front j tr with
    | .ok f => pure f
    | .error e => say s!"front: REFUSED {e}"; return 1
  let rs0 ← match Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg f.P) (f.ts.map (·.tid)).toArray with
    | .ok rs => pure rs
    | .error _ => say "ranges: REFUSED"; return 1
  let cuts ← match Spot.RecordText.cutsOf f.doc with
    | .ok c => pure c
    | .error e => say s!"cuts: REFUSED {e}"; return 1
  let W0 : Spot.Wires.Wiring :=
    { templates := f.ts.map fun t => (t.tid, t.wired), cuts := cuts, wires := [], order := some (Spot.RecordText.orderOf rs0) }
  let W ← match Spot.Edges.edgesOf f.P rs0 W0 with
    | .ok ws => pure { W0 with wires := ws }
    | .error e => say s!"edges: REFUSED {e}"; return 1
  let x ← match Spot.RecordText.inputsOf f.ts W f.doc with
    | .ok x => pure x
    | .error e => say s!"inputsOf: REFUSED {e}"; return 1
  let theta := fun (p : String) => ((thetaOf f.T f.doc x).lookup p).getD false
  let reads := x.decls.filter fun d => match d.fill with | .read _ => true | _ => false
  say s!"params {x.params.length}, θ holds {(x.params.filter (theta ·.1)).length}; declarations {x.decls.length}, registered reads {reads.length}"
  say s!"registered reads by parameter (first 12): {((reads.filterMap fun d => match d.fill with | .read p => some p | _ => none).eraseDups.take 12)}"
  say s!"sharedRuns: {(sharedRuns f.P).length}"
  let .ok rs := Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg f.P) (W.templates.map (·.1)).toArray
    | say "ranges: REFUSED"; return 1
  let some sh := sharedOf f.P | say "sharedOf: none (refused)"; return 1
  say s!"shared arguments: {sh.length}"
  let nodes := (f.P.defs[f.P.root]!).body.nodes
  let mut tally : Std.HashMap String Nat := {}
  let mut shown := 0
  for s in sh do
    let some r := rs.find? (·.node == s.node) | tally := tally.insert "no range" ((tally.getD "no range" 0) + 1); continue
    let tid := r.template
    let u0 := (rangesAt rs tid).foldl (fun a q => if q.first < r.first then a + q.count else a) 0
    let some t := f.ts.find? (·.tid == tid) | continue
    let some (port, _) := (f.P.defs[t.defn]!).params[s.arg]? | continue
    let words := (t.rows.lookup port).map (·.1) |>.getD 0
    let k := W.cutsOf tid port
    let wn := ((nodes[s.input]?).map fun nd => inputWidth? (f.P.defs[nd.fn]!).key).join
    for cc in List.range k do
      let d := x.decls.find? (fun d => (d.w.indexOf tid u0 port cc).isSome)
      let why ← match sharedPlan f.P rs f.ts W x.decls s cc with
        | none =>
          match d with
          | none => pure "no declaration covers the row"
          | some d => match d.fill with
            | .read _ => pure "a registered read covers it; the plan's shape test refuses"
            | .wire _ => pure "a wire fills it"
            | _ => pure "another declaration fills it"
        | some (_, _, _, n, p, g, gb, _) =>
          match heldFrom x.decls theta tid port cc u0 (u0 + n) p g gb (x.decls.length + 1) u0 with
          | some true => pure "held"
          | some false => pure s!"heldFrom false (θ {p}: {theta p})"
          | none => pure "heldFrom past its bound"
      tally := tally.insert why ((tally.getD why 0) + 1)
      if why != "held" && shown < 4 then
        shown := shown + 1
        say s!"REFUSED ROW: node {s.node} ({(f.P.defs[nodes[s.node]!.fn]!).id}, {nodes[s.node]!.members} members) argument {s.arg} ({port}, {words} words, cut {k} ways, row {cc}); input node {s.input} ({(f.P.defs[nodes[s.input]!.fn]!).id}, leaf width {wn}); run base {s.base} inner {s.inner} count {s.count}; template {tid} units from {u0}: {why}"
        match d with
        | some d => say s!"   covering declaration: {repr d.fill}, reader {d.w.reader} port {d.w.port} start {d.w.start} units {d.w.units} radix {d.w.radix} base {d.w.base} step {d.w.step} cut0 {d.w.cut0} cuts {d.w.cuts}"
        | none => pure ()
  say s!"TALLY rows: {tally.toList}"
  return 0
