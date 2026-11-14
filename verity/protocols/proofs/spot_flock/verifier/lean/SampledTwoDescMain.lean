module

public import VStarRecursive.SampledTwoDesc

/-!
# A test's executable: the two-stage form's descriptor check on one play

`sampled-two-desc check PARAMS COMMITMENT EVIDENCE [BLOB ...]` decodes a play as the form's check does (the params, the
boundary registration, the evidence, the blobs, and the run of record's input from `SampledTwo.inputAt`), then runs the
descriptor check (`SampledTwoDesc.descOk`) on it. Where the form refuses the play before its run of record (a session's
verifier that does not accept), it prints the form's reason and runs the check all the same: the check reads the
params, the boundary registration, the rule and the interior's params, replay block and circuits, and no session.

* on the play as it is, printing its verdict and how long the check took, then `pass` or the first of its clauses that
  fails (`why`), every clause that fails (`fails`) and, for each template, the numbers `memberOk` reads and its count
  of units (`sizes`);
* on the play with its second template given the first's name, which `namesOnce` must refuse; a play of fewer than two
  templates has no such case, and the line says so;
* on the play with its interior registering a value, which the check must refuse at that clause.

It exits 0 when the play passes, each refused case is refused at its clause, and `why` agrees with `descOk` on all.
No party runs it and nothing dispatches to it: nothing in the form calls `descOk` yet.
-/

@[expose] public section

namespace VStar.Recursive.SampledTwoDesc

open Lean (Json)
open Flock

/-- The first clause of `cs` that fails. -/
def firstFail (cs : List (String × Bool)) : Option String := (cs.find? fun c => !c.2).map (·.1)

/-- **`descOk`'s clauses**, in its order, each with its verdict; an error where a step before them is refused. -/
def clauses (p : SampledTwo.Params) (bs : Core.Blobs) (i : Spot.Run.Input) (it : Spot.Run.Interior) :
    Except String (List (String × Bool)) :=
  match firstOf i, cellsOf i it, refsOf p bs with
  | none, _, _ => .error "firstOf"
  | _, none, _ => .error "cellsOf"
  | _, _, none =>
    let errs := (List.range p.circuits.size).filterMap fun t => match refOf p bs t with
      | .error e => some s!"template {t}: {e}"
      | .ok _ => none
    .error s!"refsOf ({"; ".intercalate errs})"
  | some f, some (ns, ks), some rs =>
    match (Spot.Slots.layout f.verifier.program f.verifier.query).toOption,
        (firstLaw f.rule f.verifier.program f.verifier.query f.verifier.workTable).toOption with
    | none, _ => .error "layout"
    | _, none => .error "firstLaw"
    | some l, some law =>
      let T := it.circuits.size
      let names := (it.circuits.map (·.1)).toList
      let links := Spot.SecondStage.replayLinks i.params.verifier.program.prog it.replay.prog.programFn i.record
      let wires := Spot.SecondStage.replayWires i.params.verifier.program.prog it.replay.prog.programFn
      let rows := Spot.SecondStage.replayRows i.params.verifier.program.prog it.replay.prog.programFn
      let refReads := fun t => (it.circuits.map (·.2.2.2)).toList.getD t .null
      let circ := circAt (circsOf rs)
      let vi := it.params.verifier
      .ok [
        ("as many templates as the params name", p.circuits.size == T), ("a reference statement", !rs.isEmpty),
        ("fits", rs.all fits), ("zkOk", rs.all fun r => zkOk r.spec r.sts), ("templateOk", rs.all templateOk),
        ("scopeOkH", rs.all fun r => ProvedScope.scopeOkH r.stmt),
        ("memberOk", (List.range T).all fun t => memberOk (circ t)),
        ("designOk", designOk law l), ("a count a template", ks.size == T), ("counts at least 1", ks.toList.all (1 ≤ ·)),
        ("a cell a template", ns.size == T), ("namesOnce", namesOnce T names),
        ("writersFirst", writersFirst T names wires), ("sourceRowsNamed", sourceRowsNamed T names links wires circ),
        ("wiresMoveRows", wiresMoveRows T names links wires rows circ ns (unitsAt (requests l) ns T)),
        ("wiresAlike", wiresAlike T names wires circ),
        ("writtenRowsUnwired", writtenRowsUnwired T names links wires circ),
        ("writtenRowsRead", writtenRowsRead T names links wires circ),
        ("refReadsOwn", refReadsOwn T names links wires refReads (vi.ports.map (·.1)).toList),
        ("registersNone", (vi.registered.map (·.1)).toList.isEmpty), ("portsOnce", portsOnce T circ),
        ("outsOk", outsOk T circ)]

/-- **Why `descOk` refuses**, off its clauses' verdicts, clause by clause in its order: none where every clause
holds. -/
def why : Except String (List (String × Bool)) → Option String
  | .error e => some e
  | .ok cs => firstFail cs

/-- **Every clause of `descOk` that fails**, off its clauses' verdicts, in its order; the refused step alone where one
before them is. -/
def fails : Except String (List (String × Bool)) → List String
  | .error e => [e]
  | .ok cs => (cs.filter fun c => !c.2).map (·.1)

/-- **The numbers `memberOk` reads**, a line a template: its statement circuit's unit, a unit's inputs and returned
output bits against the bound, and the template's units (requests times cells), which the bound does not read. -/
def sizes (p : SampledTwo.Params) (bs : Core.Blobs) (i : Spot.Run.Input) (it : Spot.Run.Interior) : List String :=
  match firstOf i, cellsOf i it, refsOf p bs with
  | some f, some (ns, ks), some rs =>
    match (Spot.Slots.layout f.verifier.program f.verifier.query).toOption with
    | some l =>
      let cs := circsOf rs
      (List.range it.circuits.size).map fun t =>
        let c := circAt cs t
        s!"template {t}: unitLog {c.unit.unitLog}, upv {c.upv}, inWords {c.unit.inWords}, " ++
          s!"constPos {c.unit.constPos}, unitIns {unitIns c}, outBits {outBits c}, " ++
          s!"bound {Spot.OneStageZk.UNIT_CLASS_MAX}, memberOk {memberOk c}, requests {requests l}, " ++
          s!"cells {ns.getD t 0}, count {ks.getD t 0}, units {unitsAt (requests l) ns it.circuits.size t}"
    | none => []
  | _, _, _ => []

/-- The play's parameters, blobs, run of record's input and its interior, decoded as the form's check decodes them
(`SampledTwo.inputAt`), with no reason. Where the form refuses the play there, the reason is the form's and the input
is `SampledTwo.inputOf`'s with no verdict, second-stage byte or receipt: `descOk` reads none of those, and the fields
it does read (`params`, `record`, `rule`, `closure`, and the interior's `params`, `replay` and `circuits`) are the same
in both. -/
def decode (params commitment evidence : ByteArray) (blobs : Array ByteArray) :
    V (SampledTwo.Params × Core.Blobs × Spot.Run.Input × Spot.Run.Interior × Option String) := do
  let p ← Core.canonicalJson params >>= SampledTwo.paramsOfJson
  let c ← Spot.OneStage.commitmentOf commitment
  let e ← SampledTwo.evidenceOf evidence
  let bs := Core.Blobs.ofArray blobs
  let (i, refused) : Spot.Run.Input × Option String := match SampledTwo.inputAt p c e.source e bs with
    | .ok (_, _, _, i) => (i, none)
    | .error r => (SampledTwo.inputOf p c e.source ByteArray.empty e "" #[], some r)
  let some it := i.interior | throw "the run of record's input has no interior"
  pure (p, bs, i, it, refused)

/-- The interior with its second template given the first's name. -/
def renamed (it : Spot.Run.Interior) : Spot.Run.Interior :=
  match it.circuits[0]?, it.circuits[1]? with
  | some a, some b => { it with circuits := it.circuits.set! 1 (a.1, b.2) }
  | _, _ => it

/-- The interior with its registration registering a value. -/
def registering (it : Spot.Run.Interior) : Spot.Run.Interior :=
  { it with params := { it.params with verifier := { it.params.verifier with registered := #[("x", 1)] } } }

/-- One line: the case, `descOk`'s verdict and `why`'s. -/
def report (what : String) (ok : Bool) (w : Option String) : String :=
  s!"{what}: descOk {ok}, {match w with | none => "pass" | some c => s!"refused at {c}"}"

end VStar.Recursive.SampledTwoDesc

open VStar.Recursive.SampledTwoDesc in
def main (args : List String) : IO UInt32 := do
  match args with
  | "check" :: params :: commitment :: evidence :: blobs =>
    let bs ← blobs.toArray.mapM fun (f : String) => IO.FS.readBinFile f
    match decode (← IO.FS.readBinFile params) (← IO.FS.readBinFile commitment) (← IO.FS.readBinFile evidence) bs with
    | .error e =>
      IO.println s!"undecoded: {e}"
      return 2
    | .ok (p, bs, i, it, refused) =>
      if let some r := refused then
        IO.println s!"the form refuses the play before its run of record ({r}); the descriptor check reads no session"
      let out ← IO.getStdout
      let t₀ ← IO.monoMsNow
      let ok := descOk p bs i it
      IO.println s!"play: descOk {ok} on {it.circuits.size} templates"
      let t₁ ← IO.monoMsNow
      IO.println s!"play: descOk took {t₁ - t₀} ms"
      out.flush
      let cs := clauses p bs i it
      let w := why cs
      IO.println (report "play" ok w)
      let fs := fails cs
      IO.println s!"play: clauses that fail: {if fs.isEmpty then "none" else "; ".intercalate fs}"
      for s in sizes p bs i it do IO.println s
      out.flush
      let two : Bool := decide (2 ≤ it.circuits.size)
      let itN := renamed it
      let okN := descOk p bs i itN
      let wN := why (clauses p bs i itN)
      if two then IO.println (report "play with template 1 named as template 0" okN wN)
      else IO.println s!"play with template 1 named as template 0: not made, the play has {it.circuits.size} template"
      out.flush
      let itR := registering it
      let okR := descOk p bs i itR
      let wR := why (clauses p bs i itR)
      IO.println (report "play with its interior registering a value" okR wR)
      let agree := (ok == w.isNone) && (okN == wN.isNone) && (okR == wR.isNone)
      if !agree then IO.println "why disagrees with descOk"
      let named := !two || (!okN && wN == some "namesOnce")
      return if agree && ok && named && !okR && wR == some "registersNone" then 0 else 1
  | _ =>
    IO.eprintln "usage: sampled-two-desc check PARAMS COMMITMENT EVIDENCE [BLOB ...]"
    return 64
