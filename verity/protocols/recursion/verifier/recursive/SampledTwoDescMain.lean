module

public import VStarRecursive.SampledTwoDesc

/-!
# A test's executable: the two-stage form's descriptor check on one play

`sampled-two-desc check PARAMS COMMITMENT EVIDENCE [BLOB ...]` decodes a play as the form's check does (the params, the
boundary registration, the evidence, the blobs, and the run of record's input from `SampledTwo.inputAt`), then runs the
descriptor check (`SampledTwoDesc.descOk`) on it twice:

* on the play as it is, printing `pass` or the first of its clauses that fails (`why`);
* on the play with its second template given the first's name, which `namesOnce` must refuse.

It exits 0 when the play passes, the renamed play is refused at `namesOnce`, and `why` agrees with `descOk` on both.
No party runs it and nothing dispatches to it: nothing in the form calls `descOk` yet.
-/

@[expose] public section

namespace VStar.Recursive.SampledTwoDesc

open Lean (Json)
open Flock

/-- The first clause of `cs` that fails. -/
def firstFail (cs : List (String × Bool)) : Option String := (cs.find? fun c => !c.2).map (·.1)

/-- **Why `descOk` refuses**, clause by clause in its order: none where every clause holds. -/
def why (p : SampledTwo.Params) (bs : Core.Blobs) (i : Spot.Run.Input) (it : Spot.Run.Interior) : Option String :=
  match firstOf i, cellsOf i it, refsOf p bs with
  | none, _, _ => some "firstOf"
  | _, none, _ => some "cellsOf"
  | _, _, none =>
    let errs := (List.range p.circuits.size).filterMap fun t => match refOf p bs t with
      | .error e => some s!"template {t}: {e}"
      | .ok _ => none
    some s!"refsOf ({"; ".intercalate errs})"
  | some f, some (ns, ks), some rs =>
    match (Spot.Slots.layout f.verifier.program f.verifier.query).toOption,
        (firstLaw f.rule f.verifier.program f.verifier.query f.verifier.workTable).toOption with
    | none, _ => some "layout"
    | _, none => some "firstLaw"
    | some l, some law =>
      let T := it.circuits.size
      let names := (it.circuits.map (·.1)).toList
      let links := Spot.SecondStage.replayLinks i.params.verifier.program.prog it.replay.prog.programFn i.record
      let wires := Spot.SecondStage.replayWires i.params.verifier.program.prog it.replay.prog.programFn
      let refReads := fun t => (it.circuits.map (·.2.2.2)).toList.getD t .null
      let circ := circAt rs
      let vi := it.params.verifier
      firstFail [
        ("as many templates as the params name", p.circuits.size == T), ("a reference statement", !rs.isEmpty),
        ("fits", rs.all fits), ("zkOk", rs.all fun r => zkOk r.spec r.sts), ("templateOk", rs.all templateOk),
        ("scopeOkH", rs.all fun r => ProvedScope.scopeOkH r.stmt),
        ("memberOk", (List.range T).all fun t => memberOk (circ t) (requests l * ns.getD t 0)),
        ("designOk", designOk law l), ("a count a template", ks.size == T), ("counts at least 1", ks.toList.all (1 ≤ ·)),
        ("a cell a template", ns.size == T), ("namesOnce", namesOnce T names),
        ("writersFirst", writersFirst T names wires), ("sourceRowsNamed", sourceRowsNamed T names links wires circ),
        ("wiresAlike", wiresAlike T names wires circ),
        ("writtenRowsUnwired", writtenRowsUnwired T names links wires circ),
        ("writtenRowsRead", writtenRowsRead T names links wires circ),
        ("refReadsOwn", refReadsOwn T names links wires refReads (vi.ports.map (·.1)).toList),
        ("registersNone", (vi.registered.map (·.1)).toList.isEmpty), ("portsOnce", portsOnce T circ),
        ("outsOk", outsOk T circ)]

/-- The play's parameters, blobs, run of record's input and its interior, decoded as the form's check decodes them. -/
def decode (params commitment evidence : ByteArray) (blobs : Array ByteArray) :
    V (SampledTwo.Params × Core.Blobs × Spot.Run.Input × Spot.Run.Interior) := do
  let p ← Core.canonicalJson params >>= SampledTwo.paramsOfJson
  let c ← Spot.OneStage.commitmentOf commitment
  let e ← SampledTwo.evidenceOf evidence
  let bs := Core.Blobs.ofArray blobs
  let (_, _, _, i) ← SampledTwo.inputAt p c e.source e bs
  let some it := i.interior | throw "the run of record's input has no interior"
  pure (p, bs, i, it)

/-- The interior with its second template given the first's name. -/
def renamed (it : Spot.Run.Interior) : Spot.Run.Interior :=
  match it.circuits[0]?, it.circuits[1]? with
  | some a, some b => { it with circuits := it.circuits.set! 1 (a.1, b.2) }
  | _, _ => it

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
    | .ok (p, bs, i, it) =>
      let ok := descOk p bs i it
      let w := why p bs i it
      IO.println (report "play" ok w)
      let it' := renamed it
      let ok' := descOk p bs i it'
      let w' := why p bs i it'
      IO.println (report "play with template 1 named as template 0" ok' w')
      let agree := (ok == w.isNone) && (ok' == w'.isNone)
      if !agree then IO.println "why disagrees with descOk"
      return if agree && ok && !ok' && w' == some "namesOnce" then 0 else 1
  | _ =>
    IO.eprintln "usage: sampled-two-desc check PARAMS COMMITMENT EVIDENCE [BLOB ...]"
    return 64
