import NciCheck.Record

/-! A probe, never for landing: on every Program of a file of difftest cases (or of one instance file), `ownRows`
with the test it made of a root batch before (every axis given) against the test it makes now (every argument has
`dims`), and, on every batch node of every Definition, whether an argument has `dims` exactly where its axis is
given.
  lean --run OwnProbe.lean cases FILE     (one case a line, the Program at `instance.program`)
  lean --run OwnProbe.lean instance FILE  (one JSON object, the Program at `program`) -/

open Lean (Json)
open Flock.Program

/-- `ownRows` with the batch test as a parameter: the body of `NciCheck.Record.ownRows`, word for word. -/
def ownRowsWith (test : Node → Bool) (P : Prog) : Bool :=
  let nodes := (P.defs[P.root]!).body.nodes
  let units := nodes.toList.filter fun n => !NciCheck.Desc.isInputNode P n
  let parts := units.flatMap fun n => n.args.toList.flatMap fun a => a.seq.parts.toList
  let runs := parts.filterMap fun
    | .affine false i base count =>
      if (nodes[i]?.map fun n => NciCheck.Desc.isInputNode P n && (n.form == .prim || n.form == .batch)).getD false then
        some (i, base, count)
      else none
    | _ => none
  let sorted := (runs.filter fun r => r.2.2 > 0).mergeSort fun a b => a.1 < b.1 || (a.1 == b.1 && a.2.1 ≤ b.2.1)
  units.all test &&
    runs.length == parts.length &&
    (sorted.zip sorted.tail).all fun (a, b) => a.1 != b.1 || a.2.1 + a.2.2 ≤ b.2.1

def before (n : Node) : Bool := n.form != .batch || (n.axes.size == n.args.size && n.axes.all (·.isSome))
def now (n : Node) : Bool := n.form != .batch || n.args.all (·.dims.isSome)

structure Tally where
  programs : Nat := 0
  undecoded : Nat := 0
  ownTrue : Nat := 0
  ownFalse : Nat := 0
  /-- `ownRows` as compiled differs from the copy with the test of now (the copy is not the code) -/
  copyDiffers : Nat := 0
  /-- the copy with the test of before differs from `ownRows` -/
  verdictMoves : Nat := 0
  /-- Programs whose root has a unit the two tests tell apart -/
  testMoves : Nat := 0
  batches : Nat := 0
  rootBatchUnits : Nat := 0
  mappedArgs : Nat := 0
  unmappedArgs : Nat := 0
  /-- batch nodes with `axes` not as long as `args` -/
  sizeOff : Nat := 0
  /-- arguments with an axis given and no `dims`, or `dims` and no axis -/
  argOff : Nat := 0
  /-- Programs refused by the before-test alone, by the now-test alone -/
  onlyBefore : Nat := 0
  onlyNow : Nat := 0

def Tally.line (t : Tally) : String :=
  s!"programs {t.programs} (not decoded {t.undecoded}); ownRows true {t.ownTrue}, false {t.ownFalse}; " ++
  s!"copy differs from the code {t.copyDiffers}; verdict moves {t.verdictMoves}; batch test moves {t.testMoves} " ++
  s!"(fails before only {t.onlyBefore}, now only {t.onlyNow}); batch nodes {t.batches} " ++
  s!"(root units {t.rootBatchUnits}), arguments mapped {t.mappedArgs}, unmapped {t.unmappedArgs}; " ++
  s!"axes not as long as args {t.sizeOff}; arguments with an axis and no dims or dims and no axis {t.argOff}"

def step (t : Tally) (text : String) : Tally := Id.run do
  let mut t := { t with programs := t.programs + 1 }
  match Flock.Program.decode text.toUTF8 with
  | .error _ => return { t with undecoded := t.undecoded + 1 }
  | .ok P =>
    let code := NciCheck.Record.ownRows P
    let a := ownRowsWith before P
    let b := ownRowsWith now P
    t := if code then { t with ownTrue := t.ownTrue + 1 } else { t with ownFalse := t.ownFalse + 1 }
    if b != code then t := { t with copyDiffers := t.copyDiffers + 1 }
    if a != code then t := { t with verdictMoves := t.verdictMoves + 1 }
    let units := (P.defs[P.root]!).body.nodes.toList.filter fun n => !NciCheck.Desc.isInputNode P n
    if units.any fun n => before n != now n then t := { t with testMoves := t.testMoves + 1 }
    if !units.all before && units.all now then t := { t with onlyBefore := t.onlyBefore + 1 }
    if units.all before && !units.all now then t := { t with onlyNow := t.onlyNow + 1 }
    t := { t with rootBatchUnits := t.rootBatchUnits + (units.filter fun n => n.form == .batch).length }
    for d in P.defs do
      for n in d.body.nodes do
        if n.form == .batch then
          t := { t with batches := t.batches + 1 }
          if n.axes.size != n.args.size then t := { t with sizeOff := t.sizeOff + 1 }
          for i in [0:n.args.size] do
            let ax := (n.axes[i]?.getD none).isSome
            let dm := (n.args[i]!).dims.isSome
            t := if ax then { t with mappedArgs := t.mappedArgs + 1 } else { t with unmappedArgs := t.unmappedArgs + 1 }
            if ax != dm then t := { t with argOff := t.argOff + 1 }
    return t

def main (args : List String) : IO UInt32 := do
  let [mode, path] := args | IO.eprintln "usage: OwnProbe.lean cases|instance FILE"; return 2
  let body ← IO.FS.readFile path
  let texts ← match mode with
    | "cases" =>
      (body.splitOn "\n").filter (! ·.isEmpty) |>.mapM fun line => do
        let .ok c := Json.parse line | throw (IO.userError "a line is not JSON")
        -- a case whose instance names no Program text is counted as not decoded
        pure (((c.getObjValD "instance").getObjValAs? String "program").toOption.getD "")
    | _ => do
      let .ok c := Json.parse body | throw (IO.userError "not JSON")
      let .ok s := c.getObjValAs? String "program" | throw (IO.userError "no program")
      pure [s]
  let t := texts.foldl step {}
  IO.println s!"OWN-ROWS PROBE {mode} {path}: {t.line}"
  return (if t.copyDiffers == 0 && t.verdictMoves == 0 && t.testMoves == 0 && t.sizeOff == 0 && t.argOff == 0 then 0 else 1)
