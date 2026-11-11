import NciCheck.Record

/-! A probe, never for landing: on every Program of a file of difftest cases (or of one instance file), the test
`RecordText.templatesOf` made of the root's nodes before (no root node calls a composite keyed as an input id)
against the test it makes now (no root node's callee's kind and key disagree), and the root's nodes by kind.
  lean --run PrimProbe.lean cases FILE     (one case a line, the Program at `instance.program`)
  lean --run PrimProbe.lean instance FILE  (one JSON object, the Program at `program`)
Exit 1 where a Program is refused by one test and not by the other, or the copy of the test of now is not the
code's. -/

open Lean (Json)
open Flock.Program

/-- The test of before, word for word. -/
def before (P : Prog) : Bool :=
  (P.defs[P.root]!).body.nodes.any fun n => !(P.defs[n.fn]!).prim && isInputId (P.defs[n.fn]!).key

/-- The test of now, word for word. -/
def now (P : Prog) : Bool :=
  (P.defs[P.root]!).body.nodes.any fun n => (P.defs[n.fn]!).prim != isInputId (P.defs[n.fn]!).key

structure Tally where
  programs : Nat := 0
  undecoded : Nat := 0
  rootNodes : Nat := 0
  /-- root nodes of a primitive keyed as an input id: the input nodes -/
  inputs : Nat := 0
  /-- root nodes of a composite keyed as no input id: the units -/
  units : Nat := 0
  /-- root nodes of a composite keyed as an input id -/
  keyed : Nat := 0
  /-- root nodes of a primitive keyed as no input id: units of a primitive -/
  primUnits : Nat := 0
  refusedBefore : Nat := 0
  refusedNow : Nat := 0
  /-- Programs one test refuses and the other does not -/
  moves : Nat := 0
  /-- Programs on which the copy of the test of now is not the code's -/
  copyDiffers : Nat := 0

def Tally.line (t : Tally) : String :=
  s!"programs {t.programs} (not decoded {t.undecoded}); root nodes {t.rootNodes}: input primitives {t.inputs}, " ++
  s!"composites keyed as no input {t.units}, composites keyed as an input {t.keyed}, primitives keyed as no input " ++
  s!"{t.primUnits}; Programs refused by the test of before {t.refusedBefore}, by the test of now {t.refusedNow}; " ++
  s!"Programs on which the two differ {t.moves}; copy differs from the code {t.copyDiffers}"

def step (t : Tally) (text : String) : Tally := Id.run do
  let mut t := { t with programs := t.programs + 1 }
  match Flock.Program.decode text.toUTF8 with
  | .error _ => return { t with undecoded := t.undecoded + 1 }
  | .ok P =>
    for n in (P.defs[P.root]!).body.nodes do
      let prim := (P.defs[n.fn]!).prim
      let inp := isInputId (P.defs[n.fn]!).key
      t := { t with rootNodes := t.rootNodes + 1 }
      t := match prim, inp with
        | true, true => { t with inputs := t.inputs + 1 }
        | false, false => { t with units := t.units + 1 }
        | false, true => { t with keyed := t.keyed + 1 }
        | true, false => { t with primUnits := t.primUnits + 1 }
    let a := before P
    let b := now P
    if a then t := { t with refusedBefore := t.refusedBefore + 1 }
    if b then t := { t with refusedNow := t.refusedNow + 1 }
    if a != b then t := { t with moves := t.moves + 1 }
    if b != NciCheck.RecordText.miskeyed P then t := { t with copyDiffers := t.copyDiffers + 1 }
    return t

def main (args : List String) : IO UInt32 := do
  let [mode, path] := args | IO.eprintln "usage: PrimProbe.lean cases|instance FILE"; return 2
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
  IO.println s!"PRIMITIVE-UNIT PROBE {mode} {path}: {t.line}"
  return (if t.moves == 0 && t.copyDiffers == 0 then 0 else 1)
