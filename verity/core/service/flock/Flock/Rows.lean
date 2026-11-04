/-
`flock-rows TYPES --root DIGEST --name NAME [--in-split A,B..] [--out-split A,B..] [--leaves FILE] [--cut FILE] [--logical]`:
the rows `Flock.Derive.derive` gives the type `DIGEST`, as `flock-ir-unit/v2` on stdout, or with `--logical` its logical list
(`Derive.logicalText`). `TYPES` is `{"types": [content, ...]}`,
callees before callers, each checked as `flock-verify circuit-type` checks it. `--leaves` and `--cut` name files holding
the lowering's `LEAVES` and `CUT` JSON, which are passed through. The Python layout is a mirror held to this output
(`backends/flock/tests/test_flock_rows.py`).
-/
import Flock.Derive
import Flock.DeriveCheck

namespace Flock.Rows

open Lean

def argOf (args : List String) (k : String) : Option String :=
  match args.dropWhile (· != k) with
  | _ :: v :: _ => some v
  | _ => none

def splitArg : Option String → List Nat
  | none => []
  | some s => (s.splitOn ",").filterMap String.toNat?

def usage : String :=
  "usage: flock-rows TYPES --root DIGEST --name NAME [--in-split A,B..] [--out-split A,B..] [--leaves FILE] [--cut FILE] [--logical]"

def readLine? (f : Option String) : IO (Option String) :=
  f.mapM fun p => return (← IO.FS.readFile p).trimAsciiEnd.toString

/-- `flock-rows --archive FILE --part unit|delta|logical|check|DIGEST`: that part of the archive's derivation, once its
rows pass `deriveChecked`'s check, or `ok` when they do. -/
def archiveCmd (path part : String) : IO UInt32 := do
  let run := fun j => if part == "check" then DeriveAll.archiveCheck j else DeriveAll.archivePart j part
  match Json.parse (← IO.FS.readFile path) >>= run with
  | .ok s => IO.print s; return 0
  | .error e => IO.eprintln s!"flock-rows: {e}"; return 1

/-- `flock-rows`: the command. -/
def cmd (args : List String) : IO UInt32 := do
  if let (some path, some part) := (argOf args "--archive", argOf args "--part") then return ← archiveCmd path part
  let some path := args.head? | IO.eprintln usage; return 2
  let (some root, some name) := (argOf args "--root", argOf args "--name") | IO.eprintln usage; return 2
  match Json.parse (← IO.FS.readFile path) >>= fun j => j.getObjValAs? (Array Json) "types" with
  | .error e => IO.eprintln s!"flock-rows: not a types file: {e}"; return 1
  | .ok ts =>
    let mut known : Array (String × CircuitType) := #[]
    for j in ts do
      let t ← match CircuitType.ofJson j with
        | .ok t => pure t
        | .error e => IO.eprintln s!"flock-rows: {e}"; return 1
      let d ← match t.digest with
        | .ok d => pure d
        | .error e => IO.eprintln s!"flock-rows: {e}"; return 1
      let callee := fun (x : String) => (known.find? (·.1 == x)).map (·.2)
      match CircuitType.check callee t with
      | .ok () => known := known.push (d, t)
      | .error e => IO.eprintln s!"flock-rows: type {d} breaks the rule '{e}'"; return 1
    let some (_, t) := known.find? (·.1 == root) | IO.eprintln s!"flock-rows: no type {root}"; return 1
    match Derive.derive t (splitArg (argOf args "--in-split")) (splitArg (argOf args "--out-split")) with
    | .error e => IO.eprintln s!"flock-rows: {e}"; return 1
    | .ok d =>
      if args.contains "--logical" then IO.print (Derive.logicalText d)
      else IO.print (Derive.netlist name d (← readLine? (argOf args "--leaves")) (← readLine? (argOf args "--cut")))
      return 0

end Flock.Rows
