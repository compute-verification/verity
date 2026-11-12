module

public import Spot.Bits
public import Flock.ProvedScope
public import Flock.Zk
public import Flock.HmRow
public import Flock.Program
public import Lean.Data.Json

/-!
# A scratch probe: `Spot.Bits.check` on a round's circuit files

`bits-check-probe PROGRAM CIRCUIT...`: each circuit file read as `verify --zk` reads a session's statement circuit
(`ProvedScope.circuitOf` at `Zk.tags Tags.circuit`, then `HmOut.unrolled`), held to its own template (META
`program_digests`' key, `HmRow.queryTemplate`) of the Program `PROGRAM` by `Spot.Bits.check`. Beside the verdict it
prints, for each file, what `Spot.Bits.unitReads` and `readsWhy` say of each parameter's bits, and each step's time.

One line a file, `BITSCHECK {json}`, and a last line `BITSCHECK-RESULT {json}`. Exit 0 when `check` accepts every
file, 1 when it refuses one, 2 when a file doesn't parse. Scratch: nothing calls it and it lands nowhere.
-/

@[expose] public section

namespace Spot.BitsCheckProbe

open Lean (Json)

def timed {α : Type} (act : Unit → α) : IO (α × Nat) := do
  let t0 ← IO.monoMsNow
  let x ← IO.lazyPure act
  let t1 ← IO.monoMsNow
  return (x, t1 - t0)

/-- Each parameter of `d`: its name, its bits, how many of them no row of the unit reads, and the first such bit, by the
columns `Spot.Bits.readsWhy` reads them at. -/
def unreadOf (c : Flock.Circuit) (d : Flock.Program.Defn) (reads : Array Bool) :
    Array (String × Nat × Nat × Option Nat) := Id.run do
  let pc := Flock.portCols c.unit.inGroups
  let mut P := 0
  let mut out := #[]
  for (nm, t) in d.params do
    let n := Spot.Bits.bitsOf t
    let mut unread := 0
    let mut first : Option Nat := none
    for β in [0:n] do
      unless reads.getD ((pc.getD (P + β / 16) (0, 0)).1 + β % 16) false do
        unread := unread + 1
        if first.isNone then first := some β
    out := out.push (nm, n, unread, first)
    P := P + Spot.Bits.wordsOf t
  return out

def optStr : Option String → Json
  | some s => .str s
  | none => .null

def optNat : Option Nat → Json
  | some n => Lean.toJson n
  | none => .null

/-- One file's line: `(json, accepted, parsed)`. -/
def one (P : Flock.Program.Prog) (path : String) : IO (Json × Bool × Bool) := do
  let bytes ← IO.FS.readBinFile path
  let tags := Flock.Zk.tags Flock.Tags.circuit (Flock.Zk.maskLogOf bytes) 1
  let t0 ← IO.monoMsNow
  let parsed ← IO.lazyPure fun _ => Flock.ProvedScope.circuitOf tags bytes []
  let t1 ← IO.monoMsNow
  match parsed with
  | .error e =>
    return (Json.mkObj [("file", .str path), ("bytes", Lean.toJson bytes.size), ("parse", .str e)], false, false)
  | .ok c0 =>
    let c := Flock.HmOut.unrolled c0
    let tmpl := (Flock.HmRow.queryTemplate c).map (·.1)
    let key := tmpl.getD ""
    let d? := P.defs.find? (·.key == key)
    let root := P.defs.getD P.root default
    let calls := root.body.nodes.toList.filterMap fun x =>
      if (P.defs.getD x.fn default).key == key then some (Json.mkObj [("form", .str x.form.name), ("n", Lean.toJson x.n)])
      else none
    let (why, checkMs) ← timed fun _ => Spot.Bits.check c P key
    let (reads, readsMs) ← timed fun _ => Spot.Bits.unitReads c
    let R := Spot.Bits.Rows.ofNet c.unit
    let params := match d? with
      | some d => (unreadOf c d reads).toList.map fun (nm, n, unread, first) =>
          Json.mkObj [("name", .str nm), ("bits", Lean.toJson n), ("words", Lean.toJson ((n + 15) / 16)),
            ("unread", Lean.toJson unread), ("first_unread", optNat first),
            ("readsWhy", optStr (Spot.Bits.readsWhy c d reads [nm]))]
      | none => []
    let ret := match d? with
      | some d => (Spot.Bits.fieldsOf d.ret).map fun t =>
          Json.mkObj [("bits", Lean.toJson (Spot.Bits.bitsOf t)), ("words", Lean.toJson (Spot.Bits.wordsOf t))]
      | none => []
    let padding := match d? with
      | some d => optStr (Spot.Bits.readsWhy c d reads [])
      | none => .null
    let j := Json.mkObj [
      ("file", .str path), ("bytes", Lean.toJson bytes.size), ("name", .str c.name),
      ("templated", .bool c0.tmpl.isSome), ("template", optStr tmpl),
      ("definition_of_program", .bool d?.isSome), ("root_calls", Json.arr calls.toArray),
      ("program_sha512_meta", (c.metaJ.getObjVal? "program_sha512").toOption.getD .null),
      ("upv", Lean.toJson c.upv), ("vus_per_block", Lean.toJson c.g), ("k_log", Lean.toJson c.kLog),
      ("nets", Lean.toJson c.nets.size),
      ("ports", Json.arr (c.ports.map fun p => Json.arr #[.str p.name, Lean.toJson p.words])),
      ("unit_in_columns", Lean.toJson R.nIn), ("unit_rows", Lean.toJson R.nComp),
      ("unit_in_ports", Lean.toJson (Flock.portCols c.unit.inGroups).size),
      ("check", optStr why), ("accepted", .bool why.isNone),
      ("params", Json.arr params.toArray), ("return_fields", Json.arr ret.toArray),
      ("reads_padding", padding),
      ("read_columns", Lean.toJson (reads.foldl (fun n b => if b then n + 1 else n) 0)),
      ("parse_ms", Lean.toJson (t1 - t0)), ("check_ms", Lean.toJson checkMs), ("unit_reads_ms", Lean.toJson readsMs)]
    return (j, why.isNone, true)

def run (args : List String) : IO UInt32 := do
  match args with
  | program :: files =>
    let pbytes ← IO.FS.readBinFile program
    match Flock.Program.decode pbytes with
    | .error e =>
      IO.println s!"BITSCHECK-RESULT {(Json.mkObj [("program", .str e)]).compress}"
      return 2
    | .ok P =>
      let root := P.defs.getD P.root default
      let nodes := root.body.nodes.toList.map fun x =>
        Json.mkObj [("form", .str x.form.name), ("n", Lean.toJson x.n), ("fn", .str (P.defs.getD x.fn default).key)]
      IO.println s!"BITSCHECK-PROGRAM {(Json.mkObj [("sha512", .str (Flock.hexOf P.sha512)),
        ("definitions", Lean.toJson P.defs.size), ("root", .str root.key),
        ("root_nodes", Json.arr nodes.toArray)]).compress}"
      let mut accepted := 0
      let mut refused := 0
      let mut unparsed := 0
      for f in files do
        let (j, ok, parsed) ← one P f
        IO.println s!"BITSCHECK {j.compress}"
        (← IO.getStdout).flush
        if !parsed then unparsed := unparsed + 1
        else if ok then accepted := accepted + 1
        else refused := refused + 1
      IO.println s!"BITSCHECK-RESULT {(Json.mkObj [("files", Lean.toJson files.length),
        ("accepted", Lean.toJson accepted), ("refused", Lean.toJson refused),
        ("unparsed", Lean.toJson unparsed)]).compress}"
      return if unparsed > 0 then 2 else if refused > 0 then 1 else 0
  | [] =>
    IO.eprintln "usage: bits-check-probe PROGRAM CIRCUIT..."
    return 2

end Spot.BitsCheckProbe

def main (args : List String) : IO UInt32 := Spot.BitsCheckProbe.run args
