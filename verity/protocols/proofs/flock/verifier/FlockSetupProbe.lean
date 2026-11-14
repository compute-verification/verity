/- Scratch, never for landing: `flock-setup-probe` times a statement's setup at the sample of no unit in compiled Lean. -/
module

public import Flock
public import Flock.JsonParse

@[expose] public section

open Flock
open Lean (Json)

namespace SetupProbe

def argOf (args : List String) (k : String) : Option String :=
  match args.dropWhile (· != k) with
  | _ :: v :: _ => some v
  | _ => none

def readTables (args : List String) : IO (List (String × ByteArray)) := do
  let some dir := argOf args "--tables" | return []
  let mut out := []
  for t in MUFU_TABLES do
    let p : System.FilePath := dir ++ s!"/{t.name}.u32"
    if ← p.pathExists then out := out ++ [(t.name, ← IO.FS.readBinFile p)]
  return out

def emptyDraw (N : Nat) : Json := Flock.Draw.UnitDraw.toJson ⟨.subset 0, N, #[], #[]⟩

/-- The process's peak and current resident memory, as `/proc/self/status` words them. -/
def mem : IO String := do
  let s ← IO.FS.readFile "/proc/self/status"
  let pick (k : String) : String :=
    (((s.splitOn "\n").filter (·.startsWith k)).headD s!"{k} ?").replace "\t" " "
  return s!"{pick "VmHWM:"}; {pick "VmRSS:"}"

def shape (st : Stmt) : String :=
  s!"n = {st.n}, blocks = {st.blocksReal}, nbl = {st.nbl}, m = {st.m}, k_log = {st.c.kLog}, g = {st.c.g}, " ++
    s!"regions = {st.regions.size}"

/-- `Zk.shapeOk st`, timed. -/
def shapeTimed (st : Stmt) : IO String := do
  let t0 ← IO.monoMsNow
  match Zk.shapeOk st with
  | .ok _ =>
    let t1 ← IO.monoMsNow
    return s!"shapeOk {t1 - t0} ms: ok"
  | .error e =>
    let t1 ← IO.monoMsNow
    return s!"shapeOk {t1 - t0} ms: refused ({e})"

def run (args : List String) : IO UInt32 := do
  let some cfP := argOf args "--circuit" | IO.eprintln "--circuit FILE --public FILE [--partition F] [--program F] [--tables DIR] [--tag NAME] [--whole]"; return 2
  let some pfP := argOf args "--public" | IO.eprintln "--public FILE"; return 2
  let tagName := (argOf args "--tag").getD "verity/flock-circuit"
  let t0 ← IO.monoMsNow
  let cf ← IO.FS.readBinFile cfP
  let pf ← IO.FS.readBinFile pfP
  let partition ← match argOf args "--partition" with
    | none => pure none
    | some p => do
      match JsonParse.parse (← IO.FS.readFile p) with
      | .ok j => pure (some j)
      | .error e => throw (IO.userError s!"partition: {e}")
  let program ← match argOf args "--program" with
    | none => pure none
    | some p => do pure (some (← IO.FS.readBinFile p))
  let tables ← readTables args
  let t1 ← IO.monoMsNow
  IO.println s!"PROBE: read circuit {cf.size} bytes, public file {pf.size} bytes, {tables.length} tables in {t1 - t0} ms; {← mem}"
  let some tags0 := Tags.find? tagName | IO.println s!"PROBE: no tags named {tagName}"; return 2
  let I : Witness.Inputs :=
    { tags := tags0, circuitFile := cf, publicFile := pf, count := 1, tables := tables, partition := partition,
      program := program }
  let tags := Witness.zkTags I
  let coins := Witness.zkCoins I
  let t2 ← IO.monoMsNow
  match ProvedScope.circuitOf tags cf tables with
  | .error e => IO.println s!"PROBE: the circuit does not parse: {e}"; return 1
  | .ok c =>
    let t3 ← IO.monoMsNow
    IO.println s!"PROBE: parse (ProvedScope.circuitOf) {t3 - t2} ms; k_log = {c.kLog}, g = {c.g}, templated: {c.tmpl.isSome}; {← mem}"
    match HmOut.loadPublic tags c pf with
    | .error e => IO.println s!"PROBE: the public file does not load: {e}"; return 1
    | .ok pop =>
      let t4 ← IO.monoMsNow
      let named := (pop.header.getObjVal? "public_inputs").toOption.isSome
      IO.println s!"PROBE: loadPublic {t4 - t3} ms; population n = {pop.n}; its header names public_inputs: {named}"
      let t5 ← IO.monoMsNow
      match Stmt.setupHiddenOf (.ok c) tags pf coins (some (emptyDraw pop.n)) partition program none with
      | .error e => IO.println s!"PROBE: the setup at the sample of no unit refuses: {e}"; return 1
      | .ok st =>
        let t6 ← IO.monoMsNow
        let z ← shapeTimed st
        IO.println s!"PROBE: setupHiddenOf at the sample of no unit on that parse {t6 - t5} ms; {z}; {shape st}; {← mem}"
        if args.contains "--whole" then
          let t7 ← IO.monoMsNow
          match Stmt.setupHidden tags cf pf coins tables (some (emptyDraw pop.n)) partition program none with
          | .error e => IO.println s!"PROBE: the whole setup from the bytes refuses: {e}"; return 1
          | .ok st' =>
            let t8 ← IO.monoMsNow
            let z' ← shapeTimed st'
            IO.println s!"PROBE: setupHidden from the bytes at the sample of no unit {t8 - t7} ms; {z'}; {shape st'}; {← mem}"
        return 0

end SetupProbe

def main (args : List String) : IO UInt32 := SetupProbe.run args
