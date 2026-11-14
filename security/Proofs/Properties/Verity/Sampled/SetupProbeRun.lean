import Spot.OneStageZk
import Flock.WitnessZk
import Flock.HmRow
import Flock.Zk

/-! # Scratch probe, not for landing: a statement's setup at the sample of no unit, run on a fixture

`SetupProbe.setupHidden_sibling` says a statement that set up at a sample sets up at the sample of no unit. This file
runs the verifier's own setup both ways on the files of the form-refusals fixture's `honest` case (a directory the
environment names, `PROBE_FIXTURE`), and prints what each returns. It proves nothing and lands nowhere. -/

open Lean (Json)

namespace Verity.Sampled.SetupProbeRun

/-- The sample of no unit over a population of `N` units. -/
def emptyDraw (N : Nat) : Json := Flock.Draw.UnitDraw.toJson ⟨.subset 0, N, #[], #[]⟩

/-- The sample of unit 0 alone over a population of `N` units. -/
def oneDraw (N : Nat) : Json := Flock.Draw.UnitDraw.toJson ⟨.subset 1, N, #[0], #[]⟩

def verdict (r : Flock.V Unit) : String :=
  match r with
  | .ok _ => "ok"
  | .error e => s!"refused ({e})"

def shape (st : Flock.Stmt) : String :=
  s!"n = {st.n}, blocks = {st.blocksReal}, nbl = {st.nbl}, m = {st.m}, k_log = {st.c.kLog}, g = {st.c.g}, " ++
    s!"regions = {st.regions.size}, shapeOk: {verdict (Flock.Zk.shapeOk st)}"

def run (dir : String) : IO Unit := do
  let cf ← IO.FS.readBinFile s!"{dir}/honest/stage/circuit.txt"
  let pf ← IO.FS.readBinFile s!"{dir}/honest/pub-4.bin"
  let partS ← IO.FS.readFile s!"{dir}/honest/stage/partition.json"
  let prog ← IO.FS.readBinFile s!"{dir}/honest/stage/program.json"
  IO.println s!"PROBE: files: circuit {cf.size} bytes, public file {pf.size} bytes, program {prog.size} bytes"
  let some tags0 := Flock.Tags.find? Spot.OneStageZk.STATEMENT | IO.println "PROBE: the statement's tags are not known"
  let .ok part := Flock.JsonParse.parse partS | IO.println "PROBE: the partition object does not parse"
  let I : Flock.Witness.Inputs :=
    { tags := tags0, circuitFile := cf, publicFile := pf, count := 1, tables := [], partition := some part,
      program := some prog }
  let tags := Flock.Witness.zkTags I
  let coins := Flock.Witness.zkCoins I
  match Flock.Stmt.setupHidden tags cf pf coins [] (some (oneDraw 4)) I.partition I.program none with
  | .error e => IO.println s!"PROBE: the setup at one unit refuses: {e}"
  | .ok st₁ =>
    IO.println s!"PROBE: the setup at one unit: {shape st₁}"
    match Flock.HmOut.loadPublic tags st₁.c pf with
    | .error e => IO.println s!"PROBE: the public file does not load: {e}"
    | .ok pop =>
      let named := (pop.header.getObjVal? "public_inputs").toOption.isSome
      let held := verdict (Flock.HmOut.checkPublic st₁.c pop)
      IO.println s!"PROBE: the population: n = {pop.n}; its header names public_inputs: {named}; checkPublic on it: {held}"
      match Flock.Stmt.setupHidden tags cf pf coins [] (some (emptyDraw pop.n)) I.partition I.program none with
      | .error e => IO.println s!"PROBE: the setup at the sample of no unit refuses: {e}"
      | .ok st₂ =>
        let same := st₂.c.sha.data == st₁.c.sha.data
        IO.println s!"PROBE: the setup at the sample of no unit: {shape st₂}; its circuit is the sibling's: {same}"

#eval show IO Unit from do
  match ← IO.getEnv "PROBE_FIXTURE" with
  | none => IO.println "PROBE: PROBE_FIXTURE is unset, nothing run"
  | some dir => run dir

end Verity.Sampled.SetupProbeRun
