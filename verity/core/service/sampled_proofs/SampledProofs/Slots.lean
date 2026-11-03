/-
The claims' slots over the verifier's circuit files (`Claims.Sampling.Partition` and `Deploy.checked`): the partition of
the claims' program of each circuit file (`Flock.Bridge.partOf` of `Placed.keyProg`), and the slots of the proof blocks a
draw checks.

* **Classes.** One, the population. Each circuit file, with its public file, is one stratum, the strata in the order of
  their first unit. A file's proof blocks are its instances in order (the public file's `units.indices`), instance `i`
  unit `units[i]`, its positions those of `FileUnits.block`: its computed gates, then each source it owns. A source no
  computed row reads is in no proof block, nor are the constant and the zero.
* **Positions.** Each file's are its own program's (`FileUnits.size`), the files' laid end to end in stratum order.
* **Slots.** A proof block's slot is its index in stratum and block order (`Layout.blocks`). Where the files' units run
  `0, 1, …` in stratum order, unit `u`'s slot is `u`.
* **Checked.** `checked` lists the slot of every proof block the draw proves (`Select.select`'s proved units: its units
  and a work draw's closure), in slot order. It is the claims' reading of a draw; `Run.run` doesn't call it.
* **Registration.** `popOk` reads no file: the program's strata under the query (`Partition.strata`) are the units
  `0, …, n − 1` in order, each with its gates (`Partition.unitGates`).
* **JSON.** A proof block is `{"unit", "pus"}`: its unit and its positions as `[lo, hi]` intervals, its computed gates',
  then `[g, g + 1]` for each source it owns. `checked` is its `[slot, 0]` pairs, until a format version: the
  difftest's Python side (`one_stage.slots`) reads them.
-/
import SampledProofs.Select
import SampledProofs.FileUnits

namespace SampledProofs.Slots

open Lean (Json)
open Flock (V)

/-- A proof block: the population's unit it is, and its positions as intervals `[lo, hi)`: its computed gates', then one
per source it owns. -/
structure ProofBlock where
  unit : Option Nat
  gates : Array (Nat × Nat)
  deriving BEq, Inhabited, Repr

/-- The claims' partition over the verifier's: its classes, each its strata, each its proof blocks, in slot order. -/
structure Layout where
  classes : Array (Array (Array ProofBlock))
  deriving BEq, Inhabited, Repr

/-- The proof blocks, slot by slot. -/
def Layout.blocks (l : Layout) : Array ProofBlock := l.classes.flatten.flatten

/-- Unit `u`'s slot: the proof block that is it. -/
def Layout.slotOf (l : Layout) (u : Nat) : Option Nat := l.blocks.findIdx? (·.unit == some u)

/-- **The claims' partition over the circuit files `files`** (each a circuit and its public file): one class, a stratum
per file in the order of its first unit, each instance's proof block at its file's offset. An error where a file names
no unit or its units aren't its instances'. -/
def layout (files : Array (Flock.Circuit × Flock.Public)) : V Layout := do
  let fs ← files.mapM fun (c, pub) => do
    let f ← FileUnits.ofFile c pub
    let some u := f.units[0]? | throw "a circuit file's public file names no unit"
    pure (u, f)
  let fs := fs.qsort (·.1 < ·.1)
  let (pop, _) := fs.foldl (fun (acc : Array (Array ProofBlock) × Nat) (_, f) =>
    (acc.1.push ((Array.range f.units.size).map fun i =>
      { unit := some f.units[i]!, gates := (f.block i).map fun (lo, hi) => (acc.2 + lo, acc.2 + hi) }),
     acc.2 + f.size)) (#[], 0)
  return { classes := #[pop] }

/-- The population's units under `query`, stratum by stratum (`Partition.strata`), each with its gates
(`Partition.unitGates`). An error where the query doesn't apply. -/
def popUnits (p : Program) (query : Json) : V (Array Nat) := do
  let rs ← Partition.evaluate p query
  let gs ← Partition.unitGates p query
  let us := (Flock.Draw.strataOf (rs.map fun r => (r.template, r.first, r.count))).flatMap fun (_, ranges) =>
    ranges.flatMap fun (lo, hi) => (Array.range (hi - lo)).map (lo + ·)
  if let some u := us.find? (gs.size ≤ ·) then throw s!"unit {u} has no gates"
  return us

/-- **The population is the registration's `n` units, in order**: the units of `p` under `query` (`popUnits`) are
`0, …, n − 1`, stratum by stratum. The run refuses a registration whose program and query fail it (`Run.run`,
`population-order`): the sampled audit's knowledge soundness reads slot `u` as the game's unit `u`. -/
def popOk (p : Program) (query : Json) (n : Nat) : Bool :=
  match popUnits p query with
  | .error _ => false
  | .ok us => decide (us.toList = List.range n)

/-- **The checked slots** (`Deploy.checked`) of a draw that proves the units `proved`: every proof block among them, in
slot order. -/
def checked (l : Layout) (proved : Array Nat) : Array Nat :=
  (l.blocks.zipIdx.filter fun (b, _) => b.unit.any proved.contains).map (·.2)

/-- **The challenger's draw in the claims' slots**: the layout of the circuit files `files`, and the slots of the units
`Select.select`'s draw on the program `p` under `query` and the coins `src` proves. An error where either is. -/
def select (rule : Select.Rule) (files : Array (Flock.Circuit × Flock.Public)) (p : Program) (query : Json)
    (table : Option Json) (cm : Option (Array (Int × Array Int))) (src : ByteArray) : V (Layout × Array Nat) := do
  let l ← layout files
  let (_, proved, _) ← Select.select rule p query table cm src
  return (l, checked l proved)

/-- **The circuit files, parsed** as the deployed verifier parses them (`ProvedScope.circuitOf`, `publicOf`, under the
one-stage statement `OneStageZk.STATEMENT`): each circuit file's and public file's bytes, and the lookup tables they
may name by name. -/
def parseFiles (statement : String) (bytes : Array (ByteArray × ByteArray)) (tables : List (String × ByteArray)) :
    V (Array (Flock.Circuit × Flock.Public)) := do
  let some tags := Flock.Tags.find? statement | throw s!"the statement {statement} is not known"
  bytes.mapM fun (cb, pb) => do
    let c ← Flock.ProvedScope.circuitOf tags cb tables
    pure (c, ← Flock.ProvedScope.publicOf tags c pb)

def ProofBlock.toJson (b : ProofBlock) : Json :=
  Json.mkObj [("unit", (b.unit.map Lean.toJson).getD .null),
    ("pus", Json.arr (b.gates.map fun (lo, hi) => Json.arr #[Lean.toJson lo, Lean.toJson hi]))]

def Layout.toJson (l : Layout) : Json := Json.arr (l.classes.map fun c => Json.arr (c.map fun s => Json.arr (s.map (·.toJson))))

end SampledProofs.Slots
