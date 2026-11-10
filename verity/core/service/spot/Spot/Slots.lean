/-
The claims' slots over the verifier's circuit file (`Claims.Sampling.Partition` and `Deploy.checked`): the partition of
the claims' program of the circuit file (`Flock.Bridge.partOf` of `Placed.keyProg`, at the registration's strata), and
the slots of the proof blocks a sample checks.

* **Classes.** One, the population, in the registration's strata (`FilePart.sizes` of `Partition.strata`, which must
  be the units `0, 1, …` in order), consecutive runs of slots. Slot `u` is unit `u`: the circuit file's unit `u`, its
  positions those of `FilePart.block` as intervals, its rows' lowered gates then each source it owns. The constant, the
  zero, a public input, an unread input and a repeated source past its first column are in no proof block.
* **Expansion.** The circuit file is summarized once (`FilePart.ofCircuit`), and `layout` writes that summary out unit
  by unit, the summary shifted by the unit; `fileLayout` is that at the registration's strata, the layout `select`
  reads.
* **Checked.** `checked` lists the slot of every proof block the sample proves (`Select.select`'s proved units: its units
  and a work sample's closure), in slot order. It is the claims' reading of a sample; `Run.run` doesn't call it. No run
  opens an input row on its own: the sampled proof blocks' session (`HmRow.drawn`) opens only the rows its units read.
* **Registration.** `popOk` reads no file: the program's strata under the query (`Partition.strata`) are the units
  `0, …, n − 1` in order, each with its gates (`Partition.unitGates`).
* **JSON.** A proof block is `{"unit", "pus"}`: its unit and its positions as `[lo, hi]` intervals, its computed gates',
  then `[g, g + 1]` for each source it owns. `checked` is its `[slot, 0]` pairs, until a format version: the
  difftest's Python side (`one_stage.slots`) reads them.
-/
module

public import Spot.Select
public import Spot.FilePart
meta import Spot.Select
meta import Spot.FilePart

@[expose] public section

namespace Spot.Slots

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

/-- **Unit `u`'s proof block** (`FilePart.block` as intervals `[lo, hi)`): its rows' lowered gates, then each source it
owns. -/
def blockOf (f : FilePart.FilePart) (u : Nat) : ProofBlock :=
  { unit := some u
    gates := #[(f.inGates + 3 * f.m * u, f.inGates + 3 * f.m * (u + 1))] ++
      f.srcs.map fun b => (2 + u * f.B + b, 2 + u * f.B + b + 1) }

/-- **The claims' partition of a circuit file's key program** (its summary `f`, `FilePart.ofCircuit`) in strata of sizes
`ns`: one class, the strata consecutive runs of slots, slot `u` unit `u`'s proof block (`blockOf`). `f` is the summary at
`ns`'s sum of units. -/
def layout (f : FilePart.FilePart) (ns : Array Nat) : Layout :=
  let blocks := (Array.range f.n).map (blockOf f)
  let (pop, _) := ns.foldl (fun (acc : Array (Array ProofBlock) × Nat) k =>
    (acc.1.push (blocks.extract acc.2 (acc.2 + k)), acc.2 + k)) (#[], 0)
  { classes := #[pop] }

#guard (layout { n := 3, B := 2, m := 1, srcs := #[1] } #[2, 1]).classes.map (·.map (·.map fun b => (b.unit, b.gates))) ==
  #[#[#[(some 0, #[(8, 11), (3, 4)]), (some 1, #[(11, 14), (5, 6)])], #[(some 2, #[(14, 17), (7, 8)])]]]
#guard let f : FilePart.FilePart := { n := 3, B := 4, m := 2, srcs := #[0, 1, 3] }
  (List.range 3).all fun u =>
    ((blockOf f u).gates.toList.flatMap fun (lo, hi) => List.range' lo (hi - lo)) == (f.block u).toList

/-- **The claims' partition of the circuit file `c`** under the program `p` and `query`: `layout` of `c`'s key program,
summarized at the strata's sum of units (`FilePart.ofCircuit`), in the strata of `p` under `query` (`FilePart.sizes` of
`Partition.strata`). An error where the query doesn't apply or the strata are not the units in order. -/
def fileLayout (c : Flock.Circuit) (p : Program) (query : Json) : V Layout := do
  let some ns := FilePart.sizes (← Partition.strata p query)
    | throw "partition: the registration's strata, read stratum by stratum, are not its units in order"
  return layout (FilePart.ofCircuit c (ns.foldl (· + ·) 0)) ns

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

/-- **The checked slots** (`Deploy.checked`) of a sample that proves the units `proved`: every proof block among them, in
slot order. -/
def checked (l : Layout) (proved : Array Nat) : Array Nat :=
  (l.blocks.zipIdx.filter fun (b, _) => b.unit.any proved.contains).map (·.2)

/-- **The challenger's sample in the claims' slots**: the layout of the circuit file `c` in the strata of the program `p`
under `query` (`fileLayout`), and the slots of the units `Select.select`'s sample on the coins `src` proves (its units and
a work sample's closure). An error where either is. -/
def select (rule : Select.Rule) (c : Flock.Circuit) (p : Program) (query : Json)
    (table : Option Json) (cm : Option (Array (Int × Array Int))) (src : ByteArray) : V (Layout × Array Nat) := do
  let l ← fileLayout c p query
  let (_, proved, _) ← Select.select rule p query table cm src
  return (l, checked l proved)

/-- **The circuit file, parsed** as the deployed verifier parses it (`ProvedScope.circuitOf`, under the statement
`statement`, the one-stage audit's `OneStageZk.STATEMENT`): its bytes, and the lookup tables it may name by name. -/
def parseCircuit (statement : String) (bytes : ByteArray) (tables : List (String × ByteArray)) : V Flock.Circuit := do
  let some tags := Flock.Tags.find? statement | throw s!"the statement {statement} is not known"
  Flock.ProvedScope.circuitOf tags bytes tables

def ProofBlock.toJson (b : ProofBlock) : Json :=
  Json.mkObj [("unit", (b.unit.map Lean.toJson).getD .null),
    ("pus", Json.arr (b.gates.map fun (lo, hi) => Json.arr #[Lean.toJson lo, Lean.toJson hi]))]

def Layout.toJson (l : Layout) : Json := Json.arr (l.classes.map fun c => Json.arr (c.map fun s => Json.arr (s.map (·.toJson))))

end Spot.Slots
