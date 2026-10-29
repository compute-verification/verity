/-
The claims' slots over the verifier's partition (`Claims.Sampling.Partition` and `Deploy.checked`): the partition the
verifier evaluates, as the claims read it, and the proof units a draw checks, each as its replay unit's slot and its index
there.

* **Classes.** Two, the population and then the inputs. The population's strata are the query's (`Partition.strata`:
  one per template or Definition, in the order of its first unit), each stratum's replay units its units in ascending
  order, each unit one proof unit, its gates (`Partition.unitGates`): an instance's, a `Q_call` unit's interval, or the
  span of the gates a `Q_word` unit owns, which another unit's gates may interleave. The input class is one stratum (none
  when the root has no input) of every gate of the root's input nodes, in gate order, each a replay unit of one proof
  unit; the claims' design reads it `all`, as `partition_object`'s input unit is never drawn. No run opens an input
  row on its own: `Run.run` proves the drawn units (`Draw.proved`), whose session (`HmRow.drawn`) opens only the rows
  its units read, and registration opens none. So an input row is opened only through a drawn unit that reads it.
* **Slots.** A replay unit's slot is its index in class, stratum and replay-unit order (`Partition.rus`). Where no two
  strata's units interleave, unit `i`'s slot is `i`.
* **Checked.** `checked` lists the proof units of every replay unit the draw proves (`Select.select`'s proved units: its
  units and a work draw's closure) and of every input slot, each as `(slot, index)`, in slot order. The rules a stream
  runs make every replay unit one proof unit, so each index is `0`. It is the claims' reading of a draw; `Run.run`
  doesn't call it, and the session proves no input slot of a unit outside the draw.
-/
import SampledProofs.Select

namespace SampledProofs.Slots

open Lean (Json)
open Flock (V)
open Flock.TemplateQuery (Desc)

/-- A replay unit: the population's unit it is (`none` for an input gate), and its proof units' gates, each `[lo, hi)`. -/
structure RU where
  unit : Option Nat
  pus : Array (Nat × Nat)
  deriving BEq, Inhabited, Repr

/-- The claims' partition over the verifier's: its classes, each its strata, each its replay units, in slot order. -/
structure Layout where
  classes : Array (Array (Array RU))
  deriving BEq, Inhabited, Repr

/-- The replay units, slot by slot. -/
def Layout.rus (l : Layout) : Array RU := l.classes.flatten.flatten

/-- Unit `u`'s slot: the replay unit that is it. -/
def Layout.slotOf (l : Layout) (u : Nat) : Option Nat := l.rus.findIdx? (·.unit == some u)

/-- The gates of the root's input nodes (`TemplateQuery.isInput`), in gate order. -/
def inputGates (d : Desc) : Array Nat := Id.run do
  let some root := d.def? d.root | return #[]
  let mut out := #[]
  let mut at_ := 0
  for x in root.nodes do
    let tid := d.callees.getD x.fn ""
    let n := x.members * d.gates tid
    if Flock.TemplateQuery.isInput tid then out := out ++ (Array.range n).map (at_ + ·)
    at_ := at_ + n
  return out

/-- The claims' partition over the verifier's own program `p` under `query` (`Partition.evaluate`): the population's
strata, then the input class. An error where the query doesn't apply. -/
def layout (p : Program) (query : Json) : V Layout := do
  let rs ← Partition.evaluate p query
  let gs ← Partition.unitGates p query
  let strata := Flock.Draw.strataOf (rs.map fun r => (r.template, r.first, r.count))
  let pop ← strata.mapM fun (_, ranges) =>
    (ranges.flatMap fun (lo, hi) => (Array.range (hi - lo)).map (lo + ·)).mapM fun u => do
      let some g := gs[u]? | throw s!"unit {u} has no gates"
      pure ({ unit := some u, pus := #[g] } : RU)
  let inputs := (inputGates p.desc).map fun g => ({ unit := none, pus := #[(g, g + 1)] } : RU)
  return { classes := #[pop, if inputs.isEmpty then #[] else #[inputs]] }

/-- A layout's population: its first class's replay units, in slot order. -/
def popOf (l : Layout) : List RU := ((l.classes.toList.headD #[]).toList.map Array.toList).flatten

/-- **The population is the registration's `n` units, in order**: the layout of `p` under `query` is defined, and its
population is `n` replay units with slot `u` unit `u`. The run refuses a registration whose program and query fail it
(`Run.run`, `population-order`): the sampled audit's knowledge soundness reads slot `u` as the game's unit `u`. -/
def popOk (p : Program) (query : Json) (n : Nat) : Bool :=
  match layout p query with
  | .error _ => false
  | .ok l => decide ((popOf l).map (·.unit) = (List.range n).map some)

/-- **The checked proof units** (`Deploy.checked`) of a draw that proves the units `proved`: each proof unit of every
replay unit among them and of every input slot, as `(slot, index)`, in slot order. An input slot listed here is
opened only where a drawn unit reads its row. -/
def checked (l : Layout) (proved : Array Nat) : Array (Nat × Nat) :=
  (l.rus.zipIdx.filter fun (r, _) => r.unit.all proved.contains).flatMap fun (r, v) =>
    (Array.range r.pus.size).map (v, ·)

/-- **The challenger's draw in the claims' slots**: the layout of `p` under `query`, and the proof units of the units
`Select.select`'s draw on the coins `src` proves. An error where either is. -/
def select (rule : Select.Rule) (p : Program) (query : Json) (table : Option Json) (cm : Option (Array (Int × Array Int)))
    (src : ByteArray) : V (Layout × Array (Nat × Nat)) := do
  let l ← layout p query
  let (_, proved, _) ← Select.select rule p query table cm src
  return (l, checked l proved)

def RU.toJson (r : RU) : Json :=
  Json.mkObj [("unit", (r.unit.map Lean.toJson).getD .null),
    ("pus", Json.arr (r.pus.map fun (lo, hi) => Json.arr #[Lean.toJson lo, Lean.toJson hi]))]

def Layout.toJson (l : Layout) : Json := Json.arr (l.classes.map fun c => Json.arr (c.map fun s => Json.arr (s.map (·.toJson))))

end SampledProofs.Slots
