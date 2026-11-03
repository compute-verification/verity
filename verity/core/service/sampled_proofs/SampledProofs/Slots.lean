/-
The claims' slots over the verifier's partition (`Claims.Sampling.Partition` and `Deploy.checked`): the partition the
verifier evaluates, as the claims read it, and the slots of the proof blocks a draw checks.

* **Classes.** One, the population. Its strata are the query's (`Partition.strata`: one per template or Definition, in the
  order of its first unit), each stratum's proof blocks its units in ascending order. Unit `u`'s proof block is its gates
  (`Partition.unitGates`: an instance's activation, a `Q_call` unit's interval, or exactly the gates a `Q_word` unit owns,
  which another unit's gates may interleave), then each input it owns (`Partition.unitInputs`): a gate of the root's input
  nodes whose first reader, in unit order, is `u`, in gate order. An input no unit reads is in no proof block. An input row
  is checked with the unit that owns it, as the deployed verifier opens it: `Run.run` proves the drawn units
  (`Draw.proved`), whose session (`HmRow.drawn`) opens only the rows its units read, and registration opens none.
* **Slots.** A proof block's slot is its index in stratum and block order (`Layout.blocks`). Where no two strata's units
  interleave, unit `u`'s slot is `u`.
* **Checked.** `checked` lists the slot of every proof block the draw proves (`Select.select`'s proved units: its units
  and a work draw's closure), in slot order. It is the claims' reading of a draw; `Run.run` doesn't call it.
* **JSON.** A proof block is `{"unit", "pus"}`: its unit and its gates as `[lo, hi]` intervals, the unit's own, ascending,
  then `[g, g + 1]` for each input it owns. `checked` is its `[slot, 0]` pairs, until a format version: the difftest's
  Python side reads them.
-/
import SampledProofs.Select

namespace SampledProofs.Slots

open Lean (Json)
open Flock (V)

/-- A proof block: the population's unit it is, and its gates as intervals `[lo, hi)`: the unit's own, then one per input
it owns. -/
structure ProofBlock where
  unit : Nat
  gates : Array (Nat × Nat)
  deriving BEq, Inhabited, Repr

/-- The claims' partition over the verifier's: its classes, each its strata, each its proof blocks, in slot order. -/
structure Layout where
  classes : Array (Array (Array ProofBlock))
  deriving BEq, Inhabited, Repr

/-- The proof blocks, slot by slot. -/
def Layout.blocks (l : Layout) : Array ProofBlock := l.classes.flatten.flatten

/-- Unit `u`'s slot: the proof block that is it. -/
def Layout.slotOf (l : Layout) (u : Nat) : Option Nat := l.blocks.findIdx? (·.unit == u)

/-- The claims' partition over the verifier's own program `p` under `query` (`Partition.evaluate`): one class, the
population's strata. An error where the query doesn't apply. -/
def layout (p : Program) (query : Json) : V Layout := do
  let rs ← Partition.evaluate p query
  let as ← Partition.acts p query
  let gs := Partition.gatesOf p.prog as
  let ins ← Partition.unitInputs p as
  let strata := Flock.Draw.strataOf (rs.map fun r => (r.template, r.first, r.count))
  let pop ← strata.mapM fun (_, ranges) =>
    (ranges.flatMap fun (lo, hi) => (Array.range (hi - lo)).map (lo + ·)).mapM fun u => do
      let some g := gs[u]? | throw s!"unit {u} has no gates"
      pure ({ unit := u, gates := g ++ (ins.getD u #[]).map fun x => (x, x + 1) } : ProofBlock)
  return { classes := #[pop] }

/-- A layout's population: its first class's proof blocks, in slot order. -/
def popOf (l : Layout) : List ProofBlock := ((l.classes.toList.headD #[]).toList.map Array.toList).flatten

/-- **The population is the registration's `n` units, in order**: the layout of `p` under `query` is defined, and its
population is `n` proof blocks with slot `u` unit `u`. The run refuses a registration whose program and query fail it
(`Run.run`, `population-order`): the sampled audit's knowledge soundness reads slot `u` as the game's unit `u`. -/
def popOk (p : Program) (query : Json) (n : Nat) : Bool :=
  match layout p query with
  | .error _ => false
  | .ok l => decide ((popOf l).map (·.unit) = List.range n)

/-- **The checked slots** (`Deploy.checked`) of a draw that proves the units `proved`: every proof block among them, in
slot order. -/
def checked (l : Layout) (proved : Array Nat) : Array Nat :=
  (l.blocks.zipIdx.filter fun (b, _) => proved.contains b.unit).map (·.2)

/-- **The challenger's draw in the claims' slots**: the layout of `p` under `query`, and the slots of the units
`Select.select`'s draw on the coins `src` proves. An error where either is. -/
def select (rule : Select.Rule) (p : Program) (query : Json) (table : Option Json) (cm : Option (Array (Int × Array Int)))
    (src : ByteArray) : V (Layout × Array Nat) := do
  let l ← layout p query
  let (_, proved, _) ← Select.select rule p query table cm src
  return (l, checked l proved)

def ProofBlock.toJson (b : ProofBlock) : Json :=
  Json.mkObj [("unit", Lean.toJson b.unit),
    ("pus", Json.arr (b.gates.map fun (lo, hi) => Json.arr #[Lean.toJson lo, Lean.toJson hi]))]

def Layout.toJson (l : Layout) : Json := Json.arr (l.classes.map fun c => Json.arr (c.map fun s => Json.arr (s.map (·.toJson))))

end SampledProofs.Slots
