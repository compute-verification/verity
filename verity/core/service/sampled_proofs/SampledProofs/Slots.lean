/-
The claims' slots over the verifier's partition (`Claims.Sampling.Partition` and `Deploy.checked`): the partition the
verifier evaluates, as the claims read it, and the proof units a draw checks, each as its replay unit's slot and its index
there.

* **Classes.** One, the population. Its strata are the query's (`Partition.strata`: one per template or Definition, in
  the order of its first unit), each stratum's replay units its units in ascending order. A unit's replay unit is its
  gates (`Partition.unitGates`: an instance's, a `Q_call` unit's interval, or the span of the gates a `Q_word` unit owns,
  which another unit's gates may interleave), then a one-gate proof unit for each input row it owns (`owned`), in gate
  order. A unit owns an input row, a gate of the root's input nodes, when it is the first unit that reads it: the least
  unit one of whose gates has it as an operand. An input row no unit reads is in no slot.
* **Slots.** A replay unit's slot is its index in stratum and replay-unit order (`Partition.rus`). Where no two strata's
  units interleave, unit `i`'s slot is `i`.
* **Checked.** `checked` lists the proof units of every replay unit the draw proves (`Select.select`'s proved units: its
  units and a work draw's closure), each as `(slot, index)`, in slot order: index `0` the unit's gates, then its input
  rows. No run opens an input row on its own: `Run.run` proves the drawn units (`Draw.proved`), whose session
  (`HmRow.drawn`) opens only the rows its units read, and registration opens none. So the run opens an input row through
  every drawn unit that reads it, while `checked` counts it once, with its owner: every row `checked` lists, the run
  opens. `checked` is the claims' reading of a draw; `Run.run` doesn't call it.
-/
import SampledProofs.Select

namespace SampledProofs.Slots

open Lean (Json)
open Flock (V)
open Flock.TemplateQuery (Desc)

/-- A replay unit: the population's unit it is, and its proof units' gates, each `[lo, hi)`: the unit's, then its input
rows'. -/
structure RU where
  unit : Nat
  pus : Array (Nat × Nat)
  deriving BEq, Inhabited, Repr

/-- The claims' partition over the verifier's: its classes, each its strata, each its replay units, in slot order. -/
structure Layout where
  classes : Array (Array (Array RU))
  deriving BEq, Inhabited, Repr

/-- The replay units, slot by slot. -/
def Layout.rus (l : Layout) : Array RU := l.classes.flatten.flatten

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

/-- **The reads of the input gates `isIn`**: `(h, g)` for each gate `h` and each operand `g` of it in `isIn`
(`Flock.Program.operands`), in gate order. A root node none of whose arguments resolves into `isIn` is not walked: a gate
inside it reads only its own node's gates and the gates its node's arguments resolve to. -/
def inputReads (P : Flock.Program.Prog) (isIn : Std.HashSet Nat) : V (Array (Nat × Nat)) := do
  let root := P.rootScope
  let body := (P.defs[P.root]!).body
  let mut out := #[]
  for k in [0:body.nodes.size] do
    let mut reads := false
    for a in body.nodes[k]!.args do
      for r in a.seq.toArray do
        if isIn.contains (← Flock.Program.resolve P root r) then reads := true
    if !reads then continue
    let mut stack : Array (Flock.Program.Scope × Nat × Nat × Nat) := #[(root, k, 0, k + 1)]
    for _ in [0:Flock.Steps.FUEL] do
      if stack.size == 0 then break
      let (sc, j, m, e) := stack.back!
      if j ≥ e then
        stack := stack.pop
        continue
      let b := (P.defs[sc.d]!).body
      let x := b.nodes[j]!
      if m ≥ x.members then
        stack := stack.set! (stack.size - 1) (sc, j + 1, 0, e)
        continue
      stack := stack.set! (stack.size - 1) (sc, j, m + 1, e)
      let callee := P.defs[x.fn]!
      let o := sc.off + b.offsets[j]!
      if !callee.prim then
        stack := stack.push (.child sc x.fn (o + m * callee.gates) j m, 0, 0, callee.body.nodes.size)
        continue
      for g in ← Flock.Program.operands P sc x m do
        if isIn.contains g then out := out.push (o + m, g)
    if stack.size > 0 then throw "a walk of more steps than this verifier follows"
  return out

/-- The first index of `rd`, ascending by gate, at or past gate `lo`, in `[a, b)`. -/
def firstAt (rd : Array (Nat × Nat)) (lo a b : Nat) : Nat :=
  if a < b then
    let m := (a + b) / 2
    if rd[m]!.1 < lo then firstAt rd lo (m + 1) b else firstAt rd lo a m
  else a
termination_by b - a

/-- **Each unit's input rows**, by unit, in gate order: the gates of the root's input nodes (`inputGates`) it owns, the
least unit `u` one of whose gates `gs[u]` reads it (`inputReads`). An input gate no unit's gates read is no unit's. -/
def owned (p : Program) (gs : Array (Nat × Nat)) : V (Array (Array Nat)) := do
  let ins := inputGates p.desc
  let rd ← inputReads p.prog (ins.foldl (·.insert ·) {})
  let mut owner : Std.HashMap Nat Nat := {}
  for u in [0:gs.size] do
    let (lo, hi) := gs[u]!
    for i in [firstAt rd lo 0 rd.size:rd.size] do
      let (h, g) := rd[i]!
      if h ≥ hi then break
      if !owner.contains g then owner := owner.insert g u
  return ins.foldl (fun acc g => match owner[g]? with
    | some u => acc.modify u (·.push g)
    | none => acc) (Array.replicate gs.size #[])

/-- The claims' partition over the verifier's own program `p` under `query` (`Partition.evaluate`): the population's
strata, each unit's replay unit its gates, then its input rows (`owned`). An error where the query doesn't apply. -/
def layout (p : Program) (query : Json) : V Layout := do
  let rs ← Partition.evaluate p query
  let gs ← Partition.unitGates p query
  let ins ← owned p gs
  let strata := Flock.Draw.strataOf (rs.map fun r => (r.template, r.first, r.count))
  let pop ← strata.mapM fun (_, ranges) =>
    (ranges.flatMap fun (lo, hi) => (Array.range (hi - lo)).map (lo + ·)).mapM fun u => do
      let some g := gs[u]? | throw s!"unit {u} has no gates"
      pure ({ unit := u, pus := #[g] ++ (ins.getD u #[]).map fun i => (i, i + 1) } : RU)
  return { classes := #[pop] }

/-- A layout's population: its first class's replay units, in slot order. -/
def popOf (l : Layout) : List RU := ((l.classes.toList.headD #[]).toList.map Array.toList).flatten

/-- **The population is the registration's `n` units, in order**: the layout of `p` under `query` is defined, and its
population is `n` replay units with slot `u` unit `u`. The run refuses a registration whose program and query fail it
(`Run.run`, `population-order`): the sampled audit's knowledge soundness reads slot `u` as the game's unit `u`. -/
def popOk (p : Program) (query : Json) (n : Nat) : Bool :=
  match layout p query with
  | .error _ => false
  | .ok l => decide ((popOf l).map (·.unit) = List.range n)

/-- **The checked proof units** (`Deploy.checked`) of a draw that proves the units `proved`: each proof unit of every
replay unit among them, as `(slot, index)`, in slot order. -/
def checked (l : Layout) (proved : Array Nat) : Array (Nat × Nat) :=
  (l.rus.zipIdx.filter fun (r, _) => proved.contains r.unit).flatMap fun (r, v) =>
    (Array.range r.pus.size).map (v, ·)

/-- **The challenger's draw in the claims' slots**: the layout of `p` under `query`, and the proof units of the units
`Select.select`'s draw on the coins `src` proves. An error where either is. -/
def select (rule : Select.Rule) (p : Program) (query : Json) (table : Option Json) (cm : Option (Array (Int × Array Int)))
    (src : ByteArray) : V (Layout × Array (Nat × Nat)) := do
  let l ← layout p query
  let (_, proved, _) ← Select.select rule p query table cm src
  return (l, checked l proved)

def RU.toJson (r : RU) : Json :=
  Json.mkObj [("unit", Lean.toJson r.unit),
    ("pus", Json.arr (r.pus.map fun (lo, hi) => Json.arr #[Lean.toJson lo, Lean.toJson hi]))]

def Layout.toJson (l : Layout) : Json := Json.arr (l.classes.map fun c => Json.arr (c.map fun s => Json.arr (s.map (·.toJson))))

end SampledProofs.Slots
