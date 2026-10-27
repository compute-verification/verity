import Definitions.Accounting.Gates.Registry
import Properties.Accounting.Compose

/-!
# The declared circuit of a compiled Program (`Properties.Compose.dagOf`, `Properties.Compose.declaredOf`)

`Accounting.Compose.Declared` is the composed credit's circuit (`Definitions.Accounting.Guarantees.Compose`): its gates,
its call sites and rounds, its coins, and each gate's proof unit. This module builds one from the verifier's own
program (`SampledProofs.Program`), so that a statement about `Declared` is about the circuit the service proves:

* **`dagOf`, the gates**: root gate `i` of the program, in canonical order (the index space of the verifier's partition,
  `SampledProofs.Slots.layout`), is the code's own gate locator's (`Flock.Program.Prog.gate`): its primitive and its
  operands. Its operation is its primitive's class (`Accounting.Gates.classOf`), an input its slot `i` bound as its id
  says (`Input{n}_v1` at run time, `RegisteredInput{n}_v1` at registration, as the decoder's `Defn.binding`
  reads it), its declared interval its output word's two's-complement range (`Gates.signed`), and it is committed where
  it is one of the root's returned gates (`retGates`). Compose reads no gate's value, so a constant's, an affine map's
  and a budgeted primitive's function are left zero; and every gate's site is `other`, so no gate is a served word yet:
  reading the sites (an `S8Dot_v1` step's, an H1 + fx1 receipt word's) and the ports (a coin input's) is the next step.
  It refuses a primitive with no class, an input whose class and the decoder's binding disagree, a primitive of more
  than one output word, and a gate that reads a later one.
* **`declaredOf`, the units**: `dagOf`'s gates under a given schedule (`Schedule`: call sites, rounds, coins, commit
  coins and seed preimages), each gate the unit of the population replay unit whose span holds it (`unitOf`), with the
  population's `N_p` units (`SampledProofs.Partition.units`). It refuses where a non-input gate lies in no population
  unit's span, an input slot's gate is no input, or a population unit is past `N_p` (`unitsChecked`).

`DagOfProgram` and `DeclaredOfProgram` state that what they accept is that. `LayoutUnits` is
`Properties.Compose.UnitsOfLayout`'s units and draw conjuncts, word for word, without its commit-coin and seed
conjuncts: those are PoUW's seeded schedule, under which nothing is in `Accounting.Compose.Scope`
(`Properties.Served.SeededScheduleNotLive`), while the schedule `declaredOf` is given is the live one
`Accounting.Served.Tie` states.
-/

open Lean (Json)
open Pouw.RelativeSecurity (prob)

namespace Properties.Compose

open Accounting.Compose (Declared)

/-- **An input primitive's binding**, as the decoder's `Defn.binding` names it (`Flock.Program.inputId?`): at run time
for `run`, at registration for `registration`; none for any other primitive. -/
def bindingOf (b : Option String) : Option Accounting.Gates.Binding :=
  if b = some Flock.Program.RUN then some .run
  else if b = some Flock.Program.REGISTRATION then some .registered
  else none

/-- An operation's input binding: `some b` for an input bound at `b`, none for any other. -/
def opBinding {ι : Type} : Accounting.Gates.Op ι → Option Accounting.Gates.Binding
  | .input b _ => some b
  | _ => none

/-- **A class's operation** at gate `i`: an input in slot `i`, an s8-family step or a gather with its class's
parameters, and a constant, an affine map or a budgeted primitive with its function zeroed (Compose reads no value). -/
def opOf (i : ℕ) : Accounting.Gates.Class → Accounting.Gates.Op ℕ
  | .input b => .input b i
  | .const => .const 0
  | .affine => .affine 0 []
  | .family a b k sat => .mma a b k sat
  | .gather low fill => .gather low fill
  | .budgeted cls => .budgeted cls fun _ => 0

/-- **Root gate `i`'s gate**, of primitive `d` with operands `ops` among the root's returned gates `rs`: its class's
operation (`opOf`), its operands, its output word's range `[−2^(w−1), 2^(w−1))`, engine step `0`, its primitive's id,
site `other`, committed where `rs` holds it. It refuses a primitive with no class (`classOf`), a coin class (which
no id lists: a coin is its port's), an input class whose binding isn't the decoder's (`bindingOf`), a primitive of
more than one output word, and an operand at `i` or later. -/
def gateOf (i : ℕ) (d : Flock.Program.Defn) (ops : Array ℕ) (rs : List ℕ) :
    Except String (Accounting.Gates.Gate ℕ) :=
  match Accounting.Gates.classOf d.id, d.ret with
  | some c, .value w =>
    if c = .input .coin then .error s!"gate {i}: a coin class for {d.id}"
    else if opBinding (opOf i c) ≠ bindingOf d.binding then .error s!"gate {i}: {d.id}'s class and binding disagree"
    else if !ops.toList.all (· < i) then .error s!"gate {i} reads a later gate"
    else .ok
      { op := opOf i c, args := ops.toList, lo := -2 ^ (w - 1), hi := 2 ^ (w - 1) - 1, step := 0, prim := d.id,
        site := .other, commit := rs.contains i }
  | none, _ => .error s!"gate {i}: primitive {d.id} has no class"
  | some _, _ => .error s!"gate {i}: primitive {d.id} returns more than one word"

/-- **The root's returned gates**: each leaf of the root body's `ret`, resolved to its gate (`Flock.Program.resolve`). -/
def retGates (P : Flock.Program.Prog) : Except String (List ℕ) :=
  let ret := (P.defs[P.root]!).body.ret
  (List.range ret.count).mapM fun j => Flock.Program.resolve P P.rootScope (ret.get j)

/-- Root gates `i, i + 1, …`, `r` of them, after `acc` (`gateOf` of each gate's `Prog.gate`). -/
def gatesFrom (P : Flock.Program.Prog) (rs : List ℕ) :
    ℕ → ℕ → Array (Accounting.Gates.Gate ℕ) → Except String (Array (Accounting.Gates.Gate ℕ))
  | _, 0, acc => .ok acc
  | i, r + 1, acc => match P.gate i with
    | .error e => .error e
    | .ok (fn, ops, _) => match gateOf i P.defs[fn]! ops rs with
      | .error e => .error e
      | .ok g => gatesFrom P rs (i + 1) r (acc.push g)

/-- **The gates of a compiled Program**: its root gates in canonical order, each `gateOf` of its primitive and operands
(`Flock.Program.Prog.gate`), the root's returned gates committed. -/
def dagOf (p : SampledProofs.Program) : Except String (Accounting.Gates.Dag ℕ) := do
  let rs ← retGates p.prog
  let gs ← gatesFrom p.prog rs 0 (p.prog.defs[p.prog.root]!).gates #[]
  pure gs.toList

/-- **`dagOf`'s gates are the program's**: one a root gate, in their primitives' classes (`Gates.Classed`), each reading
earlier gates (`Gates.Topological`), and gate `i` the one the code's locator gives (`Prog.gate`): its operands, its
primitive's id, its primitive's binding as the decoder reads it, committed where it is a returned gate of the root. -/
def DagOfProgram : Prop :=
  ∀ (p : SampledProofs.Program) (C : Accounting.Gates.Dag ℕ), dagOf p = .ok C →
    C.length = (p.prog.defs[p.prog.root]!).gates ∧ Accounting.Gates.Classed C ∧ Accounting.Gates.Topological C ∧
      ∃ rs, retGates p.prog = .ok rs ∧ ∀ i (h : i < C.length), ∃ fn sc,
        p.prog.gate i = .ok (fn, C[i].args.toArray, sc) ∧ C[i].prim = (p.prog.defs[fn]!).id ∧
          opBinding C[i].op = bindingOf (p.prog.defs[fn]!).binding ∧ C[i].commit = rs.contains i

/-- **A schedule**: what `Declared` carries beside its gates and units, its call sites, rounds, coins, commit coins and
seed preimages. -/
structure Schedule where
  calls : ℕ
  call : ℕ → ℕ
  rounds : ℕ
  round : ℕ → ℕ
  coinMode : ℕ → Accounting.Compose.CoinMode
  commitRound : ℕ → ℕ
  seedNames : ℕ → ℕ → Bool

/-- The declared circuit of gates `dag` on schedule `s`, with `Np` units and gate `i` in unit `unit i`. -/
def Schedule.declared (s : Schedule) (dag : Accounting.Gates.Dag ℕ) (Np : ℕ) (unit : ℕ → ℕ) : Declared ℕ :=
  { dag, calls := s.calls, call := s.call, rounds := s.rounds, round := s.round, coinMode := s.coinMode,
    commitRound := s.commitRound, seedNames := s.seedNames, units := Np, unit }

/-- **The population unit whose gates hold gate `i`** in layout `l`: the first replay unit in slot order of a
population unit with a proof unit `[lo, hi)` holding `i`, and none where no population unit's does. -/
def unitOf (l : SampledProofs.Slots.Layout) (i : ℕ) : Option ℕ :=
  l.rus.toList.findSome? fun r => r.unit.filter fun _ => r.pus.toList.any fun g => decide (g.1 ≤ i ∧ i < g.2)

/-- **`declaredOf`'s checks**: every non-input gate of `Cd` lies in a population unit's span, every population unit is
below `Np`, and every input slot's gate is an input of `Cd`. -/
def unitsChecked {ι : Type} (Cd : Declared ι) (l : SampledProofs.Slots.Layout) (Np : ℕ) : Bool :=
  (List.range Cd.dag.length).all (fun i => (Cd.binding i).isSome || (unitOf l i).isSome) &&
    l.rus.toList.all fun r => match r.unit with
      | some u => decide (u < Np)
      | none => r.pus.toList.all fun g => decide (g.1 < Cd.dag.length) && (Cd.binding g.1).isSome

/-- **The declared circuit of a compiled Program** on schedule `s`: `dagOf`'s gates, each in the unit `unitOf` gives
it in the verifier's partition of program `p` under `query` (`0` for an input, which no population unit holds), with
the population's `N_p` units, where `Slots.layout` and `Partition.units` accept and `unitsChecked` holds; an error
otherwise. -/
def declaredOf (p : SampledProofs.Program) (query : Json) (s : Schedule) : Except String (Declared ℕ) := do
  let dag ← dagOf p
  let l ← SampledProofs.Slots.layout p query
  let Np ← SampledProofs.Partition.units p query
  let Cd := s.declared dag Np fun i => (unitOf l i).getD 0
  if unitsChecked Cd l Np then pure Cd
  else throw "the circuit's gates are not the partition's: a non-input gate in no unit, or an input slot no input"

/-- **`Cd`'s units and draw are the service's** on program `p` under `query` (`UnitsOfLayout`'s first three
conjuncts and its last, word for word), wherever the layout is defined: each of `Cd`'s non-input gates lies in a proof
unit of the replay unit that is its own unit, each input slot's gate is an input of `Cd`, each population unit is one
of `Cd`'s `units`; and where the population is `N_p` units, the code's `subset:K_d` draw from `L` uniform bytes is
accepted and misses a set `U` of units with probability at most the model's draw's (`Pouw.Window.subsetDraw K_d` of a
uniform permutation of `Fin N_p`). -/
def LayoutUnits {ι : Type} (Cd : Declared ι) (Np Kd : ℕ) (p : SampledProofs.Program) (query : Json) : Prop :=
  ∀ l, SampledProofs.Slots.layout p query = .ok l →
    (∀ i, i < Cd.dag.length → Cd.binding i = none →
      ∃ r ∈ l.rus.toList, r.unit = some (Cd.unit i) ∧ ∃ g ∈ r.pus.toList, g.1 ≤ i ∧ i < g.2) ∧
    (∀ r ∈ l.rus.toList, r.unit = none → ∀ g ∈ r.pus.toList, g.1 < Cd.dag.length ∧ (Cd.binding g.1).isSome) ∧
    (∀ r ∈ l.rus.toList, ∀ u, r.unit = some u → u < Cd.units) ∧
    (SampledProofs.Partition.units p query = .ok Np →
      ∀ (cm : Option (Array (Int × Array Int))) (L : ℕ) (U : Finset ℕ),
        prob (fun src : Fin L → Fin 256 => ∃ r,
            SampledProofs.Select.select (.law ("subset:" ++ toString Kd)) p query none cm
              (Properties.SampledProofs.bytesOf src) = .ok r ∧ ∀ u ∈ U, u ∉ r.2.1) ≤
          prob (fun β : Equiv.Perm (Fin Np) => ∀ t ∈ Pouw.Window.subsetDraw Kd β, t.val ∉ U))

/-- **A declared circuit `declaredOf` accepts is the program's**: its gates are `dagOf`'s, its schedule the one it was
given, its units the population's `N_p`, and its units and draw the service's (`LayoutUnits`) at every `K_d`. -/
def DeclaredOfProgram : Prop :=
  ∀ (p : SampledProofs.Program) (query : Json) (s : Schedule) (Cd : Declared ℕ), declaredOf p query s = .ok Cd →
    dagOf p = .ok Cd.dag ∧ s.declared Cd.dag Cd.units Cd.unit = Cd ∧
      ∀ Np, SampledProofs.Partition.units p query = .ok Np → Cd.units = Np ∧ ∀ Kd, LayoutUnits Cd Np Kd p query

end Properties.Compose
