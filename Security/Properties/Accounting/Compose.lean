import Definitions.Accounting.Compose
import Properties.SampledProofs.Draw

/-!
# A declared circuit's proof units and draw are the service's (`SampledProofs.Slots.layout`, `Select.select`)

`Accounting.Compose.Declared` gives each gate its proof unit (`Declared.unit`), each value its commit coin
(`Declared.commitRound`) and each round the values its seed names (`Declared.seedNames`), and the composed credit
(`Definitions.Accounting.Guarantees.Compose`) draws, proves and charges tiles by them, its draw `Pouw.Window.subsetDraw`
(`Accounting.Compose.drawOf`). `UnitsOfLayout` is the statement that the declared circuit's units and draw are the
service's on a program `p` and query:
* **units**: the service draws and proves the units of the verifier's own partition of its program
  (`SampledProofs.Slots.layout`: the population's replay units, each one proof unit and its gates, then the input
  class). Every gate of `Cd` that is not an input lies in the gates of the replay unit that is its own unit, every
  input slot's gate is an input of `Cd`, and every population unit is one of `Cd`'s. It reads `Cd`'s gate `i` as `p`'s
  root gate `i`, and a `Q_word` unit's span may hold another unit's gates (`Partition.unitGates`), so it bounds each
  gate inside its own unit's span, not the span inside the unit;
* **the draw**: the code's draw under the `subset:K_d` law, from `L` uniform bytes (`Select.select`, its drawn units
  then its closure, `Properties.SampledProofs.bytesOf`), is accepted and draws no unit of a set `U` no more often than
  `subsetDraw K_d` of a uniform permutation of the population's `N_p` tiles misses `U`.

`declaredOf` builds a `Declared` from a compiled Program: it keeps a given circuit's gates and schedule (its call sites,
rounds, coins, commit coins and seed preimages) and gives each gate the unit of the layout's population unit whose span
holds it, with the population's `N_p` units, refusing where the layout or the population is refused, where a non-input
gate lies in no population unit's span, where an input slot's gate is no input of the circuit, or where a population
unit is past `N_p`. `DeclaredUnits` is `UnitsOfLayout` of every circuit it accepts, with `N_p` its units.

`SeededSchedule` is the schedule of PoUW's seeded code. Its coins are one salt a window and the draw
(`PouwCommit.LABELS`): the salt is opened after the weights' root (`PouwCommit.epochSalt`), and the draw after the
transcript's (`PouwCommit.drawCoin` at `PouwCommit.drawAfter`). So a registered input is committed before coin `0`, and
every other value the prover writes before the draw, coin `rounds`. Each unit's seed is `seeds.PouwSeed` of the salt,
its position and its seed group's `pouw.roots` slot; the salt's derivation reads the weights' root
(`PouwCommit.saltOf`), and `pouw.roots` is a Program input that no check ties to the unit's operand rows, so a seed
names a value exactly when it is a registered input. Under that schedule `Accounting.Compose.LayerCommitted` fails
wherever a value's IR cut is before the draw and it isn't a computed value only its own unit reads (every run input any
gate reads, `Properties.Served.SeededScheduleNotLive`), and `Accounting.Compose.SeedNamed` fails wherever a seeded round
reads an activation, which is why neither the live nor the seeded statements credit anything on the seeded code. The
live schedule, one coin a round, is `Accounting.Served.Tie`'s, and `declaredOf` takes it from the circuit it is given.
-/

open Lean (Json)
open Pouw.RelativeSecurity (prob)

namespace Properties.Compose

/-- **`Cd`'s units and draw are the service's** on program `p` under `query`, wherever the layout is defined: each of
`Cd`'s non-input gates lies in a proof unit of the replay unit that is its own unit, each input slot's gate is an input
of `Cd`, each population unit is one of `Cd`'s `units`; and where the population is `N_p` units, the code's
`subset:K_d` draw from `L` uniform bytes is accepted and misses a set `U` of units with probability at most the model's
draw's (`Pouw.Window.subsetDraw K_d` of a uniform permutation of `Fin N_p`). -/
def UnitsOfLayout {ι : Type} (Cd : Accounting.Compose.Declared ι) (Np Kd : ℕ) (p : SampledProofs.Program)
    (query : Json) : Prop :=
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

/-- **`Cd` is on PoUW's seeded schedule**: each value `Cd` writes is committed before coin `0` when it is a registered
input and before the draw (`Cd.rounds`) otherwise, and each round's seed names a value `Cd` writes exactly when it is a
registered input (through the salt, which names the weights' root). -/
def SeededSchedule {ι : Type} (Cd : Accounting.Compose.Declared ι) : Prop :=
  (∀ ℓ, Cd.Writes ℓ → Cd.commitRound ℓ = match Cd.binding ℓ with
    | some .registered => 0
    | _ => Cd.rounds) ∧
  ∀ r ℓ, Cd.Writes ℓ → (Cd.seedNames r ℓ = true ↔ Cd.binding ℓ = some .registered)

/-- **The population unit whose gates hold gate `i`** in layout `l`: the first replay unit in slot order of a
population unit with a proof unit `[lo, hi)` holding `i`, and none where no population unit's does. -/
def unitOf (l : SampledProofs.Slots.Layout) (i : ℕ) : Option ℕ :=
  l.rus.toList.findSome? fun r => r.unit.filter fun _ => r.pus.toList.any fun g => decide (g.1 ≤ i ∧ i < g.2)

/-- **`Cd₀` in layout `l`'s units**: `Cd₀`'s gates and schedule, each gate the unit `unitOf` gives it (`0` for an input,
which no population unit holds), and `Np` units. -/
def withUnits {ι : Type} (Cd₀ : Accounting.Compose.Declared ι) (l : SampledProofs.Slots.Layout) (Np : ℕ) :
    Accounting.Compose.Declared ι :=
  { Cd₀ with units := Np, unit := fun i => (unitOf l i).getD 0 }

/-- **`declaredOf`'s checks**: every non-input gate of `Cd₀` lies in a population unit's span, every population unit is
below `Np`, and every input slot's gate is an input of `Cd₀`. -/
def unitsChecked {ι : Type} (Cd₀ : Accounting.Compose.Declared ι) (l : SampledProofs.Slots.Layout) (Np : ℕ) : Bool :=
  (List.range Cd₀.dag.length).all (fun i => (Cd₀.binding i).isSome || (unitOf l i).isSome) &&
    l.rus.toList.all fun r => match r.unit with
      | some u => decide (u < Np)
      | none => r.pus.toList.all fun g => decide (g.1 < Cd₀.dag.length) && (Cd₀.binding g.1).isSome

/-- **The declared circuit of a compiled Program**: `Cd₀`'s gates and schedule in the units of the verifier's partition
of program `p` under `query` (`withUnits`), where `Slots.layout` and `Partition.units` accept and `unitsChecked` holds;
an error otherwise. -/
def declaredOf {ι : Type} (Cd₀ : Accounting.Compose.Declared ι) (p : SampledProofs.Program) (query : Json) :
    Except String (Accounting.Compose.Declared ι) := do
  let l ← SampledProofs.Slots.layout p query
  let Np ← SampledProofs.Partition.units p query
  if unitsChecked Cd₀ l Np then pure (withUnits Cd₀ l Np)
  else throw "the circuit's gates are not the partition's: a non-input gate in no unit, or an input slot no input"

/-- **A declared circuit `declaredOf` accepts is the service's**: it has the gates and schedule of the circuit it was
given, the population's `N_p` units, and `UnitsOfLayout` at every `K_d`. -/
def DeclaredUnits : Prop :=
  ∀ (ι : Type) (Cd₀ Cd : Accounting.Compose.Declared ι) (p : SampledProofs.Program) (query : Json),
    declaredOf Cd₀ p query = .ok Cd →
      Cd.dag = Cd₀.dag ∧ Cd.calls = Cd₀.calls ∧ Cd.call = Cd₀.call ∧ Cd.rounds = Cd₀.rounds ∧
        Cd.round = Cd₀.round ∧ Cd.coinMode = Cd₀.coinMode ∧ Cd.commitRound = Cd₀.commitRound ∧
        Cd.seedNames = Cd₀.seedNames ∧
        ∀ Np, SampledProofs.Partition.units p query = .ok Np → Cd.units = Np ∧ ∀ Kd, UnitsOfLayout Cd Np Kd p query

end Properties.Compose
