import Definitions.Accounting.Compose
import Properties.SampledProofs.Draw

/-!
# A declared circuit's proof units, commit coins, seed preimages and draw are the service's (`SampledProofs.Slots.layout`, `Select.select`)

`Accounting.Compose.Declared` gives each gate its proof unit (`Declared.unit`), each value its commit coin
(`Declared.commitRound`) and each round the values its seed names (`Declared.seedNames`), and the composed credit
(`Definitions.Accounting.Guarantees.Compose`) draws, proves and charges tiles by them, its draw `Pouw.Window.subsetDraw`
(`Accounting.Compose.drawOf`). `UnitsOfLayout` is the statement that the declared circuit is the service's on a
program `p` and query:
* **units**: the service draws and proves the units of the verifier's own partition of its program
  (`SampledProofs.Slots.layout`: the population's proof blocks, each its unit and its gates, then the input
  class). Every gate of `Cd` that is not an input lies in the gates of the proof block that is its own unit, every
  input slot's gate is an input of `Cd`, and every population unit is one of `Cd`'s. It reads `Cd`'s gate `i` as `p`'s
  root gate `i`, and a `Q_word` unit's span may hold another unit's gates (`Partition.unitGates`), so it bounds each
  gate inside its own unit's span, not the span inside the unit;
* **commit coins**: the code's schedule. PoUW's coins are one salt a window and the draw (`PouwCommit.LABELS`): the
  salt is opened after the weights' root (`PouwCommit.epochSalt`), and the draw after the transcript's
  (`PouwCommit.drawCoin` at `PouwCommit.drawAfter`). So a registered input is committed before coin `0`, and every
  other value the prover writes before the draw, coin `rounds`. Under that schedule `Accounting.Compose.LayerCommitted`
  fails wherever a value's IR cut is before the draw and it isn't a computed value only its own unit reads (every run
  input any gate reads, `Properties.Served.SeededScheduleNotLive`), which is why the live statements credit nothing
  on today's code;
  the seeded statements' binding layer (`Declared.atDraw`) cuts every value at the draw, no earlier than this;
* **seed preimages**: what the code's seeds name (`Declared.seedNames`). Each unit's seed is `seeds.PouwSeed` of the
  salt, its position and its seed group's `pouw.roots` slot; the salt's derivation reads the weights' root
  (`PouwCommit.saltOf`), and `pouw.roots` is a Program input that no check ties to the unit's operand rows. So a
  seed names a value exactly when it is a registered input, and `Accounting.Compose.SeedNamed` fails wherever a
  seeded round reads an activation (a run input or an earlier round's output), which is why the seeded statements
  credit nothing on today's code either;
* **the draw**: the code's draw under the `subset:K_d` law, from `L` uniform bytes (`Select.select`, its drawn units
  then its closure, `Properties.SampledProofs.bytesOf`), is accepted and draws no unit of a set `U` no more often than
  `subsetDraw K_d` of a uniform permutation of the population's `N_p` tiles misses `U`.

It is stated, not proved: nothing builds a `Declared` from a compiled Program yet, which is the composed statement's
open faithfulness gap. The extractor that builds one, with this property proved of it, closes it; tying each value to
its leaf under the weights' or the transcript's root is the extractor's too. The draw conjunct is over the code alone
and could be proved first (`Select.select`'s subset draw from bytes, `Draw.drawUnits`), beside
`Properties.SampledProofs.DrawMisses`.
-/

open Lean (Json)
open Pouw.RelativeSecurity (prob)

namespace Properties.Compose

/-- **`Cd` is the service's** on program `p` under `query`, wherever the layout is defined: each of `Cd`'s non-input
gates lies in the proof block that is its own unit, each input slot's gate is an input of `Cd`, each
population unit is one of `Cd`'s `units`; each value `Cd` writes is committed before coin `0` when it is a registered
input and before the draw (`Cd.rounds`) otherwise, PoUW's salt-and-draw schedule; each round's seed names a value
`Cd` writes exactly when it is a registered input (through the salt, which names the weights' root); and where the
population is `N_p` units, the code's `subset:K_d` draw from `L` uniform bytes is accepted and misses a set `U` of
units with probability at most the model's draw's (`Pouw.Window.subsetDraw K_d` of a uniform permutation of
`Fin N_p`). -/
def UnitsOfLayout {ι : Type} (Cd : Accounting.Compose.Declared ι) (Np Kd : ℕ) (p : SampledProofs.Program)
    (query : Json) : Prop :=
  ∀ l, SampledProofs.Slots.layout p query = .ok l →
    (∀ i, i < Cd.dag.length → Cd.binding i = none →
      ∃ b ∈ l.blocks.toList, b.unit = some (Cd.unit i) ∧ b.gates.1 ≤ i ∧ i < b.gates.2) ∧
    (∀ b ∈ l.blocks.toList, b.unit = none → b.gates.1 < Cd.dag.length ∧ (Cd.binding b.gates.1).isSome) ∧
    (∀ b ∈ l.blocks.toList, ∀ u, b.unit = some u → u < Cd.units) ∧
    (∀ ℓ, Cd.Writes ℓ → Cd.commitRound ℓ = match Cd.binding ℓ with
      | some .registered => 0
      | _ => Cd.rounds) ∧
    (∀ r ℓ, Cd.Writes ℓ → (Cd.seedNames r ℓ = true ↔ Cd.binding ℓ = some .registered)) ∧
    (SampledProofs.Partition.units p query = .ok Np →
      ∀ (cm : Option (Array (Int × Array Int))) (L : ℕ) (U : Finset ℕ),
        prob (fun src : Fin L → Fin 256 => ∃ r,
            SampledProofs.Select.select (.law ("subset:" ++ toString Kd)) p query none cm
              (Properties.SampledProofs.bytesOf src) = .ok r ∧ ∀ u ∈ U, u ∉ r.2.1) ≤
          prob (fun β : Equiv.Perm (Fin Np) => ∀ t ∈ Pouw.Window.subsetDraw Kd β, t.val ∉ U))

end Properties.Compose
