import Definitions.Accounting.Compose
import SampledProofs.Slots

/-!
# A declared circuit's proof units are the verifier's (`SampledProofs.Slots.layout`)

`Accounting.Compose.Declared` gives each gate its proof unit (`Declared.unit`), and the composed credit
(`Models.Accounting.Guarantees.Compose`) draws, proves and charges tiles by it. The service draws and proves the units
of the verifier's own partition of its program (`SampledProofs.Slots.layout`: the population's replay units, each one
proof unit and its gates, then the input class). `UnitsOfLayout` is the statement that the two agree on a program `p`
and query: every gate of `Cd` that is not an input lies in the gates of the replay unit that is its own unit, every
input slot's gate is an input of `Cd`, and every population unit is one of `Cd`'s. It reads `Cd`'s gate `i` as `p`'s
root gate `i`, and a `Q_word` unit's span may hold another unit's gates (`Partition.unitGates`), so it bounds each gate
inside its own unit's span, not the span inside the unit.

It is stated, not proved: nothing builds a `Declared` from a compiled Program yet, which is the composed statement's
faithfulness gap (note:compute-accounting/20261018T1600Z-draft-theorem-compose-r5, Open). The extractor that builds
one, with this property proved of it, closes it. It covers units only: a proved version must also cover each value's
commit coin (`Declared.commitRound`, the code's root for it fixed before that coin, and no later than its IR cut by
`Assumptions.Compose.LayerCommitted`) and the draw (the code's drawn tiles are `Accounting.Compose.drawOf`'s), as
RT-COMPOSE r4's check 1 asks.
-/

open Lean (Json)

namespace Properties.Compose

/-- **`Cd`'s proof units are the layout's** on program `p` under `query`, wherever the layout is defined: each of `Cd`'s
non-input gates lies in a proof unit of the replay unit that is its own unit, each input slot's gate is an input of
`Cd`, and each population unit is one of `Cd`'s `units`. -/
def UnitsOfLayout {ι : Type} (Cd : Accounting.Compose.Declared ι) (p : SampledProofs.Program) (query : Json) : Prop :=
  ∀ l, SampledProofs.Slots.layout p query = .ok l →
    (∀ i, i < Cd.dag.length → Cd.binding i = none →
      ∃ r ∈ l.rus.toList, r.unit = some (Cd.unit i) ∧ ∃ g ∈ r.pus.toList, g.1 ≤ i ∧ i < g.2) ∧
    (∀ r ∈ l.rus.toList, r.unit = none → ∀ g ∈ r.pus.toList, g.1 < Cd.dag.length ∧ (Cd.binding g.1).isSome) ∧
    ∀ r ∈ l.rus.toList, ∀ u, r.unit = some u → u < Cd.units

end Properties.Compose
