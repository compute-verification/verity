import Proofs.Accounting.TS.Session.Held.Layer

/-!
# The premise that ties a checked unit's C-Flock circuit to its declared circuit

`OkRealizes`: the session's key program computes, at each unit, the checks of one unit of the declared circuit `Cd`,
so that its `ok` wires carrying the `ok` row means those checks hold. It belongs in
`Security/Definitions/Accounting/Assumptions/TS`, but it reads `RowsSession`, which is in the proofs package, so it is here
beside `Accounting.Assumptions.TS`'s other session premises (`Proofs.Accounting.TS.Session.Assumptions`).
-/

namespace Accounting.Assumptions.TS

open Accounting.TS Accounting.TS.Held Flock.SoundAtBridge

/-- `accounting-ts-ok-realizes` (the checked unit's circuit is its declared unit's). **The key
program's `ok` wires carry `Cd`'s checks**: at the session `s`, for every wire assignment `vals` that the key
program's evaluation (`Circuit.eval`) agrees with on every wire of unit `u`, and whose `ok` wires of `u` (`OkWire`)
each carry the bit of a value that is the `ok` row at its position (`OkVal`, the one word `1`), every check of the
declared circuit `Cd`'s unit `unitOf u` holds at `dec vals` (`Accounting.Served.Held`).

It is a fact of one compiled circuit and one declared circuit, with `unitOf` and `dec` the map between them: that the
key program of `PouwH1TileZy_v1`'s C-Flock circuit computes its `ok` word as the conjunction of the checks `Cd`
declares of the unit (the Σ check and the dequant check), on the wires `dec` reads. Nothing in Lean derives a C-Flock
circuit from a `Declared` one or checks one against the other; `circuit-check` binds the Definition to its Boolean
lowering, not to `Cd`. It stays a premise until that binding is stated. -/
def OkRealizes {ι : Type} (s : RowsSession) (ok : String) (Cd : Accounting.Compose.Declared ι)
    (cls : ℕ → Accounting.Served.Cls) (unitOf : Fin s.n → ℕ) (dec : (Fin s.Circ.N → Bool) → ι → ℤ) : Prop :=
  ∀ (vals : Fin s.Circ.N → Bool) (u : Fin s.n), (∀ g ∈ s.P.io u, s.Circ.eval vals g = vals g) →
    (∀ g, OkWire s ok u g → ∃ w, OkVal s (s.vb.pos g) w ∧ vals g = s.vb.bit w g) →
    Accounting.Served.Held Cd cls (unitOf u) (dec vals)

end Accounting.Assumptions.TS
