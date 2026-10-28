import Proofs.Accounting.TS.Session.Tile.Layer

/-!
# The named premise a tile unit's C-Flock session takes for its Definition's checks

`TileChecks`: the session's key program computes `PouwH1TileZy_v1`'s Definition at its decoded registered ports, so
that its `ok` wires carrying the `ok` row means the tile's cells check. It is the tile counterpart of
`Accounting.Assumptions.TS.LeafChecks` (option A's 33 statements), and like it reads `RowsSession`, in the proofs
package, so it sits here beside `Accounting.Assumptions.TS`'s other session premises.
`Accounting.Guarantees.TS.TileSessionSound` takes what stays of it once the Definition's `ok = 1` is proved to mean
`TileOk`, that the circuit computes its Definition (`Accounting.TS.Tile.TileComputes`, which gives it by
`tileChecks_of_computes`).
-/

namespace Accounting.Assumptions.TS

open Accounting.TS Accounting.TS.Checked Accounting.TS.Held Accounting.TS.Tile Flock.SoundAtBridge

/-- `accounting-ts-tile-checks` (a fact of `PouwH1TileZy_v1`'s circuit, to be computed by a check).
**A tile session's circuit computes its Definition at its decoded ports**: at the session `s`,
at a `K` the tile binds at (`32 ∣ K` and `K < 2^17`, `GemmAccumulatorS8`'s bind, which refuses any other; from `2^17`
a step of its s8 accumulator can clamp, so `z` need not be the exact dot), for every wire assignment `vals` that the
key program's evaluation (`Circuit.eval`) agrees with on every wire of unit
`u`, and whose `ok` wires of `u` (`OkWire`) each carry the bit of a value that is the `ok` row at its position
(`OkVal`, the one word `1`), the view `dec` reads from `vals`' registered reads at unit `u` meets `TileOk` at `dq`:
each of its cells has `z = Σ_k a_k b_k` and `y = dq [z, s_a, s_b]`.

The decode reads only the registered reads (`rowsReg`), so the premise holds of the circuit only if every input its
`ok` depends on is registered: its A and B rows with their scales and its z/y record. A port read only from the public
file is free in `vals`. flock-verify's complete check (`Flock.Registered.checkComplete`, on every statement whenever
`--registered` is given) refuses a session one of whose circuit's ports is no port of `own`, and `tileLayer`'s acceptance
includes it. `dq` is the Program's dequant on
words (`PouwZyDequant_v2`'s word view). -/
def TileChecks {R C K : ℕ} (dq : List ℤ → ℤ) (s : RowsSession) (ok : String)
    (dec : (ℕ → ℕ → Option Bool) → ℕ → TileView R C K) : Prop :=
  32 ∣ K → K < 2 ^ 17 → ∀ (vals : Fin s.Circ.N → Bool) (u : Fin s.n), (∀ g ∈ s.P.io u, s.Circ.eval vals g = vals g) →
    (∀ g, OkWire s ok u g → ∃ w, OkVal s (s.vb.pos g) w ∧ vals g = s.vb.bit w g) →
    TileOk dq (dec (rowsReg s vals) u.val)

end Accounting.Assumptions.TS
