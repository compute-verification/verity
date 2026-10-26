import Proofs.Accounting.TS.Session.Tile.Assumptions
import Proofs.Accounting.TS.Session.Lanes.Lane
import Proofs.Pouw.H1Zy.Tile

/-!
# A tile unit's `TileChecks` from its Definition

`Accounting.Assumptions.TS.TileChecks` bundles three claims: the tile session's circuit's input wires sit where `dec`
reads them, the circuit computes `PouwH1TileZy_v1{TILE}`, and that Definition's `ok = 1` means `TileOk`. The third is
proved here for every input, over either TILE (`PouwH1Tile_v1` or `PouwH1Tile_v3` at any noise), at a `K` the tile's
EXACT binds at (`32 ∣ K`, `K < 2^17`), with `TileChecks`' `dq` the Definition's own (`Accounting.TS.Lanes.dqOf`):

* `tileView`: the tile's words as `TileView` reads them, as `Accounting.Served` does (codes s8, z s32, scales and y
  their encodings);
* `tileOk_of_def`: where `Pouw.H1Zy.tileOk form dqW` is `true` on the words, their view meets `TileOk` at `dqOf dqW`
  (`Pouw.H1Zy.tileOk_spec`: every cell's z is the exact dot and y its dequant).

What stays of `TileChecks` is the first two claims, `TileComputes`: on every wire assignment the session's evaluation
agrees with at a unit, whose `ok` wires carry the `ok` row, the Definition's `ok` is `true` on the words `rd` reads
from the registered reads. `tileChecks_of_computes` gives `TileChecks` at the view of `rd` from it, and
`Accounting.Guarantees.TS.TileSessionSound` takes it in `TileChecks`' place. Still parameters:
`rd`, the words the unit's registered reads carry, the v3 noise as a function of the public inputs, and `dqW`'s float
roundings.
-/

namespace Accounting.TS.Tile

open Accounting.TS Accounting.TS.Checked Accounting.TS.Held Flock.SoundAtBridge

/-- **The tile's words as `TileView` reads them**: codes and z their s32 and s8 values, scales and y their encodings
as unsigned numbers. -/
def tileView {R C K : ℕ} {P : Type} (x : Pouw.H1Zy.TileIn R C K P) : TileView R C K where
  a i k := (x.a i k).toInt
  sa i := ((x.sa i).toNat : ℤ)
  b j k := (x.b j k).toInt
  sb j := ((x.sw j).toNat : ℤ)
  z i j := (x.zr i j).toInt
  y i j := ((x.yr i j).toNat : ℤ)

private theorem ofInt_toNat32 (v : BitVec 32) : BitVec.ofInt 32 (v.toNat : ℤ) = v := by
  rw [BitVec.ofInt_natCast, BitVec.ofNat_toNat, BitVec.setWidth_eq]

/-- **Claim 3 of `TileChecks`**: at a `K` the tile's EXACT binds at, where `PouwH1TileZy_v1{TILE}`'s `ok` on the
tile's words is `true`, their view meets `TileOk` at `dqOf dqW`. -/
theorem tileOk_of_def {R C K : ℕ} {P : Type} (form : Pouw.H1Zy.TileForm R C K P)
    (dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16) (x : Pouw.H1Zy.TileIn R C K P) (hK32 : 32 ∣ K)
    (hK : K < 2 ^ 17) (h : Pouw.H1Zy.tileOk form dqW x = true) : TileOk (Lanes.dqOf dqW) (tileView x) := by
  intro i j
  obtain ⟨hz, hy⟩ := Pouw.H1Zy.tileOk_spec form dqW x hK32 hK h i j
  refine ⟨?_, ?_⟩
  · show (x.zr i j).toInt = ∑ k, (x.a i k).toInt * (x.b j k).toInt
    rw [hz]
    exact Pouw.H1Zy.toInt_ofInt_dot _ _ hK32 hK
  · show ((x.yr i j).toNat : ℤ) = ((dqW (BitVec.ofInt 32 (x.zr i j).toInt) (BitVec.ofInt 32 ((x.sa i).toNat : ℤ))
      (BitVec.ofInt 32 ((x.sw j).toNat : ℤ))).toNat : ℤ)
    rw [BitVec.ofInt_toInt, ofInt_toNat32, ofInt_toNat32, hy, hz]

/-- **A tile session's circuit computes its Definition at its registered reads** (claims 1 and 2 of `TileChecks`):
at the session `s`, on every wire assignment `vals` the key program's evaluation agrees with on every wire of unit `u`,
whose `ok` wires of `u` each carry the bit of a value that is the `ok` row at its position (`OkVal`, the one word `1`),
`PouwH1TileZy_v1{TILE}`'s `ok` (`Pouw.H1Zy.tileOk` at TILE's form `form` and the dequant `dqW`) is `true` on the words
`rd` reads from `vals`' registered reads at `u`. -/
def TileComputes {R C K : ℕ} {P : Type} (form : Pouw.H1Zy.TileForm R C K P)
    (dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16) (s : RowsSession) (ok : String)
    (rd : (ℕ → ℕ → Option Bool) → ℕ → Pouw.H1Zy.TileIn R C K P) : Prop :=
  ∀ (vals : Fin s.Circ.N → Bool) (u : Fin s.n), (∀ g ∈ s.P.io u, s.Circ.eval vals g = vals g) →
    (∀ g, OkWire s ok u g → ∃ w, OkVal s (s.vb.pos g) w ∧ vals g = s.vb.bit w g) →
    Pouw.H1Zy.tileOk form dqW (rd (rowsReg s vals) u.val) = true

/-- **`TileChecks` from the circuit computing its Definition**: with `dq` the Definition's (`dqOf dqW`) and the view
of the words `rd` reads (`tileView`), `TileComputes` gives `Accounting.Assumptions.TS.TileChecks`, which asks only at a
`K` the tile's EXACT binds at; that the Definition's `ok = 1` is `TileOk` is `tileOk_of_def`. -/
theorem tileChecks_of_computes {R C K : ℕ} {P : Type} (form : Pouw.H1Zy.TileForm R C K P)
    (dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16) (s : RowsSession) (ok : String)
    (rd : (ℕ → ℕ → Option Bool) → ℕ → Pouw.H1Zy.TileIn R C K P) (h : TileComputes form dqW s ok rd) :
    Accounting.Assumptions.TS.TileChecks (Lanes.dqOf dqW) s ok (fun v u => tileView (rd v u)) :=
  fun hK32 hK vals u hv hok => tileOk_of_def form dqW _ hK32 hK (h vals u hv hok)

end Accounting.TS.Tile
