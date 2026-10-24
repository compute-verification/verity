import Proofs.Accounting.TS.Session.Lanes.Chain
import Proofs.Pouw.H1Zy.Stream

/-!
# `LeafChecks` at lanes over `PouwH1Tile_v3`

`h1zy.PouwH1LaneZy` takes its leaf's tile as `PouwH1Tile_v1` (z its EXACT) or `PouwH1Tile_v3` (z from its lane's O' and
D words, `h1tile._cell_z`). `Lane`'s `LaneComputes` reads the v1 form. Here is the v3 form, `LaneComputes3`: the
lane's circuit computes `Pouw.H1Zy.laneOkZ` at `laneZ3Of`, v3's cell z (`Pouw.H1Zy.laneZ3`) at the salt, position and
roots `decLane` reads. Since v3's z is v1's at every noise (`Pouw.H1Zy.laneZ3_eq`, from `streamZ_eq_gemmAccS8`), the
two `ok`s agree at a `K` the tile binds at, and `LeafChecks` follows from `LaneComputes3` at the same `dq` and `part`
(`leafChecks_of_computes3`, `leafChecks_h1_v3`).
-/

namespace Accounting.TS.Lanes

open Accounting.TS Accounting.TS.Checked Flock.SoundAtBridge PouwWindow.Lanes

/-- **`PouwH1Tile_v3`'s cell z of a lane** (`h1tile._cell_z`): `Pouw.H1Zy.laneZ3` at the public inputs' salt, position
and roots. -/
def laneZ3Of {K : ℕ} (P : LanePub) (l : Fin 32) (a : Fin 4 → Fin K → BitVec 8) (b : Fin 8 → Fin K → BitVec 8)
    (s : Fin 4) (t : Fin 8) : BitVec 32 :=
  Pouw.H1Zy.laneZ3 P.salt P.position P.roots l a b s t

/-- **Each lane's circuit computes `PouwH1LaneZy_v1{PouwH1Tile_v3}` at its port rows**: `LaneComputes` with the
lane's `ok` `Pouw.H1Zy.laneOkZ` at v3's cell z, `laneZ3Of`. -/
def LaneComputes3 {K nw : ℕ} (T : Tile)
    (chain : LanePub → Fin 32 → (Fin 4 → Fin K → BitVec 8) → (Fin 8 → Fin K → BitVec 8) → Fin nw → BitVec 32)
    (dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16) (sess : Option (Fin 32) → RowsSession) : Prop :=
  ∀ l (vals : Fin (sess (some l)).Circ.N → Bool),
    (∀ u, ∀ g ∈ (sess (some l)).P.io u, (sess (some l)).Circ.eval vals g = vals g) →
    (decL K nw T (sess (some l)) (portRead (sess (some l)) vals)).ok =
      Pouw.H1Zy.laneOkZ laneZ3Of chain dqW (laneIn K nw T (portBit (sess (some l)) (portRead (sess (some l)) vals)))

/-- At a `K` the tile binds at, `LaneComputes3` is `LaneComputes`: v3's lane `ok` is v1's. -/
theorem laneComputes_of_3 {K nw : ℕ} (T : Tile)
    (chain : LanePub → Fin 32 → (Fin 4 → Fin K → BitVec 8) → (Fin 8 → Fin K → BitVec 8) → Fin nw → BitVec 32)
    (dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16) (sess : Option (Fin 32) → RowsSession) (hK32 : 32 ∣ K)
    (hK : K < 2 ^ 17) (h : LaneComputes3 T chain dqW sess) : LaneComputes T chain dqW sess := fun l vals hv =>
  (h l vals hv).trans (Pouw.H1Zy.laneOkZ_eq_laneOk _ _ _ _ fun P l a b s t =>
    Pouw.H1Zy.laneZ3_eq P.salt P.position P.roots l a b s t hK32 hK)

/-- **`LeafChecks` from lanes over `PouwH1Tile_v3`**: at a `K` the tile binds at, `LaneComputes3` and
`CombineComputes` give `Accounting.Assumptions.TS.LeafChecks` at the same `dq` and `part` as v1's. -/
theorem leafChecks_of_computes3 {K nw : ℕ} (T : Tile)
    (chain : LanePub → Fin 32 → (Fin 4 → Fin K → BitVec 8) → (Fin 8 → Fin K → BitVec 8) → Fin nw → BitVec 32)
    (dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16) (sess : Option (Fin 32) → RowsSession) (hK32 : 32 ∣ K)
    (hK : K < 2 ^ 17) (hL : LaneComputes3 T chain dqW sess) (hC : CombineComputes sess (decC nw T)) :
    Accounting.Assumptions.TS.LeafChecks (dqOf dqW) (partOf chain) sess (decL K nw T) (decC nw T) :=
  leafChecks_of_computes T chain dqW sess hK32 hK (laneComputes_of_3 T chain dqW sess hK32 hK hL) hC

/-- **`LeafChecks` at lanes over `PouwH1Tile_v3`, at the Definition's own chain and dequant**. -/
theorem leafChecks_h1_v3 {K nw : ℕ} (T : Tile) (sess : Option (Fin 32) → RowsSession) (hK32 : 32 ∣ K)
    (hK : K < 2 ^ 17) (hL : LaneComputes3 T (laneChain (K := K) (nw := nw)) Pouw.H1Zy.dequantW sess)
    (hC : CombineComputes sess (decC nw T)) :
    Accounting.Assumptions.TS.LeafChecks (dqOf Pouw.H1Zy.dequantW) (partOf (laneChain (K := K) (nw := nw))) sess
      (decL K nw T) (decC nw T) :=
  leafChecks_of_computes3 T laneChain Pouw.H1Zy.dequantW sess hK32 hK hL hC

end Accounting.TS.Lanes
