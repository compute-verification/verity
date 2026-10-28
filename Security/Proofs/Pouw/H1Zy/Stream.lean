import Definitions.Pouw.H1Zy.Stream
import Proofs.Pouw.H1Zy.Tile

/-!
# A lane's `ok` over `PouwH1Tile_v3` is its `ok` over `PouwH1Tile_v1`

At a `K` the tile binds at (`32 ∣ K`, `K < 2^17`), v3's z of a lane's cell is EXACT's for every seed (`laneZ3_eq`, from
`streamZ_eq`), so `laneOkZ` at `laneZ3` is `laneOk` (`laneOkZ_eq_laneOk`).
-/

namespace Pouw.H1Zy

/-- **A v3 lane's cell z is v1's**: `laneZ3` is `GemmAccumulatorS8_v2`'s word on the cell's rows' codes, at a `K` the
tile binds at, for every seed. -/
theorem laneZ3_eq {K : ℕ} (x : SeedIn) (l : Fin 32) (a : Fin 4 → Fin K → BitVec 8) (b : Fin 8 → Fin K → BitVec 8)
    (s : Fin 4) (t : Fin 8) (hK32 : 32 ∣ K) (hK : K < 2 ^ 17) :
    laneZ3 x l a b s t = gemmAccS8 (a s) (b t) := by
  rw [laneZ3, streamZ_eq _ _ _ _ hK32 hK, gemmAccS8_eq _ _ hK32 hK]
  rfl

/-- **A lane's `ok` at a tile whose z is EXACT's is `laneOk`**: so `PouwH1LaneZy_v1{PouwH1Tile_v3}`'s `ok`
(`laneOkZ` at `laneZ3`, by `laneZ3_eq`) is `PouwH1LaneZy_v1{PouwH1Tile_v1}`'s. -/
theorem laneOkZ_eq_laneOk {K nw : ℕ} {P : Type}
    (zf : P → Fin 32 → (Fin 4 → Fin K → BitVec 8) → (Fin 8 → Fin K → BitVec 8) → Fin 4 → Fin 8 → BitVec 32)
    (chain : P → Fin 32 → (Fin 4 → Fin K → BitVec 8) → (Fin 8 → Fin K → BitVec 8) → Fin nw → BitVec 32)
    (dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16) (x : LaneIn K nw P)
    (hz : ∀ p l a b s t, zf p l a b s t = gemmAccS8 (a s) (b t)) :
    laneOkZ zf chain dqW x = laneOk chain dqW x := by
  have : zf = fun _ _ a b s t => gemmAccS8 (a s) (b t) := by
    funext p l a b s t
    exact hz p l a b s t
  subst this
  rfl

end Pouw.H1Zy
