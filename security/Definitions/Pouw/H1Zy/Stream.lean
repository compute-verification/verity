module

public import Definitions.Pouw.H1Zy.LaneChunk
public import Definitions.Pouw.H1Zy.Tile
public import Definitions.Pouw.PearlC.Sha512

/-!
# A lane's `ok` at any cell z (trusted)

`h1zy.PouwH1LaneZy` takes its leaf's tile as `PouwH1Tile_v1` (z its EXACT, `laneOk`) or `PouwH1Tile_v3`, whose z is
`h1tile._cell_z` on the lane's chain words (`h1zy._lane_body`). `laneOkZ` is the lane's `ok` with its cells' z a
parameter.
-/

@[expose] public section

namespace Pouw.H1Zy

/-- **`PouwH1LaneZy_v1{TILE}`'s `ok`** at a 32 × 32 leaf, `laneOk` with its cells' z the tile's `zf` (of the public
inputs, the lane `sel` names and the rows' codes): at `fun _ _ a b s t => gemmAccS8 (a s) (b t)` it is `laneOk`
(`PouwH1Tile_v1`). -/
def laneOkZ {K nw : ℕ} {P : Type}
    (zf : P → Fin 32 → (Fin 4 → Fin K → BitVec 8) → (Fin 8 → Fin K → BitVec 8) → Fin 4 → Fin 8 → BitVec 32)
    (chain : P → Fin 32 → (Fin 4 → Fin K → BitVec 8) → (Fin 8 → Fin K → BitVec 8) → Fin nw → BitVec 32)
    (dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16) (x : LaneIn K nw P) : Bool :=
  let picked : Fin 4 → Fin 8 → BitVec 32 × BitVec 16 := fun s t =>
    pick (0, 0) x.sel (List.ofFn fun ln : Fin 32 => (x.zr (laneRow ln s) (laneCol ln t), x.yr (laneRow ln s) (laneCol ln t)))
  let ok := zyCheck dqW (fun s t => (picked s t).1) (fun s t => (picked s t).2) (zf x.pub x.l x.a x.b) x.sa x.sw
  let live := tree orBit false (List.ofFn x.admit)
  let eq := List.ofFn fun w => eq32 (x.partials w) (mask32 (chain x.pub x.l x.a x.b w) live)
  let high := tree orBit false (List.ofFn fun t : Fin 11 => x.lane ⟨5 + t.val, by omega⟩)
  mux high (tree andBit true (ok :: eq)) false

end Pouw.H1Zy
