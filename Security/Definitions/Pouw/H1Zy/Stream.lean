import Definitions.Pouw.H1Zy.Chain
import Definitions.Pouw.H1Zy.Tile

/-!
# `PouwH1LaneZy_v1{PouwH1Tile_v3}` on words (trusted)

`h1zy.PouwH1LaneZy` takes its leaf's tile as `PouwH1Tile_v1` (z its EXACT, `laneOk`) or `PouwH1Tile_v3`, whose z is
`h1tile._cell_z` on the lane's chain words (`h1zy._lane_body`). On words, v3's z of lane `l`'s cell `(s, t)` is
`streamZ` (`Definitions.Pouw.H1Zy.Tile`) on the lane's 4 A rows and 8 B rows at the lane's noise (`laneTileNoise`: row
`laneRow l s`'s and column `laneCol l t`'s blocks of seed_j's stream, as `laneNoise` reads them), and the lane's `ok` is
`laneOk` with that z (`laneOkZ` at `laneZ3`).
-/

namespace Pouw.H1Zy

/-- **Lane `l`'s noise as the v3 tile's** (`h1tile.noise` at its 4 rows and 8 columns): the rows' dither bits and digit
coins and the columns' coin bits as `laneNoise` reads them, and the columns' 32 noise bytes before `PouwU129`. -/
def laneTileNoise (K : ℕ) (x : SeedIn) (l : Fin 32) : TileNoise 4 8 K where
  bd s k := (laneNoise K x l).dither s k.val
  zc s d ρ := (laneNoise K x l).digit s d ρ
  cc t k := (laneNoise K x l).coin t k.val
  ub t ρ := BitVec.ofNat 8 (streamVal (seedJ x) (colFirst K (laneCol l t)) (K + 8 * ρ.val) 8)

/-- A lane's 4 A rows and 8 B rows as the v3 tile's inputs (`streamZ` reads only the codes). -/
def laneTileIn {K : ℕ} (a : Fin 4 → Fin K → BitVec 8) (b : Fin 8 → Fin K → BitVec 8) : TileIn 4 8 K Unit where
  a := a
  sa _ := 0
  b := b
  sw _ := 0
  zr _ _ := 0
  yr _ _ := 0
  pub := ()

/-- **`PouwH1Tile_v3`'s z of lane `l`'s cell `(s, t)`** at the seeds `x`: `streamZ` at the lane's noise. -/
def laneZ3 {K : ℕ} (x : SeedIn) (l : Fin 32) (a : Fin 4 → Fin K → BitVec 8) (b : Fin 8 → Fin K → BitVec 8)
    (s : Fin 4) (t : Fin 8) : BitVec 32 :=
  streamZ (laneTileNoise K x l) (laneTileIn a b) s t

/-- **`PouwH1LaneZy_v1{TILE}`'s `ok`** at a 32 × 32 leaf, `laneOk` with its cells' z the tile's `zf` (of the public
inputs, the lane `sel` names and the rows' codes): at `fun _ _ a b s t => gemmAccS8 (a s) (b t)` it is `laneOk`
(`PouwH1Tile_v1`), at `laneZ3` of the public inputs' seeds `PouwH1Tile_v3`'s. -/
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
