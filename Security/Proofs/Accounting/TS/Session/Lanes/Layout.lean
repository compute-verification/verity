import Proofs.Accounting.TS.Session.Lanes.Layer
import PouwWindow.Lanes

/-!
# A leaf's 33 statements' ports at the served tile

`PouwWindow.Lanes` (the statements' layout, held to `h1zy.statement_ports` by its difftest) states, at a leaf's tile and
contraction, which port of each of the 33 statements holds each of the Definition's inputs and its output, at which
bits of the port's row, and where the verifier opens that row (`Source`). At the served 32 × 32 leaf (`servedTile K`):

* its lanes, rows and columns are this layer's: 32 lanes, `Tile.rowOf` is `rowOf` and `Tile.colOf` is `colOf`;
* each lane's ports are opened at the rows the leaf's values are decoded from (`Lanes.Committed`): its A row at
  row slot `s` at row `rowOf l s` of `a`, its B row at column slot `t` at row `colOf l t` of `b`, the leaf's record at
  `zy`'s row, and its partials row at row `l` of `partials`; its admit bits, its public inputs (`pub<l>`: salt,
  position, roots and the index) and its `ok` (`ok<l>`) at values the verifier registers at their known content;
* the combine's `p<x>` is opened at row `x` of `partials`, its `receipt` at `receipt`'s row, and its `ok` at `ok32`.

The leaf and the receipt are the ones the registrants' rows of `a`, `b`, `zy` and `receipt` decode to at these rows
(`Lanes.Committed`). Four lanes read each A row and four each B row, and lane `l`'s `partials` and the combine's
`p<l>` read one row: each such pair reads one row of one registered tree (`PouwWindow.Lanes.ownOf`) in full, so they
decode it alike (`Lanes.Partials`' `read_bits`). The known values' content is flock-verify's known-content and
known-read checks' (`Lanes.KnownRead`).
-/

namespace Accounting.TS.Lanes

open PouwWindow.Lanes

/-- The served leaf's tile: 32 × 32 cells at contraction `K`. -/
def servedTile (K : ℕ) : Tile := ⟨32, 32, K⟩

/-- The served leaf has 32 lanes. -/
theorem servedTile_lanes (K : ℕ) : (servedTile K).lanes = 32 := by
  simp [servedTile, Tile.lanes, Tile.g, Tile.q]

/-- The layout's row of lane `l` at row slot `s` is this layer's. -/
theorem servedTile_rowOf (K : ℕ) (l : Fin 32) (s : Fin 4) : (servedTile K).rowOf l.val s.val = (rowOf l s).val := by
  first | rfl | simp [Tile.rowOf, Tile.g, Tile.q, servedTile, rowOf]

/-- The layout's column of lane `l` at column slot `t` is this layer's. -/
theorem servedTile_colOf (K : ℕ) (l : Fin 32) (t : Fin 8) : (servedTile K).colOf l.val t.val = (colOf l t).val := by
  first | rfl | simp [Tile.colOf, Tile.q, servedTile, colOf]

/-- Lane `l`'s port `a<s>` is opened at row `rowOf l s` of the registered `a`, and holds its A row's codes and scale. -/
theorem lanePorts_a (K : ℕ) (l : Fin 32) (s : Fin 4) :
    (⟨s!"a{s.val}", .read "a" (rowOf l s).val, rowFields s!"a[{s.val}]" K, false⟩ : Port) ∈
      lanePorts (servedTile K) l.val := by
  unfold lanePorts
  apply List.mem_append_left
  apply List.mem_append_left
  refine List.mem_map.2 ⟨s.val, List.mem_range.2 s.isLt, ?_⟩
  try dsimp only
  rw [servedTile_rowOf] <;> rfl

/-- Lane `l`'s port `b<t>` is opened at row `colOf l t` of the registered `b`, and holds its B row's codes and scale. -/
theorem lanePorts_b (K : ℕ) (l : Fin 32) (t : Fin 8) :
    (⟨s!"b{t.val}", .read "b" (colOf l t).val, rowFields s!"b[{t.val}]" K, false⟩ : Port) ∈
      lanePorts (servedTile K) l.val := by
  unfold lanePorts
  apply List.mem_append_left
  apply List.mem_append_right
  refine List.mem_map.2 ⟨t.val, List.mem_range.2 t.isLt, ?_⟩
  try dsimp only
  rw [servedTile_colOf] <;> rfl

/-- Lane `l`'s last five ports: the admit bits and its public inputs at known content, the leaf's record at `zy`'s
row, its partials row at row `l` of `partials`, and its `ok` at `ok<l>`. -/
theorem lanePorts_tail (K : ℕ) (l : ℕ) :
    ∀ p ∈ ([⟨"admit", .known "admit", fieldsFrom 0 [("admit", 32)], false⟩,
      ⟨"zy", .read "zy" 0, fieldsFrom 0 [("zy.z", 32 * 32 * 32), ("zy.y", 16 * 32 * 32)], false⟩,
      ⟨"partials", .read "partials" l, fieldsFrom 0 [("partials", 32 * (servedTile K).nw)], false⟩,
      ⟨"pub", .known s!"pub{l}", fieldsFrom 0 [("salt", digestBits), ("position", positionBits),
        ("roots", 2 * digestBits), ("lane", laneBits)], false⟩,
      okPort l] : List Port), p ∈ lanePorts (servedTile K) l := by
  intro p hp
  unfold lanePorts
  exact List.mem_append_right _ hp

/-- The combine's port `p<x>` is opened at row `x` of the registered `partials`. -/
theorem combinePorts_p (K : ℕ) (x : Fin 32) :
    (⟨s!"p{x.val}", .read "partials" x.val, fieldsFrom 0 [(s!"partials[{x.val}]", 32 * (servedTile K).nw)], false⟩ :
      Port) ∈ combinePorts (servedTile K) := by
  unfold combinePorts
  apply List.mem_append_left
  exact List.mem_map.2 ⟨x.val, List.mem_range.2 (by rw [servedTile_lanes]; exact x.isLt), rfl⟩

/-- The combine's last two ports: the leaf's receipt at `receipt`'s row, and its `ok` at `ok32`. -/
theorem combinePorts_tail (K : ℕ) :
    ∀ p ∈ ([⟨"receipt", .read "receipt" 0, fieldsFrom 0 [("receipt", 32 * (servedTile K).nw)], false⟩, okPort 32] :
      List Port), p ∈ combinePorts (servedTile K) := by
  intro p hp
  unfold combinePorts
  rw [servedTile_lanes]
  exact List.mem_append_right _ hp

end Accounting.TS.Lanes
