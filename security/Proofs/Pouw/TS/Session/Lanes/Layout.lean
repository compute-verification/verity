module

public import Proofs.Pouw.TS.Session.Lanes.Layer
public import PouwWindow.Lanes

/-! The ports of a leaf's 33 statements at the served 32 × 32 tile (`servedTile K`), opened at the rows the leaf's
values decode from. -/

@[expose] public section

namespace ResourceAccounting.TS.Lanes

open PouwWindow.Lanes

/-- The served leaf's tile: 32 × 32 cells at contraction `K`. -/
def servedTile (K : ℕ) : Tile := ⟨32, 32, K⟩

theorem servedTile_lanes (K : ℕ) : (servedTile K).lanes = 32 := by
  simp [servedTile, Tile.lanes, Tile.g, Tile.q]

theorem servedTile_rowOf (K : ℕ) (l : Fin 32) (s : Fin 4) : (servedTile K).rowOf l.val s.val = (rowOf l s).val := by
  first | rfl | simp [Tile.rowOf, Tile.g, Tile.q, servedTile, rowOf]

theorem servedTile_colOf (K : ℕ) (l : Fin 32) (t : Fin 8) : (servedTile K).colOf l.val t.val = (colOf l t).val := by
  first | rfl | simp [Tile.colOf, Tile.q, servedTile, colOf]

theorem lanePorts_a (K : ℕ) (l : Fin 32) (s : Fin 4) :
    (⟨s!"a{s.val}", .read "a" (rowOf l s).val, rowFields s!"a[{s.val}]" K, false⟩ : Port) ∈
      lanePorts (servedTile K) l.val := by
  unfold lanePorts
  apply List.mem_append_left
  apply List.mem_append_left
  refine List.mem_map.2 ⟨s.val, List.mem_range.2 s.isLt, ?_⟩
  try dsimp only
  rw [servedTile_rowOf] <;> rfl

theorem lanePorts_b (K : ℕ) (l : Fin 32) (t : Fin 8) :
    (⟨s!"b{t.val}", .read "b" (colOf l t).val, rowFields s!"b[{t.val}]" K, false⟩ : Port) ∈
      lanePorts (servedTile K) l.val := by
  unfold lanePorts
  apply List.mem_append_left
  apply List.mem_append_right
  refine List.mem_map.2 ⟨t.val, List.mem_range.2 t.isLt, ?_⟩
  try dsimp only
  rw [servedTile_colOf] <;> rfl

end ResourceAccounting.TS.Lanes
