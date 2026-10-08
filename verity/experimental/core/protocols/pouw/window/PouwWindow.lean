import PouwWindow.Grid
import PouwWindow.Draw
import PouwWindow.Public
import PouwWindow.Json

/-!
# PoUW's audit window as executable Lean

The port of `verity/experimental/protocols/pouw/window.py`: the grid (`PouwWindow.Grid`), the draw size K′ and the
`sampling` item (`PouwWindow.Draw`), the public items (`PouwWindow.Public`) and their JSON (`PouwWindow.Json`). The
window statement on #1426's entry rows (`Entry.from_row`, `layout_problem`, `tile_read`, ...) is not ported yet. Code:
Lean core only, and never `Security/`. The difftest's functions (`PouwWindow.Difftest`, on the shared harness) sit
outside this root, so what imports the window imports no harness. Its properties are stated in
`Security/Properties/Pouw.lean`.
-/
