import PouwWindow.Grid
import PouwWindow.Draw
import PouwWindow.Advice
import PouwWindow.Public
import PouwWindow.Json
import PouwWindow.Difftest

/-!
# PoUW's audit window as executable Lean

The port of `verity/experimental/protocols/pouw/window.py`: the grid (`PouwWindow.Grid`), the draw size K′ and the
`sampling` item (`PouwWindow.Draw`), the advice's tile map and the window statement L4 (`PouwWindow.Advice`), the public
items (`PouwWindow.Public`), their JSON (`PouwWindow.Json`) and the difftest against the Python (`PouwWindow.Difftest`).
Code: Lean core only, and never `Security/`. Its properties are stated in `Security/Properties/Pouw.lean`.
-/
