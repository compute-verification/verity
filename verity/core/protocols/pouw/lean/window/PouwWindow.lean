import PouwWindow.Grid
import PouwWindow.Draw
import PouwWindow.Public
import PouwWindow.Json
import PouwWindow.Entry
import PouwWindow.Record
import PouwWindow.Side
import PouwWindow.Statement
import PouwWindow.Inputs

/-!
# PoUW's audit window as executable Lean

The port of `verity/core/protocols/pouw/window.py`: the grid (`PouwWindow.Grid`), the draw size K′ and the
`sampling` item (`PouwWindow.Draw`), the public items (`PouwWindow.Public`) and their JSON (`PouwWindow.Json`), and the
window's advice: the entry row and the maps from a drawn index to its rows and its -h3 word (`PouwWindow.Entry`), a call's
record (`PouwWindow.Record`), the side trees (`PouwWindow.Side`), the window statement L4 (`PouwWindow.Statement`) and the
tile instances' public inputs (`PouwWindow.Inputs`). Code: Lean core plus PoUW's commitments (`pouw_commit`: its
frame-v3-sha512 domains, C-Flock's SHA-512 and hm96, core's tagged identity), and never `Security/`. The difftest's functions (`PouwWindow.Difftest`, on the shared harness) sit
outside this root, so what imports the window imports no harness. Its properties are stated in
`Security/Properties/Pouw.lean`.
-/
