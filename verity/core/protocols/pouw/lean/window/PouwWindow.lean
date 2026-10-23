import PouwWindow.Grid
import PouwWindow.Draw
import PouwWindow.Public
import PouwWindow.Json
import PouwWindow.Length
import PouwWindow.Entry
import PouwWindow.Record
import PouwWindow.Side
import PouwWindow.Statement
import PouwWindow.Inputs
import PouwWindow.Rounds
import PouwWindow.Lanes
import PouwWindow.KnownRows
import PouwWindow.OwnPorts
import PouwWindow.FullDraw
import PouwWindow.RegFile

/-!
# PoUW's audit window as executable Lean

The port of `verity/core/protocols/pouw/window.py`: the grid (`PouwWindow.Grid`), the draw size K′ and the
`sampling` item (`PouwWindow.Draw`), the public items (`PouwWindow.Public`) and their JSON (`PouwWindow.Json`), the rows' fixed length
(`PouwWindow.Length`), and the
window's advice: the entry row and the maps from a drawn index to its rows and its -h3 word (`PouwWindow.Entry`), a call's
record (`PouwWindow.Record`), the side trees (`PouwWindow.Side`), the window statement L4 (`PouwWindow.Statement`) and the
tile instances' public inputs (`PouwWindow.Inputs`), a live window's round coins over the proof service's
challenger (`PouwWindow.Rounds`), and a drawn H1 leaf's 33 statements port by port under option A, each input's and
output's port, bits and registered row, and each input wire of the lowered unit's port and bit (`PouwWindow.Lanes`,
`h1zy.statement_ports`' layout and `class_statement.unit_lowering`'s wires, which proofs' decoders read), and the rows
the verifier registers at their known content (`PouwWindow.KnownRows`: lane `l`'s `pub` and `admit`, every statement's
`o0`), and its registered data ports, one tree per value at the layout's rows (`PouwWindow.OwnPorts`), with each
statement's `--registered` file built from, and checked against, the accepted registration and produced record
(`PouwWindow.RegFile`, which `pouw-leaf-own` runs), and each statement's session held to the full draw
(`PouwWindow.FullDraw`, which `pouw-leaf-own` runs too). Code: Lean core plus PoUW's commitments (`pouw_commit`: its
frame-v3-sha512 domains, C-Flock's SHA-512 and hm96, core's tagged identity), and never `Security/`. The difftest's functions (`PouwWindow.Difftest`, on the shared harness) sit
outside this root, so what imports the window imports no harness. Its properties are stated in
`Security/Properties/Pouw.lean`.
-/
