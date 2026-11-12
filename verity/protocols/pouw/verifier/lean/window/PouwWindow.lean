module

public import PouwWindow.Grid
public import PouwWindow.Draw
public import PouwWindow.Public
public import PouwWindow.Json
public import PouwWindow.Length
public import PouwWindow.Entry
public import PouwWindow.Record
public import PouwWindow.Side
public import PouwWindow.Statement
public import PouwWindow.Inputs
public import PouwWindow.Rounds
public import PouwWindow.Lanes
public import PouwWindow.KnownRows
public import PouwWindow.OwnPorts
public import PouwWindow.FullDraw
public import PouwWindow.RegFile

/-!
# PoUW's audit execution phase as executable Lean

The port of `verity/core/protocols/pouw/window.py`: the grid (`PouwWindow.Grid`), the sample size K′ and the
`sampling` item (`PouwWindow.Draw`), the public items (`PouwWindow.Public`) and their JSON (`PouwWindow.Json`), the rows' fixed length
(`PouwWindow.Length`), and the
execution phase's advice: the entry row and the maps from a sampled index to its rows and its -h3 word (`PouwWindow.Entry`), a call's
record (`PouwWindow.Record`), the side trees (`PouwWindow.Side`), the execution phase statement L4 (`PouwWindow.Statement`) and the
tile instances' public inputs (`PouwWindow.Inputs`), a live execution phase's round coins over the proof system's
challenger (`PouwWindow.Rounds`), and a sampled H1 leaf's 33 statements port by port under option A, each input's and
output's port, bits and registered row, and each input wire of the lowered unit's port and bit (`PouwWindow.Lanes`,
`h1zy.statement_ports`' layout and `class_statement.unit_lowering`'s wires, which proofs' decoders read), and the rows
the verifier registers at their known content (`PouwWindow.KnownRows`: lane `l`'s `pub` and `admit`, every statement's
`o0`), and its registered data ports, one tree per value at the layout's rows (`PouwWindow.OwnPorts`), with each
statement's `--registered` file built from, and checked against, the accepted registration and produced record
(`PouwWindow.RegFile`, which `pouw-leaf-own` runs), and each statement's session held to the full sample
(`PouwWindow.FullDraw`, which `pouw-leaf-own` runs too). Code: Lean core plus PoUW's commitments (`pouw_commit`: its
frame-v3-sha512 domains, Flock's SHA-512 and hm96, core's tagged identity), and never `Security/`. The difftest's functions (`PouwWindow.Difftest`, on the shared harness) sit
outside this root, so what imports the execution phase imports no harness. Its properties are stated in
`Security/Properties/Pouw.lean`.
-/

@[expose] public section
