module

public import Proofs.Pouw.Window.Copies
public import Proofs.Pouw.Window.Flock
public import Proofs.Pouw.Window.Served
import Mathlib.Combinatorics.Matroid.Init

/-!
# PoUW's window: the root of its proofs

Imports the window's three parts: the service's units and their copy-closed wrong sets (`Window.Copies`), the window's
calls from Flock's analysis (`Window.Flock`), and PoUW's reading of the window, its bound tied to what is served
(`Window.Served`).
-/

@[expose] public section
