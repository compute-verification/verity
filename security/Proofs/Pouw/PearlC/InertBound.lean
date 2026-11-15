module

public import Proofs.Pouw.Fp8Atom.Fp32Proofs
public import Proofs.Pouw.PearlC.FreezeP
public import Proofs.Pouw.PearlC.InertBound.IntForm
public import Proofs.Pouw.PearlC.InertBound.SaltDeadStep
public import Proofs.Pouw.PearlC.InertBound.SaltDead
import Mathlib.Algebra.Order.Floor.Semifield
import Mathlib.Tactic.Positivity.Finset
public import Proofs.Pouw.PearlC.InertBound.Pins

/-!
# The cap circuit's Lean pieces

The pieces of `inertBoundTile_sound` (the cap circuit `Pc8TileCap` decides rev1's `inertBoundOK`) that don't wait on the circuit's
open rulings: the circuit's integer tests and their 32-bit words (`InertBound.IntForm`), `_inert_bound_constants` at pinned shapes
(`InertBound.Pins`), and the salt-dead two-ends lemma that lets RowA decide `saltDeadP` from two evaluations of the forming
step (`InertBound.SaltDead`, on the step's monotone read in `InertBound.SaltDeadStep`).
-/

@[expose] public section
