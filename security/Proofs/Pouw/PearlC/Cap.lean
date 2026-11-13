module

public import Proofs.Pouw.Fp8Atom.Fp32Proofs
public import Proofs.Pouw.PearlC.FreezeP
public import Proofs.Pouw.PearlC.Cap.IntForm
public import Proofs.Pouw.PearlC.Cap.SaltDeadStep
public import Proofs.Pouw.PearlC.Cap.SaltDead
import Mathlib.Algebra.Order.Floor.Semifield
import Mathlib.Tactic.Positivity.Finset
public import Proofs.Pouw.PearlC.Cap.Pins

/-!
# The cap circuit's Lean pieces

The pieces of `capTile_sound` (the cap circuit `Pc8TileCap` decides rev1's `capOK`) that don't wait on the circuit's
open rulings: the circuit's integer tests and their 32-bit words (`Cap.IntForm`), `_cap_constants` at pinned shapes
(`Cap.Pins`), and the salt-dead two-ends lemma that lets RowA decide `saltDeadP` from two evaluations of the forming
step (`Cap.SaltDead`, on the step's monotone read in `Cap.SaltDeadStep`).
-/

@[expose] public section
