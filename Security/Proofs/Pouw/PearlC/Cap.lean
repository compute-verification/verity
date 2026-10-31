import Proofs.Pouw.Fp8Atom.Fp32Proofs
import Proofs.Pouw.PearlC.FreezeP
import Proofs.Pouw.PearlC.Cap.IntForm
import Proofs.Pouw.PearlC.Cap.Pins
import Proofs.Pouw.PearlC.Cap.SaltDeadStep
import Proofs.Pouw.PearlC.Cap.SaltDead

/-!
# The cap circuit's Lean pieces

The pieces of `capTile_sound` (the cap circuit `Pc8TileCap` decides rev1's `capOK`) that don't wait on the circuit's
open rulings: the circuit's integer tests and their 32-bit words (`Cap.IntForm`), `_cap_constants` at pinned shapes
(`Cap.Pins`), and the salt-dead two-ends lemma that lets RowA decide `saltDeadP` from two evaluations of the forming
step (`Cap.SaltDead`, on the step's monotone read in `Cap.SaltDeadStep`).
-/
