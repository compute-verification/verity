import Pouw.SecurityProofs.PearlC.Cap.Grid
import Pouw.SecurityProofs.PearlC.Cap.Certs
import Pouw.SecurityProofs.PearlC.Cap.IntForm
import Pouw.SecurityProofs.PearlC.Cap.Verdict
import Pouw.SecurityProofs.PearlC.Cap.Pins

/-!
# The cap circuit's Lean pieces

The pieces of `capTile_sound` (the cap circuit `Pc8TileCap` decides rev1's `capOK`) that don't wait on the circuit's
open rulings: the FP32 grid and a value's neighbours (`Cap.Grid`), Freeze's certificates on the device's step with the
circuit's interval test (`Cap.Certs`), the circuit's integer tests and their 32-bit words (`Cap.IntForm`), rev1's
`capOK` as those tests on the tile's counts (`Cap.Verdict`), and `_cap_constants` at pinned shapes (`Cap.Pins`).
-/
