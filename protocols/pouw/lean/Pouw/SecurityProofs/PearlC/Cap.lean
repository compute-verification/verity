import Pouw.SecurityProofs.PearlC.Cap.Grid
import Pouw.SecurityProofs.PearlC.Cap.Certs

/-!
# The cap circuit's Lean pieces

The pieces of `capTile_sound` (the cap circuit `Pc8TileCap` decides rev1's `capOK`) that don't wait on the circuit's
open rulings: the FP32 grid and a value's neighbours (`Cap.Grid`), and Freeze's certificates on the device's step with
the circuit's interval test (`Cap.Certs`).
-/
