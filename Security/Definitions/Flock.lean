import Definitions.Flock.Arith
import Definitions.Flock.Accounting.Schedule
import Definitions.Flock.Accounting.Bound
import Definitions.Flock.Accounting.Padded
import Definitions.Flock.Game
import Definitions.Flock.Statement
import Definitions.Flock.Piop
import Definitions.Flock.Ligerito
import Definitions.Flock.Sparse

/-!
# C-Flock's definitions

The vocabulary C-Flock's guarantees are stated in, extracted from `Proofs/Flock` without its proofs: the arithmetic the
verifier uses (`Arith`, the functions a field instance supplies), and the `fast100` Ligerito schedule with its soundness
bound as a function of the parameters, base and with a padded level 0 (`Accounting`), the public-coin interaction
model with live coins (`Game`), and the verifier as `Game` programs: the block R1CS statement and the committed values
(`Statement`), one rep's zerocheck, lincheck and opening (`Piop`), Ligerito (`Ligerito`) and the sparse statements
(`Sparse`).
-/
