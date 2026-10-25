import Definitions.Flock.Arith
import Definitions.Flock.Accounting.Schedule
import Definitions.Flock.Accounting.Bound
import Definitions.Flock.Accounting.Padded
import Definitions.Flock.Game
import Definitions.Flock.Statement
import Definitions.Flock.Piop
import Definitions.Flock.Ligerito
import Definitions.Flock.Sparse
import Definitions.Flock.Merkle
import Definitions.Flock.Circuit
import Definitions.Flock.Rewinding
import Definitions.Flock.Accounting.Fast100

/-!
# C-Flock's definitions

The vocabulary C-Flock's guarantees are stated in, extracted from `Proofs/Flock` without its proofs, bottom-up: each
module above says what it holds, and the definitions keep their names. A proof that a definition carries as a field
moves with it; every other lemma stays in `Proofs/Flock` and imports the definitions it is about.
-/
