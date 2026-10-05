import Verity.SecurityProofs.Circuit
import Verity.SecurityProofs.Boolean
import Verity.SecurityProofs.Game
import Verity.SecurityProofs.Game.Basic
import Verity.SecurityProofs.Game.Interleave
import Verity.SecurityProofs.Game.Union
import Verity.SecurityProofs.Game.Prob
import Verity.SecurityProofs.Game.Poly
import Verity.SecurityProofs.Game.Expect
import Verity.SecurityProofs.Game.Runs
import Verity.SecurityProofs.Game.Seq
import Verity.SecurityProofs.Game.Lock
import Verity.SecurityProofs.Game.Batch
import Verity.SecurityProofs.Game.BatchInduced
import Verity.SecurityProofs.Game.Rewinding
import Verity.SecurityProofs.Scalar.Vectors
import Verity.SecurityProofs.Prims.Vectors
import Verity.SecurityProofs.TC
import Verity.SecurityProofs.Round
import Verity.SecurityProofs.Fp32

/-!
# Proofs of core's guarantees

One theorem `Verity.SecurityProofs.G : Verity.Guarantees.G` per proved guarantee, and every lemma behind them.
-/
