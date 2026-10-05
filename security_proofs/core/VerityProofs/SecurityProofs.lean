import VerityProofs.SecurityProofs.Circuit
import VerityProofs.SecurityProofs.Boolean
import VerityProofs.SecurityProofs.Game
import VerityProofs.SecurityProofs.Game.Basic
import VerityProofs.SecurityProofs.Game.Interleave
import VerityProofs.SecurityProofs.Game.Union
import VerityProofs.SecurityProofs.Game.Prob
import VerityProofs.SecurityProofs.Game.Poly
import VerityProofs.SecurityProofs.Game.Expect
import VerityProofs.SecurityProofs.Game.Runs
import VerityProofs.SecurityProofs.Game.Seq
import VerityProofs.SecurityProofs.Game.Lock
import VerityProofs.SecurityProofs.Game.Batch
import VerityProofs.SecurityProofs.Game.BatchInduced
import VerityProofs.SecurityProofs.Game.Rewinding
import Verity.SecurityProofs.Scalar.Vectors
import Verity.SecurityProofs.Prims.Vectors
import VerityProofs.SecurityProofs.TC
import VerityProofs.SecurityProofs.Fp32

/-!
# Proofs of core's guarantees

One theorem `Verity.SecurityProofs.G : Verity.Guarantees.G` per proved guarantee, and every lemma behind them.
-/
