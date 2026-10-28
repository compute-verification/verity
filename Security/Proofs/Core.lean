import Proofs.Core.Circuit
import Proofs.Core.Advice
import Proofs.Core.PrivatePartition
import Proofs.Core.Boolean
import Proofs.Core.Game
import Proofs.Core.Game.Basic
import Proofs.Core.Game.Interleave
import Proofs.Core.Game.Union
import Proofs.Core.Game.Prob
import Proofs.Core.Game.Poly
import Proofs.Core.Game.Expect
import Proofs.Core.Game.Runs
import Proofs.Core.Game.Seq
import Proofs.Core.Game.Lock
import Proofs.Core.Game.Batch
import Proofs.Core.Game.BatchInduced
import Proofs.Core.Game.Rewinding
import Proofs.Core.Game.Unambiguous
import Proofs.Core.UnionBound
import Proofs.Core.Scalar.Vectors
import Proofs.Core.Prims.Vectors
import Proofs.Core.Round
import Proofs.Core.Prims.Round
import Proofs.Core.TC
import Proofs.Core.Fp32
import Proofs.Core.Formalism
import Proofs.Core.Expand
import Proofs.Core.Padding
import Proofs.Core.AndSize
import Proofs.Core.Accel.Principles
import Proofs.Core.Accel.Float
import Proofs.Core.Accel.Decode
import Proofs.Core.Accel.Witness
import Proofs.Core.Accel.Sheet
import Proofs.Core.NaiveStrassen
import Proofs.Core.Bilinear
import Proofs.Core.HongKung
import Proofs.Core.Blocks

/-!
# Proofs of core's guarantees

One theorem `Verity.SecurityProofs.G : Verity.Guarantees.G` per proved guarantee, and every lemma behind them.
-/
