import Proofs.Accounting.TS.FanIn
import Proofs.Accounting.TS.FoldFanIn
import Proofs.Accounting.TS.LowBit
import Proofs.Accounting.TS.Rounds
import Proofs.Accounting.TS.Receipt
import Proofs.Accounting.TS.Tensor
import Proofs.Accounting.TS.ProofsSound
import Proofs.Accounting.TS.Fork
import Proofs.Accounting.TS.Drawn
import Proofs.Accounting.TS.Keyed
import Proofs.Pouw.RelativeSecurity

/-!
# The tensor–scalar model's proofs

Each guarantee of `Definitions.Accounting.Guarantees.TS` as `Accounting.SecurityProofs.TS.G`, one file per prover
group: the round lemmas (`TS/Rounds.lean`), T1 and Lemma 2 (`TS/Tensor.lean`), the fan-in count (`TS/FanIn.lean`), the
fan-in floor of a fold of binding steps and of duo's and fx1's runs (`TS/FoldFanIn.lean`), the low bit of a
one-register fold and of li (`TS/LowBit.lean`), the receipt (`TS/Receipt.lean`), and the proofs' soundness, binding and
draw (`TS/ProofsSound.lean`, `TS/Fork.lean`, `TS/Drawn.lean`, `TS/Keyed.lean`).
-/
