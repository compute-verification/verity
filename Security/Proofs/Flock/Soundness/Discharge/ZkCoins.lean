import Proofs.Flock.Soundness.Discharge.ZkCoins.Basic
import Proofs.Flock.Soundness.Discharge.ZkCoins.SessionDefs
import Proofs.Flock.Soundness.Discharge.ZkCoins.Statements
import Proofs.Flock.Soundness.Discharge.ZkCoins.Weights
import Proofs.Flock.Soundness.Discharge.ZkCoins.Lazy
import Proofs.Flock.Soundness.Discharge.ZkCoins.Leaf
import Proofs.Flock.Soundness.Discharge.ZkCoins.Hybrid
import Proofs.Flock.Soundness.Discharge.ZkCoins.Erase
import Proofs.Flock.Soundness.Discharge.ZkCoins.Reads
import Proofs.Flock.Soundness.Discharge.ZkCoins.Session

/-!
# The coin step in the `--zk` session (live/PROTOCOL.md §9, C-Flock ZK)

Under C-Flock ZK the verifier's coins are committed in the coin tree at Hello and opened round by round. `sessionZKC` is
the batched `--zk` session on those coins, and `session_coins_hm96` its soundness: table `j`'s bound against live coins,
`padLinkError + (padRepError + εInner)²`, plus `2^-257` for each committed round, under the session's fresh hm96 key.

The step is a hybrid over committed rounds (`committed_le`), not over the session's rounds: a committed prover's
success probability is at most the optimal prover's against live coins (`value`), plus each round's hiding gap. So the
square in the table bound, which `value_interleave_le` proves of two reps on independent live coins, is untouched.
-/
