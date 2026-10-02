import FlockSoundness.Discharge.ZkCoins.Basic
import FlockSoundness.Discharge.ZkCoins.SessionDefs
import FlockSoundness.Discharge.ZkCoins.Statements
import FlockSoundness.Discharge.ZkCoins.Weights
import FlockSoundness.Discharge.ZkCoins.Lazy
import FlockSoundness.Discharge.ZkCoins.Leaf
import FlockSoundness.Discharge.ZkCoins.Hybrid
import FlockSoundness.Discharge.ZkCoins.Erase
import FlockSoundness.Discharge.ZkCoins.Reads
import FlockSoundness.Discharge.ZkCoins.Session

/-!
# The coin step in the `--zk` session (live/PROTOCOL.md §9, M2)

Under M2 the verifier's coins are committed in the coin tree at Hello and opened round by round. `sessionZKC` is the
batched `--zk` session on those coins, and `session_coins_hm96` its soundness: table `j`'s bound against live coins,
`padLinkError + (padRepError + εInner)²`, plus `2^-257` for each committed round, under the session's fresh hm96 key.

The step is a hybrid over committed rounds (`committed_le`), not over the session's rounds: a committed prover's
success probability is at most the optimal prover's against live coins (`value`), plus each round's hiding gap. So the
square in the table bound, which `value_interleave_le` proves of two reps on independent live coins, is untouched.
-/
