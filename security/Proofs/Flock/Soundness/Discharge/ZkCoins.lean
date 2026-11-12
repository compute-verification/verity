module

public import Proofs.Flock.Soundness.Discharge.ZkCoins.Basic
public import Proofs.Flock.Soundness.Discharge.ZkCoins.SessionDefs
public import Proofs.Flock.Soundness.Discharge.ZkCoins.Statements
public import Proofs.Flock.Soundness.Discharge.ZkCoins.Weights
public import Proofs.Flock.Soundness.Discharge.ZkCoins.Lazy
public import Proofs.Flock.Soundness.Discharge.ZkCoins.Leaf
public import Proofs.Flock.Soundness.Discharge.ZkCoins.Hybrid
public import Proofs.Flock.Soundness.Discharge.ZkCoins.Erase
public import Proofs.Flock.Soundness.Discharge.ZkCoins.Reads
import Mathlib.Combinatorics.Matroid.Init

/-!
# The coin step in the `--zk` session (live/PROTOCOL.md §9, Flock ZK)

Under Flock ZK the verifier's coins are committed in the coin tree at Hello and opened round by round. `sessionZKC` is
the batched `--zk` session on those coins. Its soundness, table `j`'s bound against live coins,
`padLinkError + (padRepError + εInner)²`, plus `2^-257` for each committed round under the session's fresh hm96 key, is
stated (`SessionCoinsHm96`) and no theorem proves it. What is proved is the coin step (`committed_le`,
`committed_le_hm96`), and that `sessionZKC` sampled live has `sessionZK`'s value (`er_sessionZKC`).

The step is a hybrid over committed rounds (`committed_le`), not over the session's rounds: a committed prover's
success probability is at most the optimal prover's against live coins (`value`), plus each round's hiding gap. So the
square in the table bound, which `value_interleave_le` proves of two reps on independent live coins, is untouched.
-/

@[expose] public section
