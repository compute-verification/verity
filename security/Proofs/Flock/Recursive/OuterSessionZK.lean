module

public import Proofs.Flock.Recursive.ZkServer
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Proofs.Flock.Recursive.ZkServerZK
import Proofs.Flock.Recursive.OuterZK
import Proofs.Flock.Soundness.CROnly.Hiding

/-!
# Recursion: the committed-coin statements' `_hOuter` at V*'s sessions

`RecursiveZK`, `RecursiveCircuitPrivate`, `RecursiveCircuitIndist` and `RecursiveCircuitPrivateLeaves` take the outer
session's zero knowledge as an obligation on the view and simulator they range over (`Flock.Guarantees.OuterViewZK`,
`Flock.Guarantees.OuterSessionsZK`). Here it is at V*'s outer `--zk` sessions as `Proofs.Flock.Recursive.Sessions`
models them, run by the honest prover's code (`proverExe`, whatever a stopped run sends), with the commitments the
ZK server sends when every message is fixed by the auditor's tape before the first commitment (the committed-coin case).

* `committed`, `zkServerCommits_committed`: messages that read none of the commit strings before them, at which the
  ZK server's commitments are `zkServerLeaf` of each message under its own salt.
* `recursiveZK_of_sessionZK`: `OuterSessionsZK` at `Sessions.view` and `Sessions.sim`, with `ε_o` V*'s sessions' error
  at the pinned key's hiding gap (`errAt`), from Flock's zero knowledge of one `--zk` table at every coin vector
  (`ZkEveryCoin.coin_close_le`, through `OuterZK.vstar_outer_zk`), on V*'s staging, each a named obligation:
  `SessionsPublic` and `SessionsHonest` at `committed msgs`, and `SessionsCoined`. V*'s sizes are `Sessions`' fields
  `hz`, `hR`, `hS`. `outerViewZK_of_sessionZK` is `RecursiveZK`'s `OuterViewZK` at one honest witness and registration.

What this does not cover: that the deployed outer sessions send `Sessions.view`, the honest prover's view at coins the
auditor's tape fixes before each session (`Sessions.coin`). `Loop` gives `OuterSessionsZK` at V*'s statements run by the
session loop's switched prover (`outerSessionsZK_loop`). Flock's session-loop bound (`SessionLoopZKAt`) is about the
deployed loop against every `V*`, but it bounds the real view by twice the simulated one plus its error, so it doesn't
give `OuterViewZK`'s additive form.
-/

@[expose] public section

namespace FlockSoundness.Discharge.Recursion.OuterZK

open Game
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- as in the ZK files this builds on; its own need is unmeasured

/-- **Messages fixed before the first commitment**: `msgs` read as live-coin messages that read none of the commit
strings before them. -/
def committed {Prog Wit Ωa Ωp : Type} {nr nh : ℕ}
    (msgs : Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → Ωp → Fin nh → ByteArray) :
    Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → Ωp → (i : Fin nh) → (Fin i → ByteArray) → ByteArray :=
  fun P w r a p i _ => msgs P w r a p i

/-- **The committed-coin case of the ZK server's commitments**: where no message reads the commit strings before it,
commitment `i` is `zkServerLeaf` of message `i` under its salt. -/
theorem zkServerCommits_committed {nh : ℕ} (m : Fin nh → ByteArray) (y : Fin nh → ZkView.Salt) :
    Flock.Guarantees.zkServerCommits (fun i _ => m i) y = fun i => Flock.Guarantees.zkServerLeaf (m i) (y i) := by
  funext i
  rw [Flock.Guarantees.zkServerCommits, ZkServerZK.zkServerPairs_eq, Flock.Guarantees.zkServerLeaf]

end FlockSoundness.Discharge.Recursion.OuterZK
