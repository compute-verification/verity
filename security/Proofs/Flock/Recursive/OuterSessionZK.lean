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
# Recursion: the committed-coin case's commitments

When every message is fixed by the auditor's tape before the first commitment (the committed-coin case), the ZK
server's commitments are each message's own:

* `committed`, `zkServerCommits_committed`: messages that read none of the commit strings before them, at which the
  ZK server's commitments are `zkServerLeaf` of each message under its own salt.
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
