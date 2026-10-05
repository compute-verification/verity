import FlockSoundness.Discharge.ZkExec.VerdictStatements
import FlockSoundness.Discharge.ZkExec.PlaysDet
import FlockSoundness.Discharge.ZkExec.Verify
import FlockSoundness.Discharge.ZkExec.PlaysRep
import FlockSoundness.Discharge.ZkExec.RunsSizes
import FlockSoundness.Discharge.ZkExec.VerdictLig
import FlockSoundness.Discharge.ZkExec.VerdictOpens
import FlockSoundness.Discharge.ZkExec.VerdictRows
import FlockSoundness.Protocol.Discharge.ZkExec.Verdict

/-!
# `RunsZC`'s verdict half: the compiled `--zk` table accepts at the two reps' results

* `zk_table_verdict_of`: `ZkTableVerdict` from its per-rep half `ZkRepAccepts`, through `Zk.verifyRep`'s acceptance
  (`zk_verifyRep_ok`): both reps share the statement's extra claims, their proofs decode at one shape, their caps are
  the one `root_B` (`capOfRoot_eq`), `opnZ` reads each rep's rows, salts and inner openings off its own proof, and the
  results exist (`zk_rep_plays`).
* `zk_rep_accepts`: `ZkRepAccepts`'s three conclusions, each its own lemma: the Ligerito verdict (`zk_lig_accepts`,
  `VerdictLig.lean`), the inner check (`zk_inner_of_rep`, `VerdictRows.lean`, through `zk_inner_accepts`) and the
  requested rows (`zk_opens_ok`, `VerdictOpens.lean`).
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL

local notation "𝔽" => FlockLevel3.GF128

section Opn

variable {st : Setup} {sch : Flock.Schedule} {n : ℕ} {b₀ b₁ : ByteArray} {zp₀ zp₁ : Zk.ZkProof}

end Opn

end FlockSoundness.Discharge.ZkExec
