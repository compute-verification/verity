import Proofs.Flock.Claims.Complete.Zk.Json.Runs
import Proofs.Flock.Claims.Complete.Zk.Json.Stmt
import Proofs.Flock.Claims.Complete.Zk.Count
import Proofs.Flock.Claims.Complete.Zk.RepRounds
import Proofs.Flock.Claims.Complete.Zk.Lanes
import Proofs.Flock.Claims.Complete.Zk.Consumed
import Proofs.Flock.Claims.Complete.Zk.Replay
import Proofs.Flock.Claims.Complete.Zk.RingSwitch
import Proofs.Flock.Claims.Complete.Zk.Extras

/-!
# Masked completeness from named premises

`complete_zk`: Flock's completeness at the deployed masked verifier at one table (`CompleteWith H (deployZk rel)
honestPZk (δcZk H)`), from `complete_zk_of_goals` with every goal proved so far discharged, so it rests on the record's
premises (`Claims.JsonSpec printJ canon` and `Claims.Out64`: `StmtOkZk` follows from the setup, `stmtOkZk`, and
`ProverOkZk` from `Out64`), the proof files' (`RepShapedZk`; the record's length is `readyZk`'s), the masked prover's
pieces returning (`ProverRunsZk`; the regions fit by `readyZk`), the extra claims' ring-switch algebra (`RsExtrasZk`),
and the goals not yet proved, each a named `Prop`: per rep the masked Ligerito (`LigeritoZk`, which gives level 0's
lane phase, `laneConstraints_goal`) and the inner proof (`Inner`). The masked replay passes
(`replay_goal`), so does the ring switch on its claims (`ringSwitchZk_goal`), and where the rep's steps return every
recorded round is consumed (`consumedZk_goal`).

`complete_zk_relation`: the same at the deployed relation (`deployZk Readable.relation`), where the extra claims'
algebra holds from `RelS` (`rsExtrasZk`), and `Out64` is `Claims.Complete`'s own hypothesis.
-/

namespace FlockComplete

open Flock FlockDeploy

/-- **Flock's masked verifier is complete** for the honest masked prover at the record's error, `δcZk H`, from named
premises. -/
theorem complete_zk (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hR : JsonRT.RepShapedZk H rel)
    (hPR : JsonRT.ProverRunsZk H rel) (hX : RsExtrasZk H rel) (hLg : LigeritoZk H rel) (hIn : Inner H rel)
    (hMR : MaskRank.MaskRankBound) :
    CompleteWith H (deployZk rel) honestPZk (δcZk H) :=
  have hS : JsonRT.StmtOkZk H rel := JsonRT.stmtOkZk S
  have hP : JsonRT.ProverOkZk H rel := JsonRT.proverOkZk h64 (JsonRT.repRoundsZk_goal H rel)
  have hD : RepDecodesZk H rel := JsonRT.repDecodesZk_goal S h64 hS hP hR
  have hRp : Replay H rel := replay_goal S h64 hS hP hR
  have hRS : RingSwitchZk H rel := ringSwitchZk_goal S h64 hS hP hR hX
  complete_zk_of_goals H rel hMR
    { session := JsonRT.sessionRunsZk_goal S hS hPR
      sends := sendCountZk_goal H rel
      linkFirst := linkFirstZk_goal H h64 rel
      finalFits := JsonRT.finalFitsZk_goal S h64 hS hP hR
      draw := JsonRT.drawMatchesZk_goal S h64 hS hP
      liveReads := JsonRT.liveReadsZk_goal S h64 hS hP
      canonical := JsonRT.canonicalZk_goal S h64 hS hP
      rootB := JsonRT.custodyRootBZk_goal S h64 hS hP
      publicsLink := JsonRT.custodyPublicsZk_goal S h64 hS hP
      points := JsonRT.custodyPointsZk_goal S h64 hS hP
      count := JsonRT.custodyCountZk_goal S h64 hS hP
      rounds := JsonRT.custodyRoundsZk_goal S h64 hS hP
      tables := tablesZk_goal H rel
      decodes := JsonRT.decodesZk_goal S h64 hS hP
      publics := JsonRT.publicsZk_goal S h64 hS hP
      schedules := schedulesZk_goal H rel
      proofCount := proofCountZk_goal H rel
      proofList := JsonRT.proofListZk_goal S h64 hS hP
      proofDigests := JsonRT.proofDigestsZk_goal S h64 hS hP
      repStream := JsonRT.repStreamZk_goal S h64 hS hP
      repExtras := JsonRT.repExtrasZk_goal S h64 hS hP
      repDecodes := hD
      repParams := JsonRT.repParamsZk_goal S h64 hS hP hR
      padsRoot := JsonRT.padsRoot_goal S h64 hS hP hR
      replay := hRp
      ringSwitch := hRS
      ligerito := hLg
      lanes := laneConstraints_goal hLg hD
      inner := hIn
      consumed := consumedZk_goal S h64 hS hP hRp hRS hLg hIn
      capCommitted := JsonRT.capCommittedZk_goal S h64 hS hP hR }

/-- **Flock's masked verifier is complete at the deployed relation** for the honest masked prover at the record's
error, `δcZk H`, from named premises. -/
theorem complete_zk_relation (H : Claims.Msg → Claims.Msg) (S : Claims.JsonSpec printJ Flock.canon)
    (hR : JsonRT.RepShapedZk H Readable.relation) (hPR : JsonRT.ProverRunsZk H Readable.relation)
    (hLg : LigeritoZk H Readable.relation) (hIn : Inner H Readable.relation) (hMR : MaskRank.MaskRankBound) :
    CompleteWith H (deployZk Readable.relation) honestPZk (δcZk H) := fun h64 =>
  complete_zk H Readable.relation S h64 hR hPR (rsExtrasZk H) hLg hIn hMR h64

end FlockComplete
