import Proofs.Flock.Claims.Complete.Zk.Json.Runs
import Proofs.Flock.Claims.Complete.Zk.Count
import Proofs.Flock.Claims.Complete.Zk.RepRounds

/-!
# Masked completeness from named premises

`complete_zk`: Flock's completeness at the deployed masked verifier (`Claims.Flock.Complete H (deployZk rel) honestPZk
(δcZk H)`), from `complete_zk_of_goals` with every goal proved so far discharged, so it rests on the record's
premises (`JsonSpec`, `Out64`, `StmtOkZk`; `ProverOkZk` follows from `Out64`), the proof files' (`RepShapedZk`,
`RecordFitsZk`), the masked prover's pieces returning (`ProverRunsZk`), and the goals not yet proved, each a named
`Prop`: per rep the masked replay (`Replay`), the ring switch (`RingSwitchZk`), the masked Ligerito (`LigeritoZk`),
level 0's lane phase (`LaneConstraints`), the inner proof (`Inner`), and every round consumed (`ConsumedZk`).
-/

namespace FlockComplete

open Flock FlockDeploy

/-- **Flock's masked verifier is complete** for the honest masked prover with error `δcZk H`, from named premises. -/
theorem complete_zk (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (S : JsonRT.JsonSpec) (h64 : JsonRT.Out64 H) (hS : JsonRT.StmtOkZk H rel)
    (hR : JsonRT.RepShapedZk H rel) (hF : JsonRT.RecordFitsZk H rel) (hPR : JsonRT.ProverRunsZk H rel)
    (hRp : Replay H rel) (hRS : RingSwitchZk H rel) (hLg : LigeritoZk H rel)
    (hLC : LaneConstraints H rel) (hIn : Inner H rel) (hCo : ConsumedZk H rel) :
    Claims.Flock.Complete H (deployZk rel) honestPZk (δcZk H) :=
  have hP : JsonRT.ProverOkZk H rel := JsonRT.proverOkZk h64 (JsonRT.repRoundsZk_goal H rel)
  complete_zk_of_goals H rel
    { session := JsonRT.sessionRunsZk_goal S hS hPR
      sends := sendCountZk_goal H rel
      linkFirst := linkFirstZk_goal H h64 rel
      finalFits := JsonRT.finalFitsZk_goal S h64 hS hP hR hF
      draw := JsonRT.drawMatchesZk_goal S h64 hS hP
      liveReads := JsonRT.liveReadsZk_goal S h64 hS hP
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
      repDecodes := JsonRT.repDecodesZk_goal S h64 hS hP hR
      repParams := JsonRT.repParamsZk_goal S h64 hS hP hR
      padsRoot := JsonRT.padsRoot_goal S h64 hS hP hR
      replay := hRp
      ringSwitch := hRS
      ligerito := hLg
      lanes := hLC
      inner := hIn
      consumed := hCo
      capCommitted := JsonRT.capCommittedZk_goal S h64 hS hP hR }

end FlockComplete
