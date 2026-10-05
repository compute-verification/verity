import FlockSoundness.Discharge.PlaysCoin.Coins
import FlockSoundness.Discharge.PlaysCoin.Tape
import FlockSoundness.Protocol.Discharge.PlaysCoin.Decodes
import FlockSoundness.Discharge.PlaysCoin.Statements

/-!
# `Decodes` from the reps' runs

`playsT_of_repRun`: a rep's `RepRun` along a spine of its game is a run on the decoded tape. The run on two lists is a
run on one tape whose coin steps are the rounds' `coinMsg true` (`coinsOfRounds`). That tape's kinds are the spine's,
up to its length (`Spine.prefix`), so its messages cut at its coins are the `RepRun`'s groups, and the tape is the
decoded one (`tapeOfGroups`). So `decodesOfRuns`, `decodesAllOfRuns` and `decodesOnOfRuns`; the reps' spine is
`repShape` (`repC_spine`). `decodesOfRunsAt` reads the spine only along the runs that end where the `RepRun` ends
(`playsT_of_repRunAt`).
-/

namespace FlockSoundness.Discharge.PlaysCoin

open Flock FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge.Exec Model Accounting Audit
  Audit.Partition

local notation "𝔽" => FlockLevel3.GF128

theorem decodesAllOfRuns : DecodesAllOfRuns := by
  intro D Sl _ _ _ mPts n Reg H E Lf T I wr opn dec fin capOf h S R
  exact decodesOfRuns _ _ _ (opn S R) (dec S R) (fin S R) (capOf S R) _ (fun pts _ => repSpine D (T S R) pts)
    (repsOf_spine D (T S R)) (h S R)

end FlockSoundness.Discharge.PlaysCoin
