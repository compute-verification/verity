import FlockSoundness.Discharge.ZkExec.PlaysRep
import FlockSoundness.Discharge.ZkExec.RunsCut
import FlockSoundness.Discharge.ZkExec.Inner
import FlockSoundness.Protocol.Discharge.ZkExec.PlaysRun

/-!
# A `--zk` rep's `RepRun`

`zk_rep_run`: `zk_rep_plays`'s run, cut by `zk_cut` along `repShapeZ`, is a `RepRun` for the decoder `decZ`. The cut's
size hypotheses come from the run itself (`repProg_sizes`: `Zk.ligerito`'s shape check on `ybar`, `verifyInner`'s
on `τ`'s commitment and `Y`), and `repShapeZ` is `shapeZE` at the executable's dimensions (`ZkTabSeam`).
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL Transcript

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

theorem zk_rep_run : ZkRepRun := by
  intro mPts Sl Hs T pts r st sch tPad extras bytes zp rounds dom tr hsch hdl hseam hlen hext hdec hroot hwf hret
    hrun hk
  obtain ⟨pend, hpl, hne⟩ := zk_rep_plays Hs T pts r st sch tPad extras zp rounds dom tr hsch hseam hlen hext hwf
    hrun hk
  obtain ⟨Tv, hl, hR⟩ := zk_rep_frames st sch tPad r.val extras zp rounds dom tr hsch hrun hk hret
  obtain ⟨hyb, htau, hy⟩ := repProg_sizes _ _ _ _ _ _ hrun
  have hcut := zk_cut st sch tPad extras bytes zp Tv dom rounds hsch hdl hdec hroot hyb htau hy hl hR
  have hσ : repShapeZ 𝔽 𝕂 Dig T (Model.extraClaims T.S T.ptLocal mPts pts).length =
      shapeZE st.m st.kLog (2 + extras.size) (schedOf sch) tPad (layZ st sch).len := by
    unfold repShapeZ shapeZE
    rw [← hlen, hseam.hm, hseam.hk, hseam.hsch, hseam.he, hseam.ht, hseam.hnp, hseam.hnP, hseam.hq,
      Nat.add_comm extras.size 2]
  refine ⟨pend, ⟨fun rd hrd => ?_, _, hpl, ?_⟩, hne⟩
  · obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hrd
    have hi' : i < rounds.size := by simpa using hi
    have := hwf i hi'
    rw [getElem!_pos rounds i hi'] at this
    simpa using this
  · rw [hσ]
    exact hcut

end FlockSoundness.Discharge.ZkExec
