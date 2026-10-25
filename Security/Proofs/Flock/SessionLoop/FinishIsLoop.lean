import Proofs.Flock.SessionLoop.FinishIsRet
import Proofs.Flock.SessionLoop.FinRepTarget
import Proofs.Flock.SessionLoop.FinRepRestart
import Proofs.Flock.SessionLoop.FinRepL0

/-!
# `FinishIs` at `SessionLoopZKDeployed`'s premises

`finishIs_of_ret` takes three facts beyond `SessionLoopZKDeployed`'s premises: the ring switch's target
(`Loop.finRepTarget`), the restart (`Loop.finRepRestart`) and level 0's openings (`Loop.finRepL0`). Each holds in the
masked scope with a witness where the loop draws level 0 from the tape (`Loop.Src`'s level-0 half).

**`Loop.finishIs`**: `FinishIs` at `SessionLoopZKDeployed`'s premises, with no other hypothesis.
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

namespace Loop

variable (vs : finS.SwVals) (L₀ : LeanRelease Flock.Wit Tp)
  (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
  (wk : (LeanRelease.ofLoopLink L₀ src (realOf src)).toSrv.Worker) (x : Msg) (w : Flock.Wit)

/-- **`FinishIs` at `SessionLoopZKDeployed`'s premises**: in the masked scope, with a witness, at a statement the loop
serves whose inner round 0 is the loop's and whose inputs from its host are the tape's, at live rounds the loop's and
causal switched values, the `Finish` the loop runs is the session's (`finishIs_of_ret`, at `finRepTarget`,
`finRepRestart` and `finRepL0`). -/
theorem finishIs (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hServe : Serves x)
    (hL0 : _root_.Flock.Guarantees.LoopInnerRound0 Masked.dataE L₀ (realOf src) x w) (hSrc : Loop.Src L₀ src x w)
    (hE : LeanRelease.LiveRoundEq finS vs (LeanRelease.ofLoopLink L₀ src (realOf src)) CoinOpen.shaH wk x w)
    (hc : finS.SwCausal vs) :
    LeanRelease.FinishIs finS vs (LeanRelease.ofLoopLink L₀ src (realOf src)) CoinOpen.shaH wk x w :=
  finishIs_of_ret vs L₀ src wk x w hsc hR hServe hL0 hSrc hE hc
    (finRepTarget vs L₀ src (realOf src) wk x w hsc hR hServe hSrc.1)
    (finRepRestart vs L₀ src (realOf src) wk x w hsc hR hSrc.1)
    (finRepL0 vs L₀ src (realOf src) wk x w hsc hR hSrc.1)

end Loop

end Claims.ZKServer
