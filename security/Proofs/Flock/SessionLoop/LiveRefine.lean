module

public import Proofs.Flock.SessionLoop.Statement
public import Proofs.Flock.Claims.ZK.FrameZK

/-! `LiveRefine` at the session loop's own frames, at every worker from `LiveRoundEqOwn` alone (`LeanRelease.liveRefine_ofLoopLink`). -/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

namespace LeanRelease

theorem liveRefine_ofLoop (T : CoinLayer.TableData) (vs : (CoinLayer.session Loop.loopNr T).SwVals)
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.ZkServer.Finish.Tab × Array ByteArray))
    (wk : (ofLoop T L₀ real).toSrv.Worker) (x : Msg) (w : Flock.Wit)
    (hE : LiveRoundEqOwn (CoinLayer.session Loop.loopNr T) vs (ofLoop T L₀ real) CoinOpen.shaH wk x w) :
    (ofLoop T L₀ real).LiveRefine CoinOpen.shaH
      (((CoinLayer.session Loop.loopNr T).withStops
        (rfOf (CoinLayer.session Loop.loopNr T) (ofLoop T L₀ real) Loop.progOf Loop.freshOf Loop.readCoins)
        (ofLoop T L₀ real).stop).swP vs).prover wk x w :=
  have hC := checkIs_coinLayer Loop.loopNr T (ofLoop T L₀ real) x
    (loop_helloParses Loop.loopNr T (withFrames T L₀ real) x) (loop_reqsFollow T (withFrames T L₀ real) x)
  liveRefine_withStops (CoinLayer.session Loop.loopNr T) _ vs (ofLoop T L₀ real) CoinOpen.shaH wk x w
    (loop_last Loop.loopNr T (withFrames T L₀ real) x)
    (stopsAt_of_readsCoins _ (ofLoop T L₀ real) CoinOpen.shaH Loop.progOf Loop.freshOf Loop.readCoins x
      ⟨ofLoop_progReads T L₀ real x, loop_coinsRead Loop.loopNr T (withFrames T L₀ real) x⟩ hC)
    (openingsRefused_of_checkIs _ (ofLoop T L₀ real) CoinOpen.shaH x hC) hE

theorem liveRefine_ofLoopLink (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array _root_.Flock.F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (_root_.Flock.ZkServer.Finish.Tab × Array ByteArray))
    (wk : (ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit)
    (hE : LiveRoundEqOwn (CoinLayer.session Loop.loopNr Masked.dataE) vs (ofLoopLink L₀ src real) CoinOpen.shaH wk x w) :
    (ofLoopLink L₀ src real).LiveRefine CoinOpen.shaH
      (((CoinLayer.session Loop.loopNr Masked.dataE).withStops
        (rfOf (CoinLayer.session Loop.loopNr Masked.dataE) (ofLoopLink L₀ src real) Loop.progOf Loop.freshOf
          Loop.readCoins) (ofLoopLink L₀ src real).stop).swP vs).prover wk x w :=
  liveRefine_ofLoop Masked.dataE vs (withFiles (withLink L₀ src) src) real wk x w hE

end LeanRelease

end Claims.ZKServer
