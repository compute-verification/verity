module

public import Proofs.Flock.SessionLoop.TakenVals
public import Proofs.Flock.Claims.ZK.FinishSwLig

/-! `Loop.L0Fixed`, the premise over `vs` that its level-0 values are the ZK server's (`LigLock.l0Clear`). -/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal
open Claims.ZKMal.CoinLayer Claims.ZKMal.Masked

namespace Loop

def L0Fixed (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : CoinLayer.Nonce) (τ : UE.Tape) (c : Fin (refK x) → FlockDeploy.Coin) (n : ℕ), ¬ Refused x c n →
    ∀ (s : FlockDeploy.Ins) (spec : _root_.Flock.SessionSpec), FlockDeploy.setupOf true x = some (s, spec, stsOf x) →
    ∀ (j : Fin (jOf x)) (r : Fin 2) (extras : Array _root_.Flock.Claim),
      (FlockDeploy.tabsOf (stsOf x))[j.val]!.extra (ptsE c spec) = .ok extras →
      ∀ k, (layE x j).zp 64 ≤ k → k < (layE x j).hab →
        vsWord vs x w ν c τ j r k =
          FlockComplete.LigLock.l0Clear (ligCoin c) (ligP0 x j r) (zkTab x j).sch (ligTab x w τ j)
            (ligBasis x c w τ j r extras) (layE x j) ((UE.tapeZk τ).extra j.val) r.val k

end Loop

end Claims.ZKServer
