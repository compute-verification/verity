module

public import Proofs.Flock.SessionLoop.TakenVals
public import Proofs.Flock.Claims.ZK.FinishSwLig

/-!
# `vs`'s level 0 is the ZK server's

The ZK server computes level 0's opening itself: from `zp 64` to `h_ab` (the out-of-domain value, each lane message and
the pair's sums) no round frees a word, the frame passes only the scheduled body, and the release is level 0's clear
values (`LigLock.l0Clear`) on the loop's packed table at the opened coins plus the pads. The switched session sends
`vs`'s values there under the same pads (`Loop.swSrcE_masked_vsWord`), so a round of level 0 is the switched message
only where `vs`'s values there are `l0Clear`. This file states that identity on the switched session's values, a
premise over `vs` beside `Loop.TakesVals`, `Loop.RowsOK` and `Loop.FinalCFixed`:

* **`Loop.L0Fixed`**: at every nonce, tape and opened coins not refused by some round, at each table and rep of the
  session's setup and the table's extra claims at the link's points, `vs`'s value at each index from `zp 64` to `h_ab`
  (`vsWord`) is `l0Clear` at the coins from the rep's PIOP's end (`ligP0`), the tape's packed table (`ligTab`), the
  basis of the rep's switched vectors at the coins (`ligBasis`) and the tape's extra lanes.
-/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal
open Claims.ZKMal.CoinLayer Claims.ZKMal.Masked

namespace Loop

/-- **`vs`'s level-0 values are the ZK server's**: at every nonce, tape and opened coins not refused by round `n`, the
session's setup, table `j`, rep `r` and the table's extra claims at the link's points (`SimLig`'s `LigEach` form),
`vs`'s value at each index from `zp 64` to `h_ab` is level 0's clear value (`l0Clear`) at the coins from `ligP0` on, the
tape's packed table, the basis of the rep's switched vectors at the coins and the tape's extra lanes, as the ZK server
computes it. -/
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
