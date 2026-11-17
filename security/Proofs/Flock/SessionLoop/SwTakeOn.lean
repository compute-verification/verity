module

public import Proofs.Flock.Claims.ZK.Blind
public import Proofs.Flock.Claims.ZK.FrameOuter

/-! The switched prover's message after a prefix of a coin vector, at one-tape causality (`RefSession.swMsgAt_take_on`). -/

@[expose] public section

namespace Claims.ZKMal

open Claims
open FlockSoundness.Discharge

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace RefSession

theorem swMsgAt_take_on (S : RefSession) (vs : S.SwVals) (Ok : (x : Msg) → (Fin (refK x) → S.Coin) → Prop)
    (x : Msg) (w : Flock.Wit) (ν : S.Nonce) (hc : S.SwCausalOn Ok vs) (t : S.Tape) (c : Fin (refK x) → S.Coin)
    (j : ℕ) (hj : j < refK x) (hO : Ok x (S.pad ((List.ofFn c).take j))) (hO' : Ok x c) :
    S.swMsgAt x (S.wt x w) (vs x w ν) t ((List.ofFn c).take j) =
      S.enc x c j (S.swCutOf x c (S.wt x w) (vs x w ν c) t) := by
  rw [S.swMsgAt_take_pre x (S.wt x w) (vs x w ν) t c j hj]
  exact S.enc_swCutOf_on vs Ok x w ν hc t _ c j (fun k hk => S.pad_take c j k hk) hO hO'

end RefSession

end Claims.ZKMal
