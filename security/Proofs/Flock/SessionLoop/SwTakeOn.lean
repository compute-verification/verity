module

public import Proofs.Flock.Claims.ZK.Blind
public import Proofs.Flock.Claims.ZK.FrameOuter

/-!
# The switched prover's messages at a live history's prefixes, at one-tape causality

`RefSession.swMsgAt_take` gives the switched prover's message after a prefix of a coin vector from the full
`SwCausal`. At `vsOf` only the one-tape form at `Loop.SwOk`'s coin vectors holds (`RefSession.SwCausalOn`), so this
file gives the message there from it:

* **`RefSession.swMsgAt_take_on`**: before the last coin, the message after a prefix is round `j`'s switched encoding
  at the coin vector itself, where `Ok` admits it and the padded prefix (`enc_swCutOf_on`, `swMsgAt_take_pre`).

At the loop, a live, unrefused history's opened coins and every prefix of them are `Loop.SwOk`'s
(`LeanRelease.swOk_of_live`, `LeanRelease.swOk_take_of_live`).
-/

@[expose] public section

namespace Claims.ZKMal

open Claims
open FlockSoundness.Discharge

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace RefSession

/-- **The switched prover's message after a prefix of a coin vector, before the last coin, at one-tape causality**:
round `j`'s switched encoding at the coin vector itself, where `Ok` admits it and the padded prefix (`swMsgAt_take` at
`SwCausalOn`). -/
theorem swMsgAt_take_on (S : RefSession) (vs : S.SwVals) (Ok : (x : Msg) → (Fin (refK x) → S.Coin) → Prop)
    (x : Msg) (w : Flock.Wit) (ν : S.Nonce) (hc : S.SwCausalOn Ok vs) (t : S.Tape) (c : Fin (refK x) → S.Coin)
    (j : ℕ) (hj : j < refK x) (hO : Ok x (S.pad ((List.ofFn c).take j))) (hO' : Ok x c) :
    S.swMsgAt x (S.wt x w) (vs x w ν) t ((List.ofFn c).take j) =
      S.enc x c j (S.swCutOf x c (S.wt x w) (vs x w ν c) t) := by
  rw [S.swMsgAt_take_pre x (S.wt x w) (vs x w ν) t c j hj]
  exact S.enc_swCutOf_on vs Ok x w ν hc t _ c j (fun k hk => S.pad_take c j k hk) hO hO'

end RefSession

end Claims.ZKMal
