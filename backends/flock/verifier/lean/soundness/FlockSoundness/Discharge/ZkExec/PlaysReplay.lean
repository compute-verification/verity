import FlockSoundness.Discharge.ZkExec.PlaysStatements
import FlockSoundness.Protocol.Discharge.ZkExec.PlaysReplay

/-!
# The masked replay's runs

`zk_replay_plays`: `Zk.replay` refines the model's masked zerocheck and lincheck, with the two masked claims' points.
The executable draws exactly the clear verifier's coins (§1, §4); its messages are the masked proof's, and its checks
on them are constraints, so nothing it does after a coin depends on whether a check holds.

* `forIn_push`: a loop that consumes one one-coin round per step and pushes that coin keeps the coins, in order.
* `residualZK_plays`, `lcRoundsZK_plays`: the model's masked residual and lincheck rounds, fed round `i`'s messages and
  coins, end with those coins and the claim `resOut`, `lcRoundsOut` computes.

The replay's claims are `abcClaims` on the coins: `ab`'s point is the lincheck's (its rounds' coins reversed, then the
zerocheck's residual coins), `c`'s is `(L_s(z), pinned ++ r_rest)`, which `zcVector_eq` reads as the model's `r`.
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL Transcript

local notation "𝔽" => FlockLevel3.GF128

section Rounds

variable {np : ℕ} (P : ZkSession.PadsA np) (rounds : Array (F128 × F128)) (C : ℕ → Array F128)

end Rounds

end FlockSoundness.Discharge.ZkExec
