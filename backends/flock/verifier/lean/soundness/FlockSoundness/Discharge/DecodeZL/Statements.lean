import FlockSoundness.Discharge.DecodeZL.Wire
import FlockSoundness.Discharge.Exec.Rounds
import FlockSoundness.Discharge.Exec.Verdict
import FlockSoundness.Discharge.Exec.LiveOn
import FlockSoundness.Discharge.Integrate.Headline
import FlockSoundness.Protocol.Discharge.DecodeZL.Statements

/-!
# The message half of R11b, and the decoder at the headline: statements

`flock_headline_exec`'s open guarantee is `hdec : DecodesOn …`: at every outcome of the live session flock-verify
accepts, each rep's tape (its rounds' messages, decoded from their bytes, and coins) plays the rep's game, and the table's
verdict accepts. The statements here:

* `ZLCarries`: the zerocheck's and lincheck's rounds carry their messages by `wZL` (`Carries`, `Wire.lean`).
* `ProofSizes`, `DecodeSizes`: the lengths `decodeProof` checks, which make the message-carrying frames well formed.
* `OPCarries`, `LIGCarries`: the same chunk lemma for the opening's ring switches and for Ligerito, at a decoded
  proof of a `fast100` statement (owners: dec-open, dec-lig).
* `TagsOk`, `RecordsLive`, `DrawOk`: what `DecodesOn` needs past #828's hypotheses. `wr` is a free parameter of the
  headline, and at a `wr` constant at an accepted transcript `DecodesOn` is false (an outcome with no rounds can't play
  `repC`, which starts with a coin); `RecordsLive` says `wr` is the coin server's record of the outcome.
* `decAt`, `opnAt`: the decoder and the openings at draw `S`, read off the table `T S R` and setupH's statement there.
* `DecodesOnExec`: `DecodesOn` at #828's tables, the decoder `decAt`, `one := true`, no final messages, `capOfRoot`.
-/

namespace FlockSoundness.Discharge.DecodeZL

open Flock Refine Exec Game Model Accounting Audit
open FlockSoundness.Discharge.Integrate

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

variable {n mPts : ℕ} {Reg : Type}

end FlockSoundness.Discharge.DecodeZL
