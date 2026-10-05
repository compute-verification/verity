import FlockSoundness.Discharge.DecodeZL.ZL
import FlockSoundness.Discharge.PlaysCoin
import FlockSoundness.Protocol.Discharge.DecodeZL.Runs

/-!
# A rep's run, from its rounds' frames (R11b)

plays-coin's `DecodesOnOfRuns` asks each rep for a `RepRun`: a run on its rounds' coins whose messages, cut at the spine's
coins, are the decoder's groups of the rounds. This file gives it from the executable's side:

* `coinPat_eq`: `Carries`' coin pattern is plays-coin's, so `cutOfPat` cuts a `Carries` back into its rounds.
* `roundGroups_eq`: on rounds that retained the encodings of `F`'s chunks, the decoder `decOf W`'s groups are
  `rdChunks W F`.
* `repRun_at`: at a table whose statement, schedule and link layout are `stmtOf`'s, `fast100`'s and `PT_LOCAL`, the rep
  run `accepts_verdict_hm96` gives, on rounds that retained `repFrames`' chunks (`accepts_rep_rounds`), is a `RepRun`
  for the decoder `wZL ++ (wOP ++ wLIG)`, given the opening's and Ligerito's chunk lemmas.
-/

namespace FlockSoundness.Discharge.DecodeZL

open Flock Refine Exec Game Model Accounting

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

theorem groupPat_eq (ns : List ℕ) (t : ℕ) : groupPat ns t = PlaysCoin.groupPat ns t := rfl

end FlockSoundness.Discharge.DecodeZL
