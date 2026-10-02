import FlockSoundness.Discharge.DecodeZL.Statements

/-!
# The zerocheck's and lincheck's rounds carry their messages (R11b)

`zl_carries`: from the domain frame through `z_partial`'s round, the rounds of `zcFrames ++ lcFrames`, read by `wZL`,
are `zcMsgs ++ lcMsgs`, cut at `zcShape ++ lcShape`'s coins.
-/

namespace FlockSoundness.Discharge.DecodeZL

theorem zl_carries : ZLCarries := by
  sorry

end FlockSoundness.Discharge.DecodeZL
