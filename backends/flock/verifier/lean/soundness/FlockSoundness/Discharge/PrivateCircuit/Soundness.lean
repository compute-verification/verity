import FlockSoundness.Discharge.PrivateCircuit.Statements

/-!
# The session ran the registered program, under `cr/sha-512`

`ran_registered`: the union of protocol 1's bad event and a program other than the registered one; on the second, the
session's opening and the registrant's are two openings of one root to two programs, so the extractor's pair collides,
which `SHA512CRStrict` bounds by `q²/2^513`.
-/

namespace FlockSoundness.Discharge.PrivateCircuit

open FlockSoundness.Game
open scoped ENNReal

theorem ran_registered : RanRegistered := by
  sorry

end FlockSoundness.Discharge.PrivateCircuit
