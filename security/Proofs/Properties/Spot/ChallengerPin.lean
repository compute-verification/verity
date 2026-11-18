module

public import Properties.Spot.ChallengerPin

/-! # Proof that the challenger's pin pins the key each check is handed (`Properties.Spot.ChallengerPin`)

The pin is tested on the parameters the protocol parsed, after its own parse (`Spot.ChallengerPin.pinned`), so an accept
leaves the parsed parameters with nothing the pin names in them (`Spot.ChallengerPin.pinned_accept`).
-/

@[expose] public section

namespace Properties.Spot.ChallengerPin

theorem challengerKeyPinned : ChallengerKeyPinned := VStar.Recursive.ChallengerPin.sampled_accept

theorem periodChallengerKeyPinned : PeriodChallengerKeyPinned := PouwVerifier.Period.pinned_accept

theorem membersPinned : MembersPinned := VStar.Recursive.ChallengerPin.members_pinned

end Properties.Spot.ChallengerPin
