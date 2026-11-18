module

public import VStarRecursive.ChallengerPin
public import PouwVerifier.Period

/-!
# The challenger's key is the auditor's pin (`Spot.ChallengerPin`)

The pin is tested on the parameters the protocol itself parsed (`Spot.ChallengerPin.pinned`), the value its check is
handed, so the key it pins is the key the check verifies under:

* `ChallengerKeyPinned`: what SPOT's executable accepts under a pin, SPOT parsed (`Sampled.paramsOfJson`, the
  protocol's `params`) to parameters whose every level names a challenger whose public key's SHA-512 is the pin's: the
  key its ledger and sessions are checked under.
* `PeriodChallengerKeyPinned`: what PoUW's period accepts under a pin, the period parsed (`PouwVerifier.Period.paramsOf`)
  to parameters whose `key` is the pinned key: the key its check verifies the draw coin under (`Credit.periodCredit`).
* `MembersPinned`: a composite under a pin is built over its members with SPOT and the period replaced by those two
  under the pin (`VStar.Recursive.ChallengerPin.members`); each member answers its own case, so the two above hold of
  each.

The window's root needs no comparison of its own: the coins are read only from the ledger's open whose root is the
parameters' window root, in a ledger checked under the case's key (`Spot.Challenger.auditedOpen`), withheld on that root
(`Spot.Challenger.withheldOn`), so under the pinned key a root the challenger never opened is refused. The pin holds
only where the verifier taking it runs on the auditor's side, from the auditor's own checkout.
-/

@[expose] public section

namespace Properties.Spot.ChallengerPin

open VStar.Recursive

/-- What SPOT's executable under `pin` accepts, SPOT parsed to parameters naming, at every level, a challenger whose key
`pin` pins. -/
def ChallengerKeyPinned : Prop :=
  ∀ (pin : _root_.Spot.ChallengerPin.Pin) (params commitment evidence : ByteArray) (blobs : Array ByteArray),
    ((ChallengerPin.sampled pin).answer params commitment evidence blobs).verdict = .accept →
      ∃ j p, Core.canonicalJson params = .ok j ∧ Sampled.paramsOfJson j = .ok p ∧
        ∀ k ∈ ChallengerPin.levels p, ∃ key, k = some key ∧ _root_.Spot.sha512Hex key.pub = pin.sha512

/-- What PoUW's period under `pin` accepts, the period parsed to parameters whose `key`, the challenger's that signs the
draw coin, is the key `pin` pins. -/
def PeriodChallengerKeyPinned : Prop :=
  ∀ (pin : _root_.Spot.ChallengerPin.Pin) (params commitment evidence : ByteArray),
    ((PouwVerifier.Period.pinned pin).answer params commitment evidence).verdict = .accept →
      ∃ j p, Core.canonicalJson params = .ok j ∧ PouwVerifier.Period.paramsOf j = .ok p ∧
        ∃ key, _root_.Spot.Challenger.ChallengerKey.ofJson p.key = .ok key ∧
          _root_.Spot.sha512Hex key.pub = pin.sha512

/-- A composite's members under `pin`, its period member given as `period` (the mains give the period under `pin`):
SPOT under `pin`, `period`, or a member that is neither. -/
def MembersPinned : Prop :=
  ∀ (pin : _root_.Spot.ChallengerPin.Pin) (period : Core.BlobProtocol) (cs : List Core.BlobProtocol),
    ∀ Q ∈ ChallengerPin.members pin period cs, Q = ChallengerPin.sampled pin ∨ Q = period ∨
        (Q ∈ cs ∧ Q.id ≠ Sampled.PROTOCOL ∧ Q.id ≠ _root_.Spot.ChallengerPin.PERIOD)

end Properties.Spot.ChallengerPin
