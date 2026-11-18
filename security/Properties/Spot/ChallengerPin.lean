module

public import VStarRecursive.ChallengerPin

/-!
# The challenger's key is the auditor's pin (`VStar.Recursive.ChallengerPin`)

* `ChallengerKeyPinned`: what SPOT's executable accepts under a pin, every level of its parameters names a challenger
  whose public key's SHA-512 is the pin's.
* `CompositeChallengerKeyPinned`: what a composite accepts under a pin, its SPOT components' parameters are refused by
  no level (`ChallengerPin.compositeReason`).

The window's root needs no comparison of its own: the coins are read only from the ledger's open whose root is the
parameters' window root, in a ledger checked under the case's key (`Spot.Challenger.auditedOpen`), withheld on that root
(`Spot.Challenger.withheldOn`), so under the pinned key a root the challenger never opened is refused. The pin holds
only where the verifier taking it runs on the auditor's side, from the auditor's own checkout.
-/

@[expose] public section

namespace Properties.Spot.ChallengerPin

open VStar.Recursive

/-- What SPOT's executable under `pin` accepts names, at every level, a challenger whose key `pin` pins. -/
def ChallengerKeyPinned : Prop :=
  ∀ (pin : ChallengerPin.Pin) (params commitment evidence : ByteArray) (blobs : Array ByteArray),
    ((ChallengerPin.sampled pin).answer params commitment evidence blobs).verdict = .accept →
      ∃ p, (Core.canonicalJson params >>= Sampled.paramsOfJson) = .ok p ∧
        ∀ k ∈ ChallengerPin.levels p, ∃ key, k = some key ∧ Spot.sha512Hex key.pub = pin.sha512

/-- What a composite `P` under `pin` accepts has SPOT components that no level of `pin`'s check refuses. -/
def CompositeChallengerKeyPinned : Prop :=
  ∀ (pin : ChallengerPin.Pin) (P : Core.BlobProtocol) (params commitment evidence : ByteArray)
    (blobs : Array ByteArray),
    ((ChallengerPin.composite pin P).answer params commitment evidence blobs).verdict = .accept →
      ∃ j, Core.canonicalJson params = .ok j ∧ ChallengerPin.compositeReason pin j = none

end Properties.Spot.ChallengerPin
