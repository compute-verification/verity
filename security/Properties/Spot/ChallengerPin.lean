module

public import VStarRecursive.ChallengerPin
public import Spot.ChallengerPin

/-!
# The challenger's key is the auditor's pin (`VStar.Recursive.ChallengerPin`)

* `ChallengerKeyPinned`: what SPOT's executable accepts under a pin, every level of its parameters names a challenger
  whose public key's SHA-512 is the pin's.
* `CompositeChallengerKeyPinned`: what a composite accepts under a pin, its parameters list its components, every SPOT
  component's parameters name, at every level, a challenger whose key is the pinned one, and every PoUW period
  component's `key` is the pinned key: the draw coin's key is the challenger's, the same one pin. Which components a
  composite holds is its own check's, not the pin's.

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
  ∀ (pin : Spot.ChallengerPin.Pin) (params commitment evidence : ByteArray) (blobs : Array ByteArray),
    ((ChallengerPin.sampled pin).answer params commitment evidence blobs).verdict = .accept →
      ∃ p, (Core.canonicalJson params >>= Sampled.paramsOfJson) = .ok p ∧
        ∀ k ∈ ChallengerPin.levels p, ∃ key, k = some key ∧ Spot.sha512Hex key.pub = pin.sha512

/-- What a composite `P` under `pin` accepts lists its components; each SPOT component's parameters name, at every
level, a challenger whose key `pin` pins, and each PoUW period component's `key`, the challenger's that signs the draw
coin, is the one `pin` pins. -/
def CompositeChallengerKeyPinned : Prop :=
  ∀ (pin : Spot.ChallengerPin.Pin) (P : Core.BlobProtocol) (params commitment evidence : ByteArray)
    (blobs : Array ByteArray),
    ((ChallengerPin.composite pin P).answer params commitment evidence blobs).verdict = .accept →
      ∃ j xs, Core.canonicalJson params = .ok j ∧ Spot.getD j "components" = .arr xs ∧ ∀ x ∈ xs.toList,
        (Spot.getD x "id" = Lean.Json.str Sampled.PROTOCOL →
          ∃ p, Sampled.paramsOfJson (Spot.getD x "params") = .ok p ∧
            ∀ k ∈ ChallengerPin.levels p, ∃ key, k = some key ∧ Spot.sha512Hex key.pub = pin.sha512) ∧
        (Spot.getD x "id" = Lean.Json.str Spot.ChallengerPin.PERIOD →
          ∃ key, Spot.Challenger.ChallengerKey.ofJson (Spot.getD (Spot.getD x "params") "key") = .ok key ∧
            Spot.sha512Hex key.pub = pin.sha512)

end Properties.Spot.ChallengerPin
