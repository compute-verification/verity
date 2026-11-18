module

public import VStarRecursive.ChallengerPin
public import PouwVerifier.Period

/-!
# The challenger's key is the auditor's pin (`Spot.ChallengerPin`)

The pin is tested on the parameters the protocol itself parsed (`Spot.ChallengerPin.pinned`), the value its check is
handed, so the key it pins is the key the check is handed. That the check verifies its ledger, sessions or draw coin
under that key and no other is not stated here:

* `ChallengerKeyPinned`: what SPOT's executable accepts under a pin, SPOT parsed (`Sampled.paramsOfJson`, the
  protocol's `params`) to parameters whose every level names a challenger whose public key's SHA-512 is the pin's and
  which the key file doesn't retire: the keys its check is handed.
* `PeriodChallengerKeyPinned`: what PoUW's period accepts under a pin, the period parsed (`PouwVerifier.Period.paramsOf`)
  to parameters whose `key` is the pinned key, not retired: still JSON, the value its check is handed, which the credit
  parses with the same `ChallengerKey.ofJson` (`Credit.periodCredit`). No registered composite holds the period
  (a `#guard` in `Core.Composites`), so this is of the period's own executable (`pouw-period`) alone.
* `MembersPinned`: a composite under a pin is built over its members with SPOT replaced by SPOT under the pin
  (`VStar.Recursive.ChallengerPin.members`); each member answers its own case, so `ChallengerKeyPinned` holds of it. It
  pins the list's top level only: a member that were itself a composite holding SPOT under another id would be a member
  that is not SPOT. No registered composite nests one (`Core.Composites.composites`, held by a `#guard` there):
  `verity/onsite/v1` holds `verity/certifier/v0`, `verity/inference-only/v0`, `verity/pous/p2/v4/run`,
  `verity/pouw/pearl-plus/v1`, `verity/remote-bandwidth/p2/v0`, `verity/sampled-proofs/v1` and `verity/wipe/v0`;
  `verity/remote/v1` holds `verity/inference-only/v0`, `verity/pouw/pearl-plus/v1` and `verity/sampled-proofs/v1`.
  Of those, SPOT alone names the challenger's key in Lean: Pearl+'s run verdict (`PouwVerifier.PearlPlus.Protocol`)
  reads its live coins as a commitment's root and openings (`Spot.Challenger.LiveCommitment`), with no key, so no Lean
  check binds that root to the challenger's key.

The window's root needs no comparison of its own: the coins are read only from the ledger's open whose root is the
parameters' window root, in a ledger checked under the case's key (`Spot.Challenger.auditedOpen`), withheld on that root
(`Spot.Challenger.withheldOn`), so under the pinned key a root the challenger never opened is refused. The pin holds
only where the verifier taking it runs on the auditor's side, from the auditor's own checkout.
-/

@[expose] public section

namespace Properties.Spot.ChallengerPin

open VStar.Recursive

/-- What SPOT's executable under `pin` accepts, SPOT parsed to parameters naming, at every level, a challenger whose key
`pin` pins and its key file doesn't retire. -/
def ChallengerKeyPinned : Prop :=
  ∀ (pin : _root_.Spot.ChallengerPin.Pin) (params commitment evidence : ByteArray) (blobs : Array ByteArray),
    ((ChallengerPin.sampled pin).answer params commitment evidence blobs).verdict = .accept →
      ∃ j p, Core.canonicalJson params = .ok j ∧ Sampled.paramsOfJson j = .ok p ∧
        ∀ k ∈ ChallengerPin.levels p, ∃ key, k = some key ∧ _root_.Spot.sha512Hex key.pub = pin.sha512 ∧
          pin.retired.contains (_root_.Flock.hexOf key.pub) = false

/-- What PoUW's period under `pin` accepts, the period parsed to parameters whose `key`, the challenger's that signs the
draw coin, is the key `pin` pins and its key file doesn't retire. -/
def PeriodChallengerKeyPinned : Prop :=
  ∀ (pin : _root_.Spot.ChallengerPin.Pin) (params commitment evidence : ByteArray),
    ((PouwVerifier.Period.pinned pin).answer params commitment evidence).verdict = .accept →
      ∃ j p, Core.canonicalJson params = .ok j ∧ PouwVerifier.Period.paramsOf j = .ok p ∧
        ∃ key, _root_.Spot.Challenger.ChallengerKey.ofJson p.key = .ok key ∧
          _root_.Spot.sha512Hex key.pub = pin.sha512 ∧ pin.retired.contains (_root_.Flock.hexOf key.pub) = false

/-- A composite's members under `pin`: SPOT under `pin`, or a member that is not SPOT, at the list's top level. -/
def MembersPinned : Prop :=
  ∀ (pin : _root_.Spot.ChallengerPin.Pin) (cs : List Core.BlobProtocol),
    ∀ Q ∈ ChallengerPin.members pin cs, Q = ChallengerPin.sampled pin ∨ (Q ∈ cs ∧ Q.id ≠ Sampled.PROTOCOL)

end Properties.Spot.ChallengerPin
