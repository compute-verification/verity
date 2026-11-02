import Properties.CertifierDevice
import Properties.NetworkCertifier

/-!
# The certifier device's output as the network certificate check reads it

What the device returns over a trace (the records `step` publishes) and what the host's assembly reads of its
configuration to certify them (each record's direction, grid and Σ_sync, the signer's leaf reservation), the check's
copy of a certified record, and the check's parameters of the device's configuration, key and certifier session.
`Properties.CertifierNetwork.Runs` states what an accepted run of the device is over them;
`Proofs.Properties.CertifierNetwork.Defs` states, as completeness, that the device's own output is accepted.

**The unit is one link pair**, the device's: `active.py`'s `Proxy` runs one boundary link (`egressLink`, `ingressLink`),
and `service.LinkWindows` is "network accounting's protocol on one link".
-/

namespace Properties.CertifierNetwork

open _root_.CertifierDevice
open _root_.NetworkCertifier (SyncSet)
open Properties.CertifierDevice (outputs)

/-- The records an event publishes, in order. -/
def publishes (acts : List Action) : List BlockGrid :=
  acts.filterMap fun | .publish r _ => some r | _ => none

/-- Every record the device publishes over the trace, in order. -/
def published (c : Config) (tr : List Event) : List BlockGrid := (outputs (init c) tr).flatMap publishes

/-- The direction a record is certified in: egress on the egress link, ingress otherwise. -/
def directionOf (c : Config) (r : BlockGrid) : NetworkCheck.Direction := if r.link = c.egressLink then .egress else .ingress

/-- The grid a record is certified under. -/
def gridOf (c : Config) (r : BlockGrid) : Params := if r.link = c.egressLink then c.egress else c.ingress

/-- Σ_sync where a clock applies: on egress. -/
def syncSetOf (c : Config) (r : BlockGrid) : Option SyncSet := if r.link = c.egressLink then some c.syncSet else none

/-- The signer's reservation at its next unused leaf `next`: refused once every one of the `2^H` leaves is used. -/
def reserve (i : Issuer) (next : Nat) : Except String Nat :=
  if next < 2 ^ i.key.params.H then .ok next else .error s!"all 2^{i.key.params.H} leaves are used"

/-- The next unused leaf after a reservation at `next`. -/
def advance (i : Issuer) (next : Nat) : Nat := if next < 2 ^ i.key.params.H then next + 1 else next

/-- A certified record as the check reads it: the same head, salt and certificate. -/
def certified (cr : CertifiedRecord) : NetworkCheck.Check.Certified :=
  ⟨cr.head, cr.salt, ⟨cr.certificate.keyId, cr.certificate.link, cr.certificate.direction, cr.certificate.window,
    cr.certificate.digest, cr.certificate.signature⟩⟩

/-- The check's parameters of the device's configuration, key and certifier session: its two links, its `W` windows, its
two grids, its Σ_sync, the key's LMS parameters and public key, and the session's id. -/
def unitParams (c : Config) (key : Lms.PrivateKey) (session : ByteArray) : NetworkCheck.Check.Params where
  egress := #[c.egressLink]
  ingress := #[c.ingressLink]
  windows := c.windows
  grid := c.egress
  ingressGrid := c.ingress
  syncSet := c.syncSet
  lms := key.params
  key := key.publicKey
  session := session

end Properties.CertifierNetwork
