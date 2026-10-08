import Properties.CertifierDevice
import Properties.NetworkCertifier

/-!
# The certifier device's own output is accepted by the network certificate check

`Properties.CertifierDevice` states what the device's `step` does over a trace; `Properties.NetworkCertifier` what the
check (`NetworkCheck.Check.protocol`, `verity/network-certificate/v2`) accepts. This module joins them: for a run of the
device on one boundary link, the commitment its host assembles from the records the device publishes and the evidence
the device's own shaper holds (the frames it handed and the syncs it declared) are accepted under the parameters of the
device's configuration (`DeviceOutputAccepted`, proved as `deviceOutputAccepted` in
`Properties.Proofs.CertifierNetwork`).

**The unit is one link pair**, the device's: `active.py`'s `Proxy` runs one boundary link (`egressLink`, `ingressLink`),
and `service.LinkWindows` is "network accounting's protocol on one link". A unit of several devices is several such
runs whose commitments and evidence are concatenated; it needs the devices to share one clock (the check reads every
egress link's sync against the first's), which one device's run says nothing about, so it is not stated here.

**The assembly** (`certifyAll`) is the host's, as `CertifierDevice/SmokeMain.lean` runs it and as `benchmarks/
network_traces/replay.py`'s `certify` does per direction: every record `step` publishes, in order, is certified by
`Issuer.certify` under its link's direction, grid and `parameters_digest` (with Σ_sync on egress), at the leaf the
signer's state reserves next (`lms.LeafState.reserve`: leaves `next, next + 1, …`, refused from `2^H` on, a refusal
leaving no certificate), and the certificates issued, in order, are the commitment's records (`service.encode` of each,
concatenated; the check reads the records, never the bytes, so the statement holds for every `bytes`). A Python
`ValueError` is an `Except.error` and aborts the assembly, as it aborts the host. The two record types are one in
Python (`certificate.CertifiedRecord`); `certified` copies the Lean device's into the check's field by field.

**The hypotheses** are each over the configuration, the trace or the device's outputs, and each is checkable on them:

* the configuration's: `Ticking` (both grids have buckets), `FieldsFit` (what a record encodes fits its field), `Distinct`
  (two links), `Digested` (each grid has a `parameters_digest`: `Flock.canon` is a `partial def`, so this is a fact
  about the configuration that only running it shows), and `ValidKey` of the issuer's key;
* the run's: `Closed` (both grids ran all their ticks), `HonestTiming` (`CompleteRows`'s: each frame handed at the bucket
  the Program hands it), `NoFault` (every record published is complete), `SyncsDeclared` (one sync per window) and
  `SyncsAdmitted` (each in Σ_sync), and `LeavesLeft` (the signer has a leaf for every record).
-/

namespace Properties.CertifierNetwork

open _root_.CertifierDevice
open _root_.NetworkCertifier (LogicalFrame SyncSet)
open Properties.CertifierDevice (outputs after handed ValidKey)

/-! ## The device's output -/

/-- The records an event publishes, in order. -/
def publishes (acts : List Action) : List Record :=
  acts.filterMap fun | .publish r _ => some r | _ => none

/-- Every record the device publishes over the trace, in order. -/
def published (c : Config) (tr : List Event) : List Record := (outputs (init c) tr).flatMap publishes

/-- The frames the shaper handed over the trace, in canonical order, as the Program's logical frames: their bits
(`bitsOf`), anchors and indices. -/
def openedFrames (c : Config) (tr : List Event) : List LogicalFrame :=
  (handed (init c) tr).map fun x => ⟨bitsOf x.1.bytes, x.1.anchor, x.1.index⟩

/-- The syncs the shaper declared over the trace, one per window. -/
def declaredSyncs (c : Config) (tr : List Event) : Array NetworkCertifier.ClockSync := (after (init c) tr).shaper.syncs

/-! ## The commitment, the evidence and the parameters -/

/-- The direction a record is certified in: egress on the egress link, ingress otherwise. -/
def directionOf (c : Config) (r : Record) : NetworkCheck.Direction := if r.link = c.egressLink then .egress else .ingress

/-- The grid a record is certified under. -/
def gridOf (c : Config) (r : Record) : Params := if r.link = c.egressLink then c.egress else c.ingress

/-- Σ_sync where a clock applies: on egress. -/
def syncSetOf (c : Config) (r : Record) : Option SyncSet := if r.link = c.egressLink then some c.syncSet else none

/-- The signer's reservation at its next unused leaf `next`: refused once every one of the `2^H` leaves is used. -/
def reserve (i : Issuer) (next : Nat) : Except String Nat :=
  if next < 2 ^ i.key.params.H then .ok next else .error s!"all 2^{i.key.params.H} leaves are used"

/-- The next unused leaf after a reservation at `next`. -/
def advance (i : Issuer) (next : Nat) : Nat := if next < 2 ^ i.key.params.H then next + 1 else next

/-- **The host's assembly**: each record, in order, certified by `Issuer.certify` at the reserved leaf under its link's
direction, grid and parameters digest; the certificates issued, in order. -/
def certifyAll (c : Config) : Issuer → Nat → List Record → Except String (Issuer × List CertifiedRecord)
  | i, _, [] => pure (i, [])
  | i, next, r :: rs => do
    let (i', cr?) ← i.certify (reserve i next) r (directionOf c r) (gridOf c r)
      (← parameters (gridOf c r) (syncSetOf c r))
    let (i'', crs) ← certifyAll c i' (advance i next) rs
    pure (i'', cr?.toList ++ crs)

/-- A certified record as the check reads it: the same head and certificate. -/
def certified (cr : CertifiedRecord) : NetworkCheck.Check.Certified :=
  ⟨cr.head, ⟨cr.certificate.keyId, cr.certificate.link, cr.certificate.direction, cr.certificate.window,
    cr.certificate.digest, cr.certificate.signature⟩⟩

/-- The commitment of the certificates `crs`, whatever its bytes. -/
def commitment (crs : List CertifiedRecord) (bytes : ByteArray) : NetworkCheck.Check.Commitment :=
  ⟨(crs.map certified).toArray, bytes⟩

/-- The evidence the device's shaper holds: on its one egress link, the frames it handed and the syncs it declared. -/
def evidence (c : Config) (tr : List Event) : NetworkCheck.Check.Evidence :=
  ⟨#[⟨openedFrames c tr, declaredSyncs c tr⟩]⟩

/-- The check's parameters of the device's configuration and key: its two links, its `W` windows, its two grids, its
Σ_sync, and the key's LMS parameters and public key. -/
def unitParams (c : Config) (key : Lms.PrivateKey) : NetworkCheck.Check.Params where
  egress := #[c.egressLink]
  ingress := #[c.ingressLink]
  windows := c.windows
  grid := c.egress
  ingressGrid := c.ingress
  syncSet := c.syncSet
  lms := key.params
  key := key.publicKey

/-! ## The configuration's conditions -/

/-- Both grids have buckets: a grid of `T = 0` never closes a window. -/
def Ticking (c : Config) : Prop := 0 < c.egress.buckets ∧ 0 < c.ingress.buckets

/-- What a record encodes fits its field (`RecordHead.encode`, `certificate_message`): each grid's T, r, B and τ_b are
u32s, every window `w < W` a u64, and each link's UTF-8 is under 2^16 bytes. -/
def FieldsFit (c : Config) : Prop :=
  (∀ g ∈ [c.egress, c.ingress], g.buckets < 2 ^ 32 ∧ g.allocation < 2 ^ 32 ∧ g.frameBytes < 2 ^ 32 ∧
    g.bucketMs < 2 ^ 32) ∧
  c.windows ≤ 2 ^ 64 ∧ c.egressLink.toUTF8.size < 2 ^ 16 ∧ c.ingressLink.toUTF8.size < 2 ^ 16

/-- The two links are two: the host tells a record's direction by its link. -/
def Distinct (c : Config) : Prop := c.egressLink ≠ c.ingressLink

/-- Each grid has a `parameters_digest`, the egress grid's with Σ_sync. -/
def Digested (c : Config) : Prop :=
  (∃ eg, parameters c.egress (some c.syncSet) = .ok eg) ∧ ∃ ig, parameters c.ingress none = .ok ig

/-! ## The run's conditions -/

/-- The run is over: both grids ran all their `W · T` ticks (`State.done`), so every window closed. -/
def Closed (c : Config) (tr : List Event) : Prop := (after (init c) tr).done = true

/-- **Honest timing**, `CompleteRows`'s hypothesis: each frame the shaper handed was handed at the bucket
`NetworkCertifier.handingBuckets` gives it under the declared syncs. -/
def HonestTiming (c : Config) (tr : List Event) : Prop :=
  (handed (init c) tr).map Prod.snd =
    NetworkCertifier.handingBuckets c.egress.buckets (declaredSyncs c tr).size (fun w => (declaredSyncs c tr)[w])
      (openedFrames c tr)

/-- No fault: every record the device published is complete. -/
def NoFault (c : Config) (tr : List Event) : Prop := ∀ r ∈ published c tr, r.status = .complete

/-- The device declared one sync for each of the `W` windows. -/
def SyncsDeclared (c : Config) (tr : List Event) : Prop := (declaredSyncs c tr).size = c.windows

/-- Every sync the device declared is in Σ_sync. -/
def SyncsAdmitted (c : Config) (tr : List Event) : Prop :=
  ∀ s ∈ (declaredSyncs c tr).toList, s.delta ∈ c.syncSet.deltas ∧ s.rho ∈ c.syncSet.rhos

/-- The signer has a leaf for each record: from its next unused leaf `next`, one per record published, all below `2^H`. -/
def LeavesLeft (i : Issuer) (next : Nat) (c : Config) (tr : List Event) : Prop :=
  next + (published c tr).length ≤ 2 ^ i.key.params.H

/-! ## The guarantee -/

/-- **The device's output is accepted.** For a configuration whose grids tick and encode, with two links and digested
grids, and an issuer with a valid key: on a closed run with honest timing, no fault, one admitted sync declared per
window, and a leaf for every record, the host's assembly certifies every record the device published, and the check
accepts that commitment, whatever its bytes, with the evidence the device's shaper holds, under the configuration's
parameters and the key's public key. -/
def DeviceOutputAccepted : Prop :=
  ∀ (c : Config) (tr : List Event) (i : Issuer) (next : Nat),
    Ticking c → FieldsFit c → Distinct c → Digested c → ValidKey i.key →
    Closed c tr → HonestTiming c tr → NoFault c tr → SyncsDeclared c tr → SyncsAdmitted c tr →
    LeavesLeft i next c tr →
    ∃ i' crs, certifyAll c i next (published c tr) = .ok (i', crs) ∧
      ∀ bytes, NetworkCheck.Check.protocol.Accepts (unitParams c i.key) (commitment crs bytes) (evidence c tr)

end Properties.CertifierNetwork
