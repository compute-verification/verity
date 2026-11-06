import Lms.Hss
import Attest.FrameV3Sha512

/-!
# A party's signed period root and its liveness records: the checks a period's proof and the verifier run

`verity.core.attest`, whose module docstring is the spec. An auditor party on a device commits each sampling block's
records (each a byte string) as the leaves of one frame-v3-sha512 tree per certification period (`root`, over the domain
`periodDomain count` at the binding `PERIOD_BINDING`, owner −1, leaf schema `PERIOD_SCHEMA`), and its lower HSS key signs the
root once per period (`Lms.periodRootMessage`, binding the deployment, the party, the attachment the challenger drew, the
link, the period's index and the record count). The records, the root and its signature go to the prover only, as
witness. The party's lower key and its certificate under the device's identity (`PartyKey`) are public: every liveness
record shows them in the clear. `checkKey` is the verifier's check of them, in the clear (`Lms.verifyCertificate`).
`check` is what the period's proof runs on its witness: the root's signature verifies under the lower key (one LMS
verification), and each record it reads opens to the root at its position (`FrameV3Sha512.verifyOpening`). `accepts` is
both, the verifier's acceptance of a period. `checkLiveness` is the verifier's check, in the clear, of the party's one
public liveness record per sampling block (`Lms.verifyLiveness`).
-/

namespace Attest

open Flock

/-- `attest.BINDING`: the period tree's binding, SHA-512 of `verity/period-records/v1`. -/
def PERIOD_BINDING : ByteArray := Sha512.hash "verity/period-records/v1".toUTF8

/-- `attest.SCHEMA`: every leaf's schema. -/
def PERIOD_SCHEMA : String := "verity/period-record/v1"

/-- `attest.domain_id(count)`'s domain: the period tree's, over the positions `0 … count − 1`, the `range-u64` kind. -/
def periodDomain (count : Nat) : FrameV3Sha512.Domain := FrameV3Sha512.Domain.makeU64 PERIOD_BINDING (-1) count

/-- Record `r` as leaf `r` (its rank and its position) of the tree over `records`. -/
def leaves (records : Array ByteArray) : Array ByteArray :=
  (Array.range records.size).map fun r => (periodDomain records.size).leaf r r PERIOD_SCHEMA records[r]!

/-- `attest.root(records)`: the period tree's root, the empty root for no records. -/
def root (records : Array ByteArray) : ByteArray := FrameV3Sha512.root (periodDomain records.size) (leaves records)

/-- `attest.path(records, position)`: the siblings of leaf `position`, leaf side first. -/
def path (records : Array ByteArray) (position : Nat) : Array ByteArray :=
  FrameV3Sha512.path (FrameV3Sha512.levels (periodDomain records.size) (leaves records)) position

/-- One record a period's proof reads: its position, its bytes and its path to the root. -/
structure Read where
  position : Nat
  record : ByteArray
  path : Array ByteArray

/-- A party's lower key as the verifier holds it, in the clear: its public key (`I ‖ T[1]`) and the device identity's
signature of its `Lms.partyKeyMessage` (its certificate), the first two parts of every HSS signature it makes. -/
structure PartyKey where
  lower : ByteArray
  certificate : Lms.Signature

/-- A period's signed root as the prover holds it: the period (attachment, link, index), the record count, the root,
and the lower key's signature of the root. -/
structure SignedRoot where
  attachment : ByteArray
  link : String
  period : Nat
  count : Nat
  root : ByteArray
  signature : Lms.Signature

/-- The message the lower key `lower` signs for `s`: `Lms.periodRootMessage` of its identifier and `s`'s fields. -/
def SignedRoot.message (lower deployment : ByteArray) (party : String) (s : SignedRoot) : Lms.E ByteArray :=
  Lms.periodRootMessage (lower.extract 0 16) deployment party s.attachment s.link s.period s.count s.root

/-- **The key check**, in the clear: `k`'s certificate verifies under the device's `identity` over its lower key's
party-key message for `party` in the deployment, at the parameters `p`. False for a lower key of another length;
refused where `Lms.verifyCertificate` refuses (an identity or a deployment hash of the wrong length). -/
def checkKey (p : Lms.Params) (identity deployment : ByteArray) (party : String) (k : PartyKey) : Lms.E Bool :=
  Lms.verifyCertificate p identity deployment party k.lower k.certificate

/-- **The period check**, as the period's proof runs it on its witness: `s`'s signature verifies under the lower key
`lower` over the root's message for `party` in the deployment, at the parameters `p`, and each read opens to `s.root`
at its position in a tree of `s.count` records. False for a lower key of another length; refused where the message
refuses (a deployment hash of the wrong length, a period or count past a u64). -/
def check (p : Lms.Params) (deployment : ByteArray) (party : String) (lower : ByteArray) (s : SignedRoot)
    (reads : Array Read) : Lms.E Bool := do
  if lower.size ≠ 16 + p.N then return false
  if !(← Lms.verify p lower (← s.message lower deployment party) s.signature) then return false
  pure (reads.all fun r =>
    FrameV3Sha512.verifyOpening (periodDomain s.count) s.root s.count r.position r.record PERIOD_SCHEMA r.path)

/-- **The verifier's acceptance of a period**: the key check of the party's public key `k` under the device's
`identity`, then the period check under its lower key. -/
def accepts (p : Lms.Params) (identity deployment : ByteArray) (party : String) (k : PartyKey) (s : SignedRoot)
    (reads : Array Read) : Lms.E Bool := do
  if !(← checkKey p identity deployment party k) then return false
  check p deployment party k.lower s reads

/-- **The liveness check**, in the clear: `sig` (its bytes) is the party's public liveness record of sampling block
`block` of the period, at the parameters `p`. Refused for bytes that are not an HSS signature at `p`. -/
def checkLiveness (p : Lms.Params) (identity deployment : ByteArray) (party : String) (attachment : ByteArray)
    (link : String) (period block : Nat) (sig : ByteArray) : Lms.E Bool := do
  Lms.verifyLiveness p identity deployment party attachment link period block (← Lms.HssSignature.decode p sig)

end Attest
