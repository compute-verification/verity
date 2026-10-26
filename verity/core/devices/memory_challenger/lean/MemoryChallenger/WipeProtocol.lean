import CoreProtocol.Cli
import CoreProtocol.Ratio
import MemoryChallenger.Protocol
import MemoryChallenger.WipeJudge

/-!
# The wipe as a core protocol

The streamed-label wipe's verdict (`StreamedWipe.verdict`, the wipe judge of record) behind the core interface
(`Core.Protocol`), the wipe's component of onsite Verity, each byte string canonical JSON:

* params `{"server": hex, "session": hex, "trusted": [hex, …]}`: the server the period's wipe is of, the session id
  the memory challenger opened for it, and the verifier's list of memory-challenger keys, which never comes from the
  session's record;
* the commitment: `WipeCommitment.to_json()`, what the memory challenger published when it opened the session; its
  digest SHA-512 of those bytes;
* the evidence: `{"signed": SignedWipe.to_json()}` once the memory challenger has signed the session, and
  `{"signed": null}` while it hasn't (its reads incomplete, or not yet verified);
* the check: a commitment of another server or session rejects; an unsigned session is undecided; else `verdict`,
  whose reasons, joined by `"; "`, are a rejection's.

On acceptance it publishes the foreign content an accepted wipe may leave and the error that holds at,
`{"certified", "egress_mib", "eps", "foreign_bits", "memory_bits"}` (`certify`): `memory_bits` is `M`, the fill's
`labels · 512` bits, which covers every declared tier exactly. When the bound covers the session (`k = 2152` timed
reads, a fill of at least one 32 MiB segment, an egress count that isn't saturated) `foreign_bits` is
`⌈f·M⌉ + E` with `f = 1/19` and `E` the egress the network certifier counted, in bits, at `eps = 2^-128`. Off it,
`foreign_bits` is `M` and `eps` 0, the vacuous statement, and the profile names each gap. The bound and what it rests
on are `EXPERIMENTAL`.

`protocol` (`verity/wipe/v0`) accepts S1's LMS (`SCHEMES`) only; `testProtocol`, under an id of its own, the test
signers' schemes and only them.
-/

namespace MemoryChallenger.WipeProtocol

open Lean (Json)
open Flock

def ID : String := "verity/wipe/v0"

/-- The timed spot reads the bound is at. -/
def READS : Nat := 2152

/-- The foreign fraction the bound leaves, `f`. -/
def F : Core.Ratio := ⟨1, 19⟩

/-- The error the bound holds at, `2^-128`. -/
def EPS : Core.Ratio := ⟨1, 2 ^ 128⟩

/-- The least fill the bound is stated for: one 32 MiB segment. -/
def LEAST_FILL_BYTES : Nat := 2 ^ 25

/-- One MiB in bits: the egress count's unit. -/
def UNIT_BITS : Nat := 8 * 2 ^ 20

def EXPERIMENTAL : String :=
  "f = 1/19 at eps = 2^-128 with 2152 timed reads is Pous.Guarantees.SecureErasureMeetsFamily's bound, proved for the " ++
  "band fill (chainBandSeg); this verdict judges the streamed-label fill, whose proved guarantee is " ++
  "Pous.Guarantees.StreamedWipe. That it implies the published bound for a fill of at least one 32 MiB segment at " ++
  "k = 2152 is a numeric step (streamed_wipe.retained_bits), not proved in Lean, and eps leaves out StreamedWipe's PRF " ++
  "advantage and its premises: every tier declared, the egress certified by the network certifier, no side channel"

private def hexJ (b : ByteArray) : Json := .str (hexOf b)
private def natJ (n : Nat) : Json := Lean.toJson n

structure Params where
  server : ByteArray
  session : ByteArray
  trusted : Array ByteArray

def Params.toJson (p : Params) : Json :=
  Json.mkObj [("server", hexJ p.server), ("session", hexJ p.session), ("trusted", .arr (p.trusted.map hexJ))]

def decodeParams : Json → V Params :=
  exactly (fun j => do
    pure ⟨← hexField j "server" "params", ← hexField j "session" "params", ← hexArr j "trusted" "params"⟩)
    Params.toJson

def decodeCommitment (b : ByteArray) : V StreamedWipe.Commitment := do
  exactly StreamedWipe.Commitment.ofJson StreamedWipe.Commitment.toJson (← Core.canonicalJson b)

/-- The signed session, `none` while the memory challenger hasn't signed it. -/
structure Evidence where
  signed : Option StreamedWipe.Signed

def Evidence.toJson (e : Evidence) : Json :=
  Json.mkObj [("signed", match e.signed with | some s => s.toJson | none => .null)]

def decodeEvidence (b : ByteArray) : V Evidence := do
  exactly (fun j => do
    match ← J.get j "signed" "evidence" with
    | .null => pure ⟨none⟩
    | s => pure ⟨some (← StreamedWipe.Signed.ofJson s)⟩) Evidence.toJson (← Core.canonicalJson b)

/-- SHA-512 of the commitment's canonical JSON, the bytes it decoded from. -/
def digest (c : StreamedWipe.Commitment) : ByteArray :=
  match canon c.toJson with
  | .ok s => Sha512.hash s.toUTF8
  | .error _ => ByteArray.empty

/-- `M`: the fill in bits, every declared tier. -/
def memoryBits (p : StreamedWipe.Params) : Nat := p.labels * StreamedWipe.LABEL_BYTES * 8

/-- Why the bound doesn't cover an accepted session of `p` with `egressMib` MiB of egress (none when it does). -/
def gaps (p : StreamedWipe.Params) (egressMib : Int) : List String :=
  let bytes := p.labels * StreamedWipe.LABEL_BYTES
  (if p.k != READS then [s!"k = {p.k}: the bound is at {READS} timed reads"] else []) ++
  (if bytes < LEAST_FILL_BYTES then
    [s!"the fill is {bytes} bytes, below one 32 MiB segment ({LEAST_FILL_BYTES} bytes), the least the bound is stated for"]
   else []) ++
  (if egressMib < 0 || egressMib ≥ StreamedWipe.EGRESS_MAX_UNITS then
    [s!"the egress count is {egressMib} MiB, outside 0 to {StreamedWipe.EGRESS_MAX_UNITS - 1}: it bounds nothing"]
   else [])

/-- The public outputs: whether the bound covers the session, its egress, the error, the foreign content it leaves and
the memory it is of, in bits. -/
structure Public where
  certified : Bool
  egressMib : Int
  eps : Core.Ratio
  foreignBits : Nat
  memoryBits : Nat

def certify (p : StreamedWipe.Params) (egressMib : Int) : Public :=
  let m := memoryBits p
  if (gaps p egressMib).isEmpty then
    ⟨true, egressMib, EPS, (m * F.num + F.den - 1) / F.den + egressMib.toNat * UNIT_BITS, m⟩
  else ⟨false, egressMib, ⟨0, 1⟩, m, m⟩

def Public.toJson (p : Public) : Json :=
  Json.mkObj [("certified", .bool p.certified), ("egress_mib", Lean.toJson p.egressMib), ("eps", .str p.eps.toString),
    ("foreign_bits", natJ p.foreignBits), ("memory_bits", natJ p.memoryBits)]

/-- The profile: the wipe the session ran, the bound and every gap. -/
def profileJson (x : StreamedWipe.Params × Int) : Json :=
  let (p, e) := x
  Json.mkObj [("batch", natJ p.batch),
    ("certificate", Json.mkObj [("eps", .str EPS.toString), ("f", .str F.toString),
      ("least_fill_bytes", natJ LEAST_FILL_BYTES), ("reads", natJ READS)]),
    ("experimental", .str EXPERIMENTAL), ("gaps", .arr ((gaps p e).map Json.str).toArray), ("k", natJ p.k),
    ("labels", natJ p.labels), ("limit_ns", natJ p.limitNs),
    ("tiers", .arr (p.tiers.map fun (t, b) => .arr #[.str t, natJ b]))]

def check (schemes : Schemes) (p : Params) (c : StreamedWipe.Commitment) (e : Evidence) :
    Core.Outcome Public (StreamedWipe.Params × Int) :=
  if c.server != p.server || c.session != p.session then
    .reject "the wipe's commitment is not of the period's server and session"
  else match e.signed with
  | none => .undecided "the memory challenger has not signed the wipe: its reads are incomplete or not yet verified"
  | some s =>
    let v := StreamedWipe.verdict schemes p.trusted.toList s c
    if v.accepted then .accept (certify c.params v.egressMib) (c.params, v.egressMib)
    else .reject ("; ".intercalate v.reasons.toList)

def mkProtocol (id : String) (schemes : Schemes) : Core.Protocol where
  id := id
  Params := Params
  Commitment := StreamedWipe.Commitment
  Evidence := Evidence
  Public := Public
  Profile := StreamedWipe.Params × Int
  params := decodeParams
  commitment := decodeCommitment
  evidence := decodeEvidence
  digest := digest
  check := check schemes
  publicJson := Public.toJson
  profileJson := profileJson

/-- `verity/wipe/v0` under S1's LMS. -/
def protocol : Core.Protocol := mkProtocol ID SCHEMES

/-- The same check under the test signers' schemes alone, its id `test/verity/wipe/v0`. -/
def testProtocol : Core.Protocol := mkProtocol (TEST_PREFIX ++ ID) TEST_SCHEMES

private def segment : StreamedWipe.Params := ⟨2 ^ 19, READS, 1, 1000, #[("gpu", 2 ^ 25)]⟩

#guard (certify segment 0).foreignBits == (2 ^ 28 + 18) / 19
#guard (certify segment 2).foreignBits == (2 ^ 28 + 18) / 19 + 2 * 2 ^ 23
#guard (certify segment 0).eps.toString == s!"1/{2 ^ 128}"
#guard (certify segment StreamedWipe.EGRESS_MAX_UNITS).certified == false
#guard (certify { segment with k := 2151 } 0).foreignBits == 2 ^ 28
#guard gaps ⟨64, READS, 1, 1, #[("gpu", 4096)]⟩ 0 ==
  ["the fill is 4096 bytes, below one 32 MiB segment (33554432 bytes), the least the bound is stated for"]
#guard (protocol.answer "{}".toUTF8 ByteArray.empty ByteArray.empty).reason.map (·.startsWith "params: ") == some true

end MemoryChallenger.WipeProtocol
