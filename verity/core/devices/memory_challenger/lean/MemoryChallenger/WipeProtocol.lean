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
`{"certified", "egress_mib", "eps", "foreign_bits", "k", "memory_bits", "r"}` (`certify`): `memory_bits` is `M`, every
declared tier in bits, the fill's `n · 512` and `R` past it (`outsideBits`, memory the fill doesn't reach); `k` is the
session's timed reads, which the memory challenger chose and its commitment names; `r = ratio k`, the largest `r` with
`(r − 1)^k · 2^129 ≤ r^k`. When the bound covers the session (`k ≤ MAX_READS`, an egress count that isn't saturated)
`foreign_bits` is `Pous.Guarantees.StreamedWipeMeetsReads`' threshold `512·(⌈n/r⌉ − 1) + n + ⌊log₂ k⌋ + 130 + R + E`
(`foreignBits`) with `E` the egress the network certifier counted, in bits, at `eps = 2^-128`. Off it, `foreign_bits`
is `M` and `eps` 0, the vacuous statement, and the profile names each gap. The bound and what it rests on are
`EXPERIMENTAL`.

`protocol` (`verity/wipe/v0`) accepts S1's LMS (`SCHEMES`) only; `testProtocol`, under an id of its own, the test
signers' schemes and only them.
-/

namespace MemoryChallenger.WipeProtocol

open Lean (Json)
open Flock

def ID : String := "verity/wipe/v0"

/-- The error the bound holds at, `2^-128`. -/
def EPS : Core.Ratio := ⟨1, 2 ^ 128⟩

/-- The most reads the certificate is computed at: `ratio` compares numbers of about `k · log₂ r` bits. -/
def MAX_READS : Nat := 2 ^ 17

/-- One MiB in bits: the egress count's unit. -/
def UNIT_BITS : Nat := 8 * 2 ^ 20

def EXPERIMENTAL : String :=
  "foreign_bits at eps = 2^-128 is Pous.Guarantees.StreamedWipeMeetsReads' threshold at the session's k and r = " ++
  "ratio(k), proved for the streamed-label fill in the wipe game; that game's acceptance is not this verdict (the " ++
  "guarantee is frozen, no-function), and eps leaves out its PRF advantage and its premises: every tier declared, the " ++
  "egress certified by the network certifier, no side channel"

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

/-- The fill in bits, `n · 512`. -/
def fillBits (p : StreamedWipe.Params) : Nat := p.labels * StreamedWipe.LABEL_BYTES * 8

/-- `M`: every declared tier in bits, the fill and `R` past it. -/
def memoryBits (p : StreamedWipe.Params) : Nat := p.tiers.foldl (· + ·.2) 0 * 8

/-- `R`: the declared memory the fill doesn't reach, in bits. -/
def outsideBits (p : StreamedWipe.Params) : Nat := memoryBits p - fillBits p

/-- Whether `(r − 1)^k · 2^129 ≤ r^k`, that is `(1 − 1/r)^k ≤ 2^-129`: `StreamedWipeMeetsReads`' sampling condition. -/
def samples (r k : Nat) : Bool := decide ((r - 1) ^ k * 2 ^ 129 ≤ r ^ k)

/-- `ratio`'s bisection, `samples lo k` and not `samples hi k` throughout; `fuel` bounds its steps. -/
def ratioSearch (k : Nat) : Nat → Nat → Nat → Nat
  | 0, lo, _ => lo
  | fuel + 1, lo, hi =>
    if hi - lo ≤ 1 then lo
    else
      let mid := (lo + hi) / 2
      if samples mid k then ratioSearch k fuel mid hi else ratioSearch k fuel lo mid

/-- The largest `r` with `(r − 1)^k · 2^129 ≤ r^k`, by bisection between `1`, which meets it for every `k ≥ 1`, and
`k / 89 + 2`, which doesn't (`(1 − 1/r)^k ≥ e^(−k/(r − 1))` and `129 · ln 2 > 89`); 64 steps are more than its
`log₂ (k / 89 + 2)`. -/
def ratio (k : Nat) : Nat := ratioSearch k 64 1 (k / 89 + 2)

/-- `ratio k`, or 0 past `MAX_READS`, where it isn't computed. -/
def ratioOf (p : StreamedWipe.Params) : Nat := if p.k ≤ MAX_READS then ratio p.k else 0

/-- `StreamedWipeMeetsReads`' threshold before the egress: `512·(⌈n/r⌉ − 1) + n + ⌊log₂ k⌋ + 130 + R`. -/
def foreignBits (p : StreamedWipe.Params) (r : Nat) : Nat :=
  StreamedWipe.LABEL_BYTES * 8 * ((p.labels + r - 1) / r - 1) + p.labels + Nat.log2 p.k + 130 + outsideBits p

/-- Why the bound doesn't cover an accepted session of `p` with `egressMib` MiB of egress (none when it does). -/
def gaps (p : StreamedWipe.Params) (egressMib : Int) : List String :=
  (if p.k > MAX_READS then [s!"k = {p.k}: the certificate is computed for at most {MAX_READS} reads"] else []) ++
  (if egressMib < 0 || egressMib ≥ StreamedWipe.EGRESS_MAX_UNITS then
    [s!"the egress count is {egressMib} MiB, outside 0 to {StreamedWipe.EGRESS_MAX_UNITS - 1}: it bounds nothing"]
   else [])

/-- The public outputs: whether the bound covers the session, its egress, the error, the foreign content it leaves,
the reads and the ratio it is at, and the memory it is of, in bits. -/
structure Public where
  certified : Bool
  egressMib : Int
  eps : Core.Ratio
  foreignBits : Nat
  k : Nat
  memoryBits : Nat
  r : Nat

def certify (p : StreamedWipe.Params) (egressMib : Int) : Public :=
  let m := memoryBits p
  let r := ratioOf p
  if (gaps p egressMib).isEmpty then
    ⟨true, egressMib, EPS, foreignBits p r + egressMib.toNat * UNIT_BITS, p.k, m, r⟩
  else ⟨false, egressMib, ⟨0, 1⟩, m, p.k, m, r⟩

def Public.toJson (p : Public) : Json :=
  Json.mkObj [("certified", .bool p.certified), ("egress_mib", Lean.toJson p.egressMib), ("eps", .str p.eps.toString),
    ("foreign_bits", natJ p.foreignBits), ("k", natJ p.k), ("memory_bits", natJ p.memoryBits), ("r", natJ p.r)]

/-- The profile: the wipe the session ran, the bound and every gap. -/
def profileJson (x : StreamedWipe.Params × Int) : Json :=
  let (p, e) := x
  Json.mkObj [("batch", natJ p.batch),
    ("certificate", Json.mkObj [("eps", .str EPS.toString), ("max_reads", natJ MAX_READS),
      ("outside_bits", natJ (outsideBits p)), ("r", natJ (ratioOf p)), ("reads", natJ p.k)]),
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

private def segment : StreamedWipe.Params := ⟨2 ^ 19, 2152, 1, 1000, #[("gpu", 2 ^ 25)]⟩

/-- `StreamedWipeMeetsH200`'s H200: 141 GiB, filled whole as one tier. -/
private def h200 : StreamedWipe.Params := ⟨2365587456, 2906, 1, 1000, #[("gpu-hbm", 151397597184)]⟩

#guard (ratio 1, ratio 90, ratio 91, ratio 2152, ratio 2905, ratio 2906, ratio 5231, ratio 5321) ==
  (1, 1, 1, 24, 32, 33, 59, 60)
#guard (certify segment 0).foreignBits == 11709069
#guard (certify segment 2).foreignBits == 11709069 + 2 * 2 ^ 23
#guard (certify segment 0).eps.toString == s!"1/{2 ^ 128}"
#guard (certify segment StreamedWipe.EGRESS_MAX_UNITS).certified == false
#guard (certify { segment with k := 5231 } 0).foreignBits == 5074062
#guard outsideBits h200 == 0
#guard (certify h200 0).foreignBits == 39068035213
#guard (certify h200 0).r == 33
#guard gaps { segment with k := MAX_READS + 1 } 0 ==
  [s!"k = {MAX_READS + 1}: the certificate is computed for at most {MAX_READS} reads"]
#guard (certify { segment with k := MAX_READS + 1 } 0).foreignBits == 2 ^ 28
#guard (protocol.answer "{}".toUTF8 ByteArray.empty ByteArray.empty).reason.map (·.startsWith "params: ") == some true

end MemoryChallenger.WipeProtocol
