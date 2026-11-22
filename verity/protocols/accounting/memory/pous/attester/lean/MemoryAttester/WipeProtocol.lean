module

public import CoreProtocol.Account
public import CoreProtocol.Cli
public import CoreProtocol.Ratio
public import MemoryAttester.Protocol
public import MemoryAttester.WipeJudge
meta import CoreProtocol.Account
meta import CoreProtocol.Cli
meta import CoreProtocol.Ratio
meta import MemoryAttester.Protocol
meta import MemoryAttester.WipeJudge

/-!
# The wipe as a core protocol

The streamed-label wipe's verdict (`StreamedWipe.verdict`, the wipe judge of record) behind the core interface
(`Core.Protocol`), the wipe's component of onsite Verity, each byte string canonical JSON:

* params `{"server": hex, "session": hex, "trusted": [hex, …]}`: the server the period's wipe is of, the session id
  the memory challenger opened for it, and the verifier's list of memory-challenger keys, which never comes from the
  session's record;
* the commitment: `WipeCommitment.to_json()`, what the memory challenger published when it opened the session; its
  digest SHA-512 of those bytes;
* the evidence: `{"egress_mib": N | null, "signed": SignedWipe.to_json() | null}`. `signed` is the session once the
  memory challenger has signed it, and `null` while it hasn't (its reads incomplete, or not yet verified).
  `egress_mib` is the one egress count the check reads (`Evidence.egress`), in whole MiB
  (`StreamedWipe.egressUnits`' unit); with `null` it is `EGRESS_MAX_UNITS`, which bounds nothing. The session signs
  no egress, and this check holds the count to nothing: that it is the network attester's count over the span
  between the session's two marks is for the check that composes the wipe with the network attester;
* the check: a commitment of another server or session rejects; an egress outside 0 to `EGRESS_MAX_UNITS` rejects;
  an unsigned session is undecided; else `verdict`, whose reasons, joined by `"; "`, are a rejection's. The deployed
  command answers only on a served road (`checkServed`), held to the served node and to the fill floor the card file
  gives (`--server NAME --fill BYTES`, `store.wipe_fill`): a wipe of another server, or one whose labels fill less, is
  refused by name before `check` reads it. The floor is not the card's memory (`store.dram`, a ceiling): the memory
  between them is filled by no label and is uncovered.

On acceptance it publishes `uncounted`, `certify` at `EGRESS_MAX_UNITS`: `certified` false, whatever count the
evidence names. No network attester signs that count, so until the check that composes the wipe with the network
attester holds it to the certifier's, the bound is taken on no count, and the profile names the evidence's count as
a gap (`writerGap`).

`certify` is what that check is to publish: the foreign content an accepted wipe may leave and the error that holds at,
`{"certified", "egress_mib", "eps", "foreign_bits", "k", "memory_bits", "r"}`: `memory_bits` is `M`, every
declared tier in bits, the fill's `n · 512` and `R` past it (`outsideBits`, memory the fill doesn't reach); `k` is the
session's timed reads, which the memory challenger chose and its commitment names; `r = ratio k`, the largest `r` with
`(r − 1)^k · 2^129 ≤ r^k`. When the bound covers the session (`1 ≤ k ≤ MAX_READS`, an egress count that isn't saturated)
`foreign_bits` is `Pous.Guarantees.StreamedWipeMeetsReads`' threshold `512·(⌈n/r⌉ − 1) + n + ⌊log₂ k⌋ + 130 + R + E`
(`foreignBits`) with `E` the evidence's egress, in bits, at `eps = 2^-128`. Off it, `foreign_bits`
is `M` and `eps` 0, the vacuous statement, and the profile names each gap. The bound and what it rests on are
`EXPERIMENTAL`.

`protocol` (`verity/wipe/v0`) accepts S1's LMS (`SCHEMES`) only; `testProtocol`, under an id of its own, the test
signers' schemes and only them.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace MemoryAttester.WipeProtocol

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

/-- The signed session, `none` while the memory challenger hasn't signed it, and beside it the egress count the check
reads, in whole MiB (`StreamedWipe.egressUnits`' unit), `none` when no count is given. The count is not signed. -/
structure Evidence where
  signed : Option StreamedWipe.Signed
  egressMib : Option Int

/-- The check's one egress value, in whole MiB: the evidence's, or `EGRESS_MAX_UNITS`, which bounds nothing, when it
gives none. -/
def Evidence.egress (e : Evidence) : Int := e.egressMib.getD StreamedWipe.EGRESS_MAX_UNITS

def Evidence.toJson (e : Evidence) : Json :=
  Json.mkObj [("egress_mib", match e.egressMib with | some n => Lean.toJson n | none => .null),
    ("signed", match e.signed with | some s => s.toJson | none => .null)]

def decodeEvidence (b : ByteArray) : V Evidence := do
  exactly (fun j => do
    let signed ← match ← J.get j "signed" "evidence" with
      | .null => pure none
      | s => pure (some (← StreamedWipe.Signed.ofJson s))
    let egressMib ← match ← J.get j "egress_mib" "evidence" with
      | .null => pure none
      | n => pure (some (← J.asInt n "evidence.egress_mib"))
    pure ⟨signed, egressMib⟩) Evidence.toJson (← Core.canonicalJson b)

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

/-- Why the bound doesn't cover an accepted session of `p` with `egressMib` MiB of egress (none when it does). `k = 0`
is one: `decodeCommitment` refuses it (`Params.check`), but `ratio 0 = 1` doesn't sample. -/
def gaps (p : StreamedWipe.Params) (egressMib : Int) : List String :=
  (if p.k == 0 then ["k = 0: a session of no timed reads bounds nothing"]
   else if p.k > MAX_READS then [s!"k = {p.k}: the certificate is computed for at most {MAX_READS} reads"] else []) ++
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

/-- What an accepted wipe publishes: `certify` at `EGRESS_MAX_UNITS`, which certifies nothing, whatever count the
evidence names, since no network attester signs that count. -/
def uncounted (p : StreamedWipe.Params) : Public := certify p StreamedWipe.EGRESS_MAX_UNITS

/-- The evidence's count, when it names one `gaps` doesn't: the evidence writer's, which no network attester signs. -/
def writerGap (egressMib : Int) : List String :=
  if egressMib < 0 || egressMib ≥ StreamedWipe.EGRESS_MAX_UNITS then []
  else [s!"the egress count is {egressMib} MiB, the evidence writer's, which no network certifier signed: no wipe " ++
    "is certified on it until the check that composes the wipe with the network certifier"]

def Public.toJson (p : Public) : Json :=
  Json.mkObj [("certified", .bool p.certified), ("egress_mib", Lean.toJson p.egressMib), ("eps", .str p.eps.toString),
    ("foreign_bits", natJ p.foreignBits), ("k", natJ p.k), ("memory_bits", natJ p.memoryBits), ("r", natJ p.r)]

/-- The profile: the wipe the session ran, the bound and every gap, the evidence's count among them. -/
def profileJson (x : StreamedWipe.Params × Int) : Json :=
  let (p, e) := x
  Json.mkObj [("batch", natJ p.batch),
    ("certificate", Json.mkObj [("eps", .str EPS.toString), ("max_reads", natJ MAX_READS),
      ("outside_bits", natJ (outsideBits p)), ("r", natJ (ratioOf p)), ("reads", natJ p.k)]),
    ("experimental", .str EXPERIMENTAL), ("gaps", .arr ((gaps p e ++ writerGap e).map Json.str).toArray),
    ("k", natJ p.k),
    ("labels", natJ p.labels), ("limit_ns", natJ p.limitNs),
    ("tiers", .arr (p.tiers.map fun (t, b) => .arr #[.str t, natJ b]))]

def check (schemes : Schemes) (p : Params) (c : StreamedWipe.Commitment) (e : Evidence) :
    Core.Outcome Public (StreamedWipe.Params × Int) :=
  if c.server != p.server || c.session != p.session then
    .reject "the wipe's commitment is not of the period's server and session"
  else if e.egress < 0 || e.egress > StreamedWipe.EGRESS_MAX_UNITS then
    .reject s!"the evidence's egress is {e.egress} MiB, outside 0 to {StreamedWipe.EGRESS_MAX_UNITS}"
  else match e.signed with
  | none => .undecided "the memory challenger has not signed the wipe: its reads are incomplete or not yet verified"
  | some s =>
    let v := StreamedWipe.verdict schemes p.trusted.toList s c
    if v.accepted then .accept (uncounted c.params) (c.params, e.egress)
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

/-- The refusal of a case whose params name another server than the served node `served`. -/
def otherServer (served : String) : String := s!"the case is a wipe of another server than {served}"

/-- The refusal of a wipe whose fill, `filled` bytes (its labels at `LABEL_BYTES` each), is less than `fill`, the fill
floor the card file gives: a floor on the fill, not the memory the wipe covers. -/
def underFilled (filled fill : Nat) : String :=
  s!"the wipe fills {filled} bytes, less than the fill floor of {fill} bytes the card file gives"

/-- `check` on a served road, whose served node `served` (its name, the public input `serve --server` takes) the case
must be a wipe of, filling at least `fill` bytes, the fill floor the card file gives (the public input `serve --fill`
takes, the card file's `store.wipe_fill`, below its `store.dram`): a case whose params' `server` is not `served`'s bytes
is refused by name (`otherServer`), then one whose commitment's labels fill less than `fill` bytes (`underFilled`),
before anything else is read. -/
def checkServed (schemes : Schemes) (served : String) (fill : Nat) (p : Params) (c : StreamedWipe.Commitment)
    (e : Evidence) : Core.Outcome Public (StreamedWipe.Params × Int) :=
  if p.server != served.toUTF8 then .reject (otherServer served)
  else if c.params.labels * StreamedWipe.LABEL_BYTES < fill then
    .reject (underFilled (c.params.labels * StreamedWipe.LABEL_BYTES) fill)
  else check schemes p c e

/-- `mkProtocol id schemes` held to the served node `served` and the fill floor `fill` the card file gives
(`checkServed`). -/
def servedProtocol (id : String) (schemes : Schemes) (served : String) (fill : Nat) : Core.Protocol where
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
  check := checkServed schemes served fill
  publicJson := Public.toJson
  profileJson := profileJson

/-- The refusal of `serve` or `check` with no served node and fill floor: the wipe answers only held to both. -/
def unheld (id : String) : String :=
  s!"{id}: serve and check are held to the served node and the fill floor the card file gives: --server NAME " ++
    "--fill BYTES serve, or --server NAME --fill BYTES check PARAMS COMMITMENT EVIDENCE; there is no form without them"

/-- The wipe's command line: `id`, and `serve` or `check` held to the served node NAME and the fill floor BYTES the card
file gives (`servedProtocol`), `--server NAME --fill BYTES` first; any other form, `serve` or `check` without both or
with a fill that is no natural number, is refused by name (`unheld`), exit 64. -/
def mainOf (id : String) (schemes : Schemes) : List String → IO UInt32
  | ["--server", served, "--fill", fill, "serve"] => match fill.toNat? with
    | some n => (servedProtocol id schemes served n).main ["serve"]
    | none => do IO.eprintln (unheld id); return 64
  | ["--server", served, "--fill", fill, "check", p, c, e] => match fill.toNat? with
    | some n => (servedProtocol id schemes served n).main ["check", p, c, e]
    | none => do IO.eprintln (unheld id); return 64
  | ["id"] => (mkProtocol id schemes).main ["id"]
  | _ => do IO.eprintln (unheld id); return 64

/-- `wipe-protocol`'s command line (`mainOf`): no form without a served node and a fill floor. -/
def main : List String → IO UInt32 := mainOf ID SCHEMES

/-- `wipe-protocol-test`'s command line: `mainOf`'s, and the test's form, `serve` and `check` of `testProtocol` held to
no node, which the difference test samples sessions of any server through; its schemes are the test signers' alone, so
nothing it accepts is the deployed protocol's. -/
def testMain (args : List String) : IO UInt32 :=
  match args with
  | ["serve"] | ["check", _, _, _] => testProtocol.main args
  | _ => mainOf (TEST_PREFIX ++ ID) TEST_SCHEMES args

/-- **What a wipe accounts for** (`Core.Account`): the foreign bits the server may still hold after an accepted wipe
(`foreign_bits`), free from above, against the memory the wipe declared (`memory_bits`, its capacity), at the error its
bound states (`eps`), at a moment. Until the network attester's count reaches the check, every accepted wipe publishes
`certified` false, and its rows say so. The wipe is computed at `memory_bits`, the tiers it declared; on the served road
its fill is at least the fill floor the card file gives (`checkServed`), and foreign bits outside what it declared
are outside its bound. The floor is no memory covered: the card's memory past the fill is uncovered.

Its answers name no auditor's device's key, so they are of the whole site (`witness`), and no step the bound held at
(`held`). Its part is every tier a wipe can declare. The session's own tiers are in its profile's `tiers` as pairs of a
label (`gpu-hbm`, `host-dram`, a disk's name) and bytes, not the list of labels `Core.Account.Amount.tiers` reads. -/
def account : Core.Account :=
  { protocol := ID, witness := none
    amounts := [{ metric := "after a wipe", part := .memory [.hbm, .dram, .disk], key := .published "foreign_bits",
                  unit := Core.Account.bits, side := .free, time := .moment, error := .published "eps",
                  capacity := some (.published "memory_bits"), uncertified := .flagged (.published "certified"),
                  params := [{ name := "memory_bits", key := .published "memory_bits" }] }] }

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
#guard (Evidence.egress ⟨none, none⟩, Evidence.egress ⟨none, some 3⟩) == (StreamedWipe.EGRESS_MAX_UNITS, 3)
#guard ((uncounted segment).certified, (uncounted segment).egressMib, (uncounted segment).foreignBits,
  (uncounted segment).eps.toString) == (false, StreamedWipe.EGRESS_MAX_UNITS, memoryBits segment, "0")
#guard (uncounted h200).certified == false
#guard writerGap 0 == ["the egress count is 0 MiB, the evidence writer's, which no network certifier signed: no wipe " ++
  "is certified on it until the check that composes the wipe with the network certifier"]
#guard (writerGap StreamedWipe.EGRESS_MAX_UNITS, writerGap (-1)) == ([], [])

/-- A commitment of no timed reads. -/
private def unread : StreamedWipe.Commitment := ⟨"n0".toUTF8, ⟨#[0]⟩, { segment with k := 0 }⟩

#guard gaps unread.params 0 == ["k = 0: a session of no timed reads bounds nothing"]
#guard (certify unread.params 0).certified == false
#guard match canon unread.toJson with
  | .ok s => (decodeCommitment s.toUTF8).toOption.isNone
  | .error _ => false
#guard (protocol.answer "{}".toUTF8 ByteArray.empty ByteArray.empty).reason.map (·.startsWith "params: ") == some true
#guard account.Formed ∧ account.protocol = protocol.id

#guard match checkServed SCHEMES "n0" 0 ⟨"n0".toUTF8, ⟨#[0]⟩, #[]⟩ unread ⟨none, none⟩ with
  | .undecided r => r.startsWith "the memory challenger has not signed"
  | _ => false
#guard match checkServed SCHEMES "n1" 0 ⟨"n0".toUTF8, ⟨#[0]⟩, #[]⟩ unread ⟨none, none⟩ with
  | .reject r => r == otherServer "n1" && r == "the case is a wipe of another server than n1"
  | _ => false

end MemoryAttester.WipeProtocol
