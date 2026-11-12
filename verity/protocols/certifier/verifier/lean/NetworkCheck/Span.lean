module

public import NetworkCheck.Period
public import Identity.Domains
meta import NetworkCheck.Period
meta import Identity.Domains

/-!
# The count over a span between two marks, a core protocol: `verity/certifier/span/v0`

A span runs on one link of the network certifier, from the execution phase of a start mark to the execution phase of
an end mark. The count decides, from certificates alone, whether every execution phase of the span and of its tail is
certified under the schedule, and how many bits can have left on the link over them. Its Python mirror is
`verity/protocols/certifier/verifier/span.py`, held to it by `tests/test_network_span_protocol.py`.

**A mark** is 64 bytes handed to the network certifier, which certifies them at an execution phase of the link
(`certificate.CertifiedMark`): it signs the mark hidden under 64 bytes of its own coins,

    signed digest = SHA-512("verity-network-certifier/mark/v1" ‖ salt ‖ mark)     (`markDigest`)

as an egress certificate on the link's marks link name, `<link>-marks` (`marksLink`), under the message every
certificate signs (`NetworkCheck.certificateMessage`), which binds the key, the certifier session, the link name and
the execution phase. The mark and its salt are witness; the certificate is a signature over a hiding digest.

**Params**, canonical JSON, all public:

    {"key": hex(I ‖ T[1]), "link": name, "lms": {"H": h, "N": n, "W": w, "toy": bool},
     "schedule": {…}, "session": hex(session)}

the network certifier's LMS key and its parameters, the certifier session's id (32 bytes), the link, its name held to
what the period's check holds it to, at most 65,535 UTF-8 bytes, and the schedule, as the period's check reads it (its
`scheduleOf`).

**Commitment**: the two marks, the start's then the end's, each

    u32be len(certificate) ‖ certificate ‖ salt (64) ‖ mark (64),
    certificate = u16be len(I) ‖ I ‖ u16be len(link) ‖ link ‖ u8 direction ‖ u64be execution phase ‖
                  signed digest (64) ‖ u32be len(signature) ‖ signature

the certificate in its one serialization (`Certificate.encode` in Python, the bytes a certified record holds it in),
read exactly (`Read.certificateOfBytes`, held to `Certificate.decode`): each length fits, direction 0 or 1, a link's
name UTF-8, nothing left over. Nothing is left after the two marks. Its `digest` is
`tagged_sha512("verity-network-certifier/span/v0", the commitment's bytes)`. The marks are witness, their execution
phases with them: a composition reads them off the commitment (`named`) and publishes only that digest.

**Evidence**, canonical JSON, one entry per execution phase of the span and its tail, in order:

    {"phases": [{"certificates": [egress, fails], "phase": w, "registration": rows}, …]}

the execution phase's two egress certificates, in the period's form (its `certificateOf`): its responses' record's,
on the link, and its fails record's, on the link's fails link name, `<link>-fails`; and the signed head of its egress
cells, a row registration (the period's `rowsOf`).

**The check**, each failure a rejection with its own reason, none of which holds a number: the two marks' execution
phases are not public, and a reason is part of an answer.

1. Each mark opens and verifies (`Mark.refusal`, in `CertifiedMark.refusal`'s order): its certificate is on the link's
   marks link name, is an egress certificate, is the key's, signs `markDigest` of its salt and its mark, and its
   signature verifies in the certifier session.
2. The marks are in order: w_s ≤ w_e, the execution phases their certificates name.
3. The span's tail is `tail` execution phases. The network certifier holds a connection's response before it releases
   it, D buckets from its admission and then r cells a bucket, at most `egress_cells` of them, so the bytes it holds at
   the end mark have left within D + ⌈egress_cells / r⌉ buckets of it. The start needs no tail.
4. The evidence shows each execution phase from w_s to w_e + tail once, in order: one outside them, one shown twice
   and a missing one each reject.
5. Each execution phase's two certificates are egress certificates on their link names at that execution phase and
   verify under the key in the session. The network certifier certifies only a complete record, and none of an
   execution phase it faulted or refused a connection in, so a span that holds such a phase is rejected.
6. Each execution phase's egress cells' head is the link's at that execution phase, names the schedule's digest and
   is signed under the key in the session (the row registration's check, `check_registration` in Python). No
   certificate binds the schedule; this head is the signed thing that does.

**Public** (`publicJson`, the one place that says what of the count is public): `{"egress_bits", "phases", "tail"}`:
the count of execution phases `phases = w_e + tail − w_s + 1`, the tail, and `egress_bits`, `phases` times the bits one
execution phase can carry on the link (`capacityBits`). A certificate hides its record, so each execution phase counts
at capacity: every cell of both its egress records, the responses' and the fails', since both leave on the link. No
rounding is applied. The two marks' execution phases are not public: they say when the span ran, which a mark's salt
hides. **Profile**: `{"link", "phases"}`.

The check walks the two marks and the evidence's entries once, never the span's length.
-/

@[expose] public section

namespace NetworkCheck.Span

open Lean (Json toJson)
open Flock (V)
open NetworkCheck.Check (Certificate)
open NetworkCheck.Period (Schedule Rows scheduleOf rowsOf certificateOf rowsDomain headDigest rowsFormat rowSchema
  outBits)

def protocolId : String := Domains.CERTIFIER_SPAN_PROTOCOL

def commitmentTag : String := Domains.CERTIFIER_SPAN

/-- `certificate.MARK_TAG`. -/
def markTag : ByteArray := Domains.CERTIFIER_MARK.toUTF8

/-- A mark is 64 bytes (`certificate.DIGEST_BYTES`). -/
def markBytes : Nat := 64

/-- What `mark_digest` hashes: `"verity-network-certifier/mark/v1" ‖ salt ‖ mark`. -/
def markInput (salt mark : ByteArray) : ByteArray := markTag ++ salt ++ mark

/-- `certificate.mark_digest`: the digest a mark certificate signs, the 64-byte mark hidden under the 64 bytes the
network certifier sampled for it. -/
def markDigest (salt mark : ByteArray) : ByteArray := Flock.Sha512.hash (markInput salt mark)

/-- `certificate.marks_link`: the link name a link's marks are certified under. -/
def marksLink (link : String) : String := link ++ "-marks"

/-- The link name a link's fails record is certified under. -/
def failsLink (link : String) : String := link ++ "-fails"

/-- The two egress records the count reads at each execution phase, by what the reasons call them and the link name
each is certified under. -/
def records (link : String) : Array (String × String) := #[("egress", link), ("fails", failsLink link)]

structure Params where
  link : String
  lms : Lms.Params
  key : ByteArray
  session : ByteArray
  schedule : Schedule
  /-- The schedule's digest (`Schedule.digest`), which each execution phase's egress cells' head names. -/
  scheduleDigest : ByteArray

/-- A mark as the network certifier hands it back (`certificate.CertifiedMark`): its certificate, the certificate's
bytes as the commitment holds them, the salt and the 64 bytes marked. -/
structure Mark where
  certificate : Certificate
  certificateBytes : ByteArray
  salt : ByteArray
  mark : ByteArray

structure Commitment where
  startMark : Mark
  endMark : Mark
  bytes : ByteArray

/-- One execution phase as the evidence shows it. -/
structure Phase where
  phase : Nat
  certificates : Array Certificate
  registration : Rows

abbrev Evidence := Array Phase

/-- What the check counted. `publicJson` says which of it is public. -/
structure Count where
  startPhase : Nat
  endPhase : Nat
  tail : Nat
  /-- The execution phases counted: the span's and its tail's. -/
  phases : Nat
  /-- The bits one execution phase can carry on the link (`capacityBits`). -/
  capacityBits : Nat

structure Profile where
  link : String
  phases : Nat

/-- The object's keys are exactly `want`, which is sorted. -/
def keyed (j : Json) (want : List String) (ctx : String) : V Unit :=
  match j with
  | .obj kvs => if kvs.keys == want then pure () else throw s!"{ctx}: the keys must be {want}, not {kvs.keys}"
  | _ => throw s!"{ctx}: not an object"

def paramsOf (j : Json) : V Params := do
  keyed j ["key", "link", "lms", "schedule", "session"] "the object"
  let link ← Flock.J.str j "link" "the object"
  unless link.toUTF8.size < 2 ^ 16 do throw "link: a name is at most 65,535 UTF-8 bytes"
  let l ← Flock.J.get j "lms" "the object"
  keyed l ["H", "N", "W", "toy"] "lms"
  let lms ← Lms.Params.make (← Flock.J.nat l "N" "lms") (← Flock.J.nat l "W" "lms") (← Flock.J.nat l "H" "lms")
    (← Flock.J.bool l "toy" "lms")
  let key ← Flock.unhex (← Flock.J.str j "key" "the object")
  unless key.size = 16 + lms.N do throw s!"key: an LMS public key is {16 + lms.N} bytes, not {key.size}"
  let session ← Flock.unhex (← Flock.J.str j "session" "the object")
  unless session.size = sessionBytes do
    throw s!"session: a certifier session id is {sessionBytes} bytes, not {session.size}"
  let schedule ← scheduleOf (← Flock.J.get j "schedule" "the object")
  pure { link, lms, key, session, schedule, scheduleDigest := ← schedule.digest }

namespace Read

open Flock.Reader

/-- `n` bytes as a big-endian number. -/
def be (n : Nat) : M Nat := do
  pure ((← bytes n).foldl (fun a x => a * 256 + x.toNat) 0)

def direction : M Direction := do
  match ← be 1 with
  | 0 => pure .egress
  | 1 => pure .ingress
  | c => throw s!"a direction code is 0 or 1, not {c}"

/-- A certificate's fields in its bytes (`Certificate.encode` in Python), its link's name UTF-8. -/
def certificate : M Certificate := do
  let keyId ← bytes (← be 2)
  let name ← bytes (← be 2)
  let direction ← direction
  let window ← be 8
  let digest ← bytes 64
  let signature ← bytes (← be 4)
  let some link := String.fromUTF8? name | throw "a certificate's link is not UTF-8"
  pure { keyId, link, direction, window, digest, signature }

/-- The certificate `b` is the bytes of (`Certificate.decode` in Python), read exactly: each length fits and nothing is
left over. -/
def certificateOfBytes (b : ByteArray) : V Certificate := do
  let (c, r) ← certificate.run ⟨b, 0⟩
  if r.pos ≠ b.size then throw "bytes left after a certificate"
  pure c

/-- A mark (`span.mark_bytes`): its certificate's bytes behind their length, kept beside the certificate they read
as, its salt, then the mark. -/
def mark : M Mark := do
  let certificateBytes ← bytes (← be 4)
  let certificate ← certificateOfBytes certificateBytes
  let salt ← bytes saltBytes
  pure { certificate, certificateBytes, salt, mark := ← bytes markBytes }

end Read

/-- The commitment's two marks, the start's then the end's, with nothing left over. -/
def commitmentOf (b : ByteArray) : V Commitment := do
  let both : Flock.Reader.M (Mark × Mark) := do
    let s ← Read.mark
    pure (s, ← Read.mark)
  let ((startMark, endMark), r) ← both.run ⟨b, 0⟩
  if r.pos ≠ b.size then throw "bytes left after the two marks"
  pure { startMark, endMark, bytes := b }

def phaseOf (j : Json) : V Phase := do
  let ctx := "an execution phase"
  keyed j ["certificates", "phase", "registration"] ctx
  let certificates ← (← Flock.J.arr j "certificates" ctx).mapM (certificateOf · s!"{ctx}.certificates")
  pure { phase := ← Flock.J.nat j "phase" ctx, certificates,
         registration := ← rowsOf (← Flock.J.get j "registration" ctx) s!"{ctx}.registration" }

def evidenceOf (b : ByteArray) : V Evidence := do
  let j ← Core.canonicalJson b
  keyed j ["phases"] "the object"
  (← Flock.J.arr j "phases" "the object").mapM phaseOf

/-- The span's tail in execution phases, ⌈(D + ⌈egress_cells / r⌉) / T⌉: a connection's response starts D buckets
after its admission and is released r cells a bucket, at most `egress_cells` of them, so what the network certifier
holds at the end mark has left within D + ⌈egress_cells / r⌉ buckets, T buckets an execution phase. -/
def tail (s : Schedule) : Nat :=
  (s.delay + (s.egressCells + s.slots - 1) / s.slots + s.window - 1) / s.window

/-- The bits that can leave on the link in one execution phase: every cell of its two egress records, the responses'
(T buckets of lanes · r cells) and the fails' (T buckets of lanes cells), B bytes a cell. -/
def capacityBits (s : Schedule) : Nat := s.window * (s.allocation + s.lanes) * s.cellBytes * 8

/-- Why a mark isn't one of the link's under the key in the session, in `CertifiedMark.refusal`'s order, or `none`
when it opens and verifies. -/
def Mark.refusal (p : Params) (m : Mark) : Option String :=
  let c := m.certificate
  if c.link != marksLink p.link then some "it is on another link than the link's marks link name"
  else if c.direction != .egress then some "it is not an egress certificate"
  else if c.keyId != p.key.extract 0 16 then some "it is under another key than the network certifier's"
  else if c.digest != markDigest m.salt m.mark then some "its salt and message do not open the digest it signs"
  else if !c.signs p.lms p.key p.session then
    some "its signature does not verify under the key in the certifier session"
  else none

/-- What a missing certificate means, in each reason that reports one. -/
def noneCertified : String :=
  "the network certifier certifies no record of an execution phase it faulted or refused a connection in, and a span \
    that holds one is rejected"

/-- The reason of a span that shows too few execution phases. -/
def missing : String := s!"an execution phase of the span or its tail is missing: {noneCertified}"

/-- Why execution phase `w`'s entry doesn't certify it under the schedule: its two egress certificates, then its
egress cells' signed head (the row registration's check, `check_registration` in Python). -/
def phaseCheck (p : Params) (w : Nat) (x : Phase) : Except String Unit := do
  let recs := records p.link
  unless x.certificates.size == recs.size do
    throw s!"not two certificates, its egress record's and its fails record's: {noneCertified}"
  for ((what, link), c) in recs.zip x.certificates do
    unless c.direction == .egress do throw s!"the {what} certificate is not an egress certificate"
    unless c.link == link do throw s!"the {what} certificate is on another link than {link}"
    unless c.window == w do throw s!"the {what} certificate is at another execution phase"
    unless c.signs p.lms p.key p.session do
      throw s!"the {what} certificate does not verify under the key in the certifier session: {noneCertified}"
  let (r, s, sd) := (x.registration, p.schedule, p.scheduleDigest)
  let (B, S, T) := (s.cellBytes, s.allocation, s.window)
  let d := Direction.egress
  unless r.parameters == Flock.hexOf sd do
    throw "the egress cells' head names another schedule's digest than the parameters'"
  unless r.format == rowsFormat && r.value == s!"{p.link}/{d.value}/{w}" && r.link == p.link &&
      r.direction == d.value && r.window == w do
    throw "the egress cells' head: not the link's egress rows at its execution phase"
  unless r.cellBytes == B && r.rows == T && r.words == S * B / 2 && r.wordBits == outBits &&
      r.rowSchema == rowSchema && r.layout.buckets == T && r.layout.slots == S &&
      r.layout.connectionSlots == s.slots && r.layout.firstBucket == w * T do
    throw "the egress cells' head: not the schedule's rows"
  let .ok dom := rowsDomain s sd p.session p.link d w | throw "the egress cells' head: no domain"
  unless r.domain == Flock.hexOf dom do throw "the egress cells' head: not the execution phase's domain"
  let .ok head := headDigest p.link d w sd s r.wiring dom r.root | throw "the egress cells' head: no encoding"
  unless r.digest == head do throw "the egress cells' head: its certificate doesn't sign it"
  let signed : Certificate :=
    { keyId := r.key, link := p.link, direction := d, window := w, digest := r.digest, signature := r.signature }
  unless signed.signs p.lms p.key p.session do
    throw "the egress cells' head: its certificate does not verify under the key in the certifier session"

/-- The evidence shows each execution phase from `first` to `last` once, in order, each certified (`phaseCheck`): one
pass over its entries. -/
def phasesCheck (p : Params) (first last : Nat) (e : Evidence) : Except String Unit := do
  for (x, i) in e.zipIdx do
    let want := first + i
    unless first ≤ x.phase && x.phase ≤ last do
      throw "an execution phase shown is outside the span and its tail"
    if x.phase < want then throw "an execution phase is shown twice"
    if x.phase > want then throw missing
    match phaseCheck p want x with
    | .error why => throw s!"an execution phase shown: {why}"
    | .ok () => pure ()
  if e.size < last - first + 1 then throw missing

def check (p : Params) (c : Commitment) (e : Evidence) : Core.Outcome Count Profile :=
  match c.startMark.refusal p, c.endMark.refusal p with
  | some why, _ => .reject s!"the start mark: {why}"
  | none, some why => .reject s!"the end mark: {why}"
  | none, none =>
    let (first, endPhase) := (c.startMark.certificate.window, c.endMark.certificate.window)
    if endPhase < first then
      .reject "the marks are out of order: the end mark's execution phase is before the start mark's"
    else
      let t := tail p.schedule
      match phasesCheck p first (endPhase + t) e with
      | .error why => .reject why
      | .ok () =>
        let phases := endPhase + t - first + 1
        .accept { startPhase := first, endPhase, tail := t, phases, capacityBits := capacityBits p.schedule }
          { link := p.link, phases }

/-- What of the count is public, the one place that says so (`span.public`): the bits that can have left over the
span and its tail, the count of execution phases and the tail. The two marks' execution phases are not: they say when
the span ran, which a mark's salt hides, and they stay in the commitment. -/
def publicJson (x : Count) : Json :=
  Json.mkObj [("egress_bits", toJson (x.phases * x.capacityBits)), ("phases", toJson x.phases), ("tail", toJson x.tail)]

def profileJson (pr : Profile) : Json := Json.mkObj [("link", .str pr.link), ("phases", toJson pr.phases)]

def protocol : Core.Protocol where
  id := protocolId
  Params := Params
  Commitment := Commitment
  Evidence := Evidence
  Public := Count
  Profile := Profile
  params := paramsOf
  commitment := commitmentOf
  evidence := evidenceOf
  digest c := (Identity.taggedSha512 commitmentTag c.bytes).toOption.getD .empty
  check := check
  publicJson := publicJson
  profileJson := profileJson

/-- What a composition's tie reads off a case, meaningful once the count accepts it: the SHA-512 of the start
certificate's bytes, which a wipe's signed message names as its start mark; the 64 bytes the two marks open to; and the
bits that can have left over the span and its tail, with no rounding applied (`none` when the marks are out of order).
The first three are witness, read off the commitment. -/
structure Named where
  startMark : ByteArray
  startMessage : ByteArray
  endMessage : ByteArray
  egressBits : Option Nat

def named (p : Params) (c : Commitment) : Named :=
  let (first, endPhase) := (c.startMark.certificate.window, c.endMark.certificate.window)
  { startMark := Flock.Sha512.hash c.startMark.certificateBytes, startMessage := c.startMark.mark,
    endMessage := c.endMark.mark,
    egressBits := if first ≤ endPhase then
      some ((endPhase + tail p.schedule - first + 1) * capacityBits p.schedule) else none }

/-- `named` of a case's parameters, as their canonical JSON text, and its commitment's bytes; `none` unless both
decode. -/
def namedOf (params : String) (commitment : ByteArray) : Option Named :=
  match Core.canonicalJson params.toUTF8 >>= paramsOf, commitmentOf commitment with
  | .ok p, .ok c => some (named p c)
  | _, _ => none

/-- A schedule of the count's `#guard`s: D = 8, r = 2, T = 16 and 17 egress cells a response. -/
def guardSchedule : Schedule :=
  { bucketMs := 100, cellBytes := 256, delay := 8, egressCells := 17, every := 1, ingressBuckets := 50,
    ingressCells := 64, lanes := 16, queue := 64, slots := 2, window := 16 }

-- `mark_digest(bytes(64), bytes(range(64)))`
#guard Flock.hexOf (markDigest ⟨Array.replicate 64 0⟩ ⟨(Array.range 64).map (·.toUInt8)⟩) ==
  "c5108f51c851520366babd954de0feea44aadce8c56000925897737e494d0e61285f4fd5a7bee0fb20fc3788d5a2ecb351e55205475fefd444aa7850f2462050"
-- ⌈(8 + ⌈17 / 2⌉) / 16⌉ = 2; one cell more a bucket and a shorter response fit one execution phase: ⌈(8 + 8) / 16⌉
#guard tail guardSchedule == 2 && tail { guardSchedule with egressCells := 16 } == 1
-- the least schedule's tail is 2: a response's one cell leaves a bucket after its delay
#guard tail { guardSchedule with delay := 1, egressCells := 1, slots := 1, window := 1 } == 2
-- the default schedule's: ⌈(10 + 4096 / 4) / 600⌉ execution phases, and 600 buckets of 64 + 16 cells of 256 bytes
#guard tail { guardSchedule with delay := 10, egressCells := 4096, slots := 4, window := 600 } == 2
#guard capacityBits { guardSchedule with slots := 4, window := 600 } == 98304000
#guard marksLink "eth0" == "eth0-marks" && records "eth0" == #[("egress", "eth0"), ("fails", "eth0-fails")]
-- a link's name is any UTF-8: `é` reads, its first byte alone does not
#guard
  (Read.certificateOfBytes (⟨#[0, 0, 0, 2, 0xc3, 0xa9, 0, 0, 0, 0, 0, 0, 0, 0, 5]⟩ ++ ⟨Array.replicate 68 0⟩)).isOk &&
  !(Read.certificateOfBytes (⟨#[0, 0, 0, 1, 0xc3, 0, 0, 0, 0, 0, 0, 0, 0, 5]⟩ ++ ⟨Array.replicate 68 0⟩)).isOk

/-- Parameters of the `#guard`s, as their canonical JSON: the link `eth0` under `guardSchedule`, a zero key and a zero
session. -/
def guardParams : String :=
  "{\"key\":\"" ++ "".pushn '0' 48 ++ "\",\"link\":\"eth0\",\"lms\":{\"H\":1,\"N\":8,\"W\":8,\"toy\":true},\
    \"schedule\":{\"bucket_ms\":100,\"cell_bytes\":256,\"delay\":8,\"egress_cells\":17,\"every\":1,\
    \"ingress_buckets\":50,\"ingress_cells\":64,\"lanes\":16,\"queue\":64,\"slots\":2,\"window\":16},\
    \"session\":\"" ++ "".pushn '0' 64 ++ "\"}"

/-- A mark of the `#guard`s, in the commitment's bytes: the 107 bytes of an egress certificate on `eth0-marks` at
execution phase `phase`, its key id and its signed digest bytes of `fill` and its signature empty, behind their length,
then a salt of `fill + 1` and a mark of `fill + 2`. -/
def guardMark (phase fill : UInt8) : ByteArray :=
  ⟨#[0, 0, 0, 107, 0, 16]⟩ ++ ⟨Array.replicate 16 fill⟩ ++ ⟨#[0, 10]⟩ ++ "eth0-marks".toUTF8 ++
    ⟨#[0, 0, 0, 0, 0, 0, 0, 0, phase]⟩ ++ ⟨Array.replicate 64 fill⟩ ++ ⟨#[0, 0, 0, 0]⟩ ++
    ⟨Array.replicate 64 (fill + 1)⟩ ++ ⟨Array.replicate 64 (fill + 2)⟩

-- what a tie reads off a case, as `span.named` has it of the same parameters and marks
-- (`test_network_span_protocol.py`): 7 + 2 − 5 + 1 execution phases of 16 · (32 + 16) cells of 256 bytes
#guard (namedOf guardParams (guardMark 5 1 ++ guardMark 7 4)).map
    (fun n => (Flock.hexOf n.startMark, n.startMessage, n.endMessage, n.egressBits)) ==
  some ("37ccc33e0f5191d268bc91c122be57358555be87a54a8b0fa8384ca157a82be17d31e4d6bae5b7c54bc387c9e7d6f8d000111046dd3372fafaef4af0455cfe69",
    ⟨Array.replicate 64 3⟩, ⟨Array.replicate 64 6⟩, some (5 * 1572864))
#guard (namedOf guardParams (guardMark 7 1 ++ guardMark 5 4)).map (·.egressBits) == some none
#guard (namedOf guardParams (guardMark 5 1)).isNone && (namedOf "{}" (guardMark 5 1 ++ guardMark 7 4)).isNone
-- a certificate's bytes are read exactly: one byte more or one fewer in their length is refused
#guard (Read.certificateOfBytes ((guardMark 5 1).extract 4 111)).isOk &&
  !(Read.certificateOfBytes ((guardMark 5 1).extract 4 112)).isOk &&
  !(Read.certificateOfBytes ((guardMark 5 1).extract 4 110)).isOk

end NetworkCheck.Span
