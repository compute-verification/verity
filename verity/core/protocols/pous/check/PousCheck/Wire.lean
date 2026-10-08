import Flock.Canon
import PousCheck.Merkle
import PousCheck.Challenge
import PousCheck.Wire.Codec

/-!
# `p2-audit-wire/v4`: a setup and one timed audit, verified to its `FINAL` frame

`p2/PROTOCOL.md` §4 ("Audit protocol and wire formats") and §5, as the reference verifier (`benchmarks/pous/p2_v1/wire.py`,
`verify_audit`) reads them. A frame is `"P2AU" ‖ 0x04 ‖ type ‖ u32be(len body) ‖ body`, and a session is
SETUP_COMMIT, SETUP_SALT, SETUP_BLOCK*, SETUP_ROOT, AUDIT_OPEN, then CHALLENGE and RESPONSE alternating. `verifyAudit`
returns the `FINAL` frame the verifier sends, byte for byte, with its verdict, failure code, the transcript's hash and one
147-byte record per round reached; or, when there is no audit identity to report under, the fault.

Every check is the Python's, in the Python's order, because the first failing check names the code:

* setup: the manifest is canonical JSON naming a P2 store (`manifestParams`); SETUP_SALT, every SETUP_BLOCK and
  SETUP_ROOT repeat the commitment (and the salt), the blocks come in order, each canonical below `p`, and a public setup's
  blocks hash to the declared root. A setup failure is reported under the first AUDIT_OPEN in the session.
* the audit: AUDIT_OPEN names this epoch, a fresh `(epoch, audit id)` pair and the audit id derived from the verifier's
  secret; round `j`'s CHALLENGE carries the derived index and nonce (`PousCheck.Challenge`); its RESPONSE echoes them and
  carries a canonical block whose path to the root verifies; the measured latency is at most `Δ + RTT`; with flag bit 0
  the block decodes to a payload below `2^payload_bits` (`Wire.Codec`); and nothing follows the last round.

Latencies are data (the verifier's measurements, `elapsed`), never read here.
-/

namespace PousCheck.Wire

open Lean (Json)

def MAGIC : ByteArray := "P2AU".toUTF8
def VERSION : Nat := 4
def TRANSCRIPT_TAG : ByteArray := "verity/pous/p2/transcript/v4".toUTF8.push 0
def EPOCH_TAG : ByteArray := "verity/pous/p2/epoch/v4".toUTF8.push 0
def MAX_W : Nat := 16448
def MAX_MANIFEST : Nat := 512
def MAX_K : Nat := 105
/-- A digest (the commitment, the root, the epoch id, a sibling, a frame's hash), bytes. -/
def D : Nat := 64
/-- The setup salt, the audit id and a nonce, bytes. -/
def S : Nat := 32

def SETUP_COMMIT : Nat := 0x01
def SETUP_SALT : Nat := 0x02
def SETUP_BLOCK : Nat := 0x03
def SETUP_ROOT : Nat := 0x04
def AUDIT_OPEN : Nat := 0x10
def CHALLENGE : Nat := 0x11
def RESPONSE : Nat := 0x12
def FINAL : Nat := 0x13
def TYPES : List Nat := [SETUP_COMMIT, SETUP_SALT, SETUP_BLOCK, SETUP_ROOT, AUDIT_OPEN, CHALLENGE, RESPONSE, FINAL]

/-! Failure codes (§4): the `FINAL`'s second status byte. -/
def SUCCESS : Nat := 0
def MALFORMED : Nat := 1
/-- Identity, round, index, replay or domain. -/
def IDENTITY : Nat := 2
def AUTHENTICATION : Nat := 3
def LATE : Nat := 4
def DECODE : Nat := 5
def TRANSPORT : Nat := 6

def OPEN_BYTES : Nat := D + S + 1 + 2 + 8 + 8 + 1 + 8
def CHALLENGE_BYTES : Nat := D + S + 2 + 8 + S
def RESPONSE_FIXED : Nat := D + S + 2 + 8 + 4 + 2
def SETUP_ROOT_BYTES : Nat := D + S + 8 + D + 1

/-- Why a session has no `FINAL`, or why a step failed: a failure code and the Python's message. -/
structure Fault where
  code : Nat
  message : String

abbrev W := Except Fault

def fault {α : Type} (code : Nat) (message : String) : W α := throw ⟨code, message⟩

def be := Merkle.be
def natOfBE := Codec.natOfBE

/-- The big-endian number in bytes `[lo, hi)` of `b`. -/
def field (b : ByteArray) (lo hi : Nat) : Nat := natOfBE (b.extract lo hi)

def envelope (type : Nat) (body : ByteArray) : ByteArray :=
  MAGIC ++ ⟨#[VERSION.toUInt8, type.toUInt8]⟩ ++ be body.size 4 ++ body

/-- A frame's type and body (`parse_envelope`), checking its type when one is `expected`. -/
def parseEnvelope (f : ByteArray) (expected : Option Nat := none) : W (Nat × ByteArray) := do
  if f.size < 10 then fault MALFORMED "truncated frame header"
  if f.extract 0 4 != MAGIC then fault MALFORMED "wrong magic"
  if (f.get! 4).toNat != VERSION then fault MALFORMED "wrong wire version"
  let type := (f.get! 5).toNat
  if f.size != 10 + field f 6 10 then fault MALFORMED "body length mismatch or trailing bytes"
  if let some e := expected then
    if type != e then fault MALFORMED "wrong message type"
  if type ∉ TYPES then fault MALFORMED "unknown message type"
  pure (type, f.extract 10 f.size)

/-! ## The manifest -/

/-- A store's shape and codec, as the manifest names them (`reference.Params`). `c0` is an integer and `p = 2^w − c0` may
be negative: `Params` accepts any `c0 ≡ 1 mod 4`, and then no block is below `p`. -/
structure Params where
  w : Nat
  c0 : Int
  blocksPerSegment : Nat
  segments : Nat
  profile : String
  payloadBits : Nat

def Params.p (q : Params) : Int := 2 ^ q.w - q.c0
def Params.widthBytes (q : Params) : Nat := q.w / 8
def Params.blocks (q : Params) : Nat := q.blocksPerSegment * q.segments
def Params.primeId (q : Params) : String := s!"p2-p{q.w}-c{q.c0}"
/-- `x < p`: the block whose big-endian value is `x` is canonical. -/
def Params.canonical (q : Params) (x : Nat) : Bool := decide ((x : Int) < q.p)

/-- `payload_bits`: `p2-<n>b` is `n` bits, `p2-<n>B` is `8n`, at least 1 and below the width; the deployment width takes
only `p2-2054B`. -/
def payloadBits (profile : String) (width : Nat) : W Nat := do
  if width == MAX_W && profile != "p2-2054B" then fault MALFORMED "deployment width requires p2-2054B"
  let cs := profile.toList
  let last := cs.getLast?
  if cs.take 3 == ['p', '2', '-'] && (last == some 'b' || last == some 'B') then
    let digits := (cs.drop 3).dropLast
    if !digits.isEmpty && digits.all Char.isDigit then
      let bits := (String.ofList digits).toNat! * (if last == some 'B' then 8 else 1)
      if 1 ≤ bits && bits < width then return bits
  fault MALFORMED "unknown payload profile"

def FIELDS : Array String := #["blocks_per_segment", "c0", "profile", "scheme", "segments", "w"]

/-- The store a SETUP_COMMIT's manifest names (`_canonical_json`, `_params`): UTF-8 JSON that is its own canonical form
(sorted keys, no whitespace, integers only), an object of exactly `FIELDS` with `scheme = "p2"`, the four numbers
integers in range, a known payload profile, and `p = 2^w − c0 ≡ 3 mod 4` with `w` a multiple of 32. Every refusal is
malformed. -/
def manifestParams (data : ByteArray) : W Params := do
  let bad {α : Type} (m : String) : W α := fault MALFORMED m
  let some text := String.fromUTF8? data | bad "manifest is not canonical UTF-8 JSON"
  let .ok j := Json.parse text | bad "manifest is not canonical UTF-8 JSON"
  let .ok kvs := j.getObj? | bad "manifest is not an object"
  match Flock.canon j with
  | .ok c => if c.toUTF8 != data then bad "manifest is not canonical JSON"
  | .error _ => bad "manifest is not canonical JSON"
  let keys := (kvs.toArray.map (·.1)).qsort (· < ·)
  if keys != FIELDS || j.getObjValD "scheme" != .str "p2" then bad "wrong manifest fields"
  let int (name : String) : W Int := match j.getObjValD name with
    | .num n => if n.exponent == 0 then pure n.mantissa else bad "manifest value has wrong type"
    | _ => bad "manifest value has wrong type"
  let (bps, c0, segments, w) := (← int "blocks_per_segment", ← int "c0", ← int "segments", ← int "w")
  let .str profile := j.getObjValD "profile" | bad "manifest value has wrong type"
  if !(1 ≤ bps && bps ≤ 2 ^ 16) || !(1 ≤ segments && segments < 2 ^ 64) || !(0 < w && w ≤ MAX_W) then
    bad "manifest geometry is out of range"
  let bits ← payloadBits profile w.toNat
  let q : Params := ⟨w.toNat, c0, bps.toNat, segments.toNat, profile, bits⟩
  if w < 8 || w % 32 != 0 || c0 < 1 || c0 % 4 != 1 || q.p % 4 != 3 then bad "no P2 prime: w, c0"
  let payloadBytes := max 1 (bits / 8)
  if !(0 < payloadBytes && payloadBytes < q.widthBytes) then bad "no P2 prime: payload"
  pure q

/-! ## Setup -/

structure Setup where
  manifest : ByteArray
  commitment : ByteArray
  salt : ByteArray
  params : Params
  blocks : Array ByteArray
  root : ByteArray
  epoch : ByteArray
  /-- A public setup's blocks do not hash to the declared root. -/
  authFailed : Bool

/-- The epoch id: `SHA512("verity/pous/p2/epoch/v4\0" ‖ u32be(len manifest) ‖ manifest ‖ commitment ‖ salt ‖ u64be(B) ‖
root)`. -/
def epochId (manifest commitment salt : ByteArray) (blocks : Nat) (root : ByteArray) : ByteArray :=
  Flock.Sha512.hash (EPOCH_TAG ++ be manifest.size 4 ++ manifest ++ commitment ++ salt ++ be blocks 8 ++ root)

/-- The setup frames (`_parse_setup`) from the first, and the index of the frame after SETUP_ROOT. -/
def parseSetup (frames : Array ByteArray) : W (Setup × Nat) := do
  let (_, body) ← parseEnvelope (frames[0]?.getD .empty) (some SETUP_COMMIT)
  if body.size < 4 + D then fault MALFORMED "truncated SETUP_COMMIT"
  let ml := field body 0 4
  if ml > MAX_MANIFEST then fault MALFORMED "manifest exceeds 512-byte limit"
  if body.size != 4 + ml + D then fault MALFORMED "SETUP_COMMIT length mismatch"
  let manifest := body.extract 4 (4 + ml)
  let q ← manifestParams manifest
  let commitment := body.extract (4 + ml) body.size
  let mut cursor := 1
  let some f := frames[cursor]? | fault TRANSPORT "missing SETUP_SALT"
  let (_, body) ← parseEnvelope f (some SETUP_SALT)
  cursor := cursor + 1
  if body.size != D + S || body.extract 0 D != commitment then fault IDENTITY "SETUP_SALT commitment mismatch"
  let salt := body.extract D (D + S)
  let n := q.widthBytes
  let mut blocks : Array ByteArray := #[]
  for f in frames.extract cursor frames.size do
    let (type, body) ← parseEnvelope f
    if type != SETUP_BLOCK then break
    cursor := cursor + 1
    if body.size < D + S + 8 + 4 then fault MALFORMED "truncated SETUP_BLOCK"
    if body.extract 0 D != commitment || body.extract D (D + S) != salt then fault IDENTITY "SETUP_BLOCK identity mismatch"
    let index := field body (D + S) (D + S + 8)
    let length := field body (D + S + 8) (D + S + 12)
    let block := body.extract (D + S + 12) body.size
    if index != blocks.size then fault IDENTITY "SETUP_BLOCK is missing, duplicated, or reordered"
    if length != n || block.size != length then fault MALFORMED "SETUP_BLOCK length mismatch"
    if !q.canonical (natOfBE block) then fault MALFORMED "noncanonical setup block"
    blocks := blocks.push block
  let some f := frames[cursor]? | fault TRANSPORT "missing SETUP_ROOT"
  let (_, body) ← parseEnvelope f (some SETUP_ROOT)
  cursor := cursor + 1
  if body.size != SETUP_ROOT_BYTES then fault MALFORMED "SETUP_ROOT length mismatch"
  if body.extract 0 D != commitment || body.extract D (D + S) != salt then fault IDENTITY "SETUP_ROOT identity mismatch"
  let declared := field body (D + S) (D + S + 8)
  let root := body.extract (D + S + 8) (D + S + 8 + D)
  let mode := (body.get! (body.size - 1)).toNat
  if mode != 0 && mode != 1 then fault MALFORMED "unknown setup mode"
  if declared != q.blocks then fault IDENTITY "setup block count disagrees with manifest"
  if mode == 1 && blocks.size != q.blocks then fault IDENTITY "public setup has missing blocks"
  if mode == 0 && !blocks.isEmpty && blocks.size != q.blocks then fault MALFORMED "partial private setup blocks"
  let bps := q.blocksPerSegment
  let failed := !blocks.isEmpty &&
    Merkle.root (blocks.mapIdx fun i b => Merkle.leaf salt (i / bps) (i % bps) b) != some root
  pure (⟨manifest, commitment, salt, q, blocks, root, epochId manifest commitment salt q.blocks root, failed⟩, cursor)

/-! ## The audit -/

structure Open where
  frame : ByteArray
  epoch : ByteArray
  audit : ByteArray
  mode : Nat
  k : Nat
  delta : Nat
  rtt : Nat
  flags : Nat
  counter : Nat

/-- AUDIT_OPEN's fields at their fixed offsets: `epoch ‖ audit id ‖ mode ‖ u16 k ‖ u64 Δ ‖ u64 RTT ‖ flags ‖ u64 counter`. -/
def openFields (f body : ByteArray) : Open :=
  let o := D + S
  ⟨f, body.extract 0 D, body.extract D o, (body.get! o).toNat, field body (o + 1) (o + 3), field body (o + 3) (o + 11),
   field body (o + 11) (o + 19), (body.get! (o + 19)).toNat, field body (o + 20) (o + 28)⟩

def parseOpen (f : ByteArray) : W Open := do
  let (_, body) ← parseEnvelope f (some AUDIT_OPEN)
  if body.size != OPEN_BYTES then fault MALFORMED "AUDIT_OPEN length mismatch"
  let o := openFields f body
  if (o.mode != 0 && o.mode != 1) || !(1 ≤ o.k && o.k ≤ MAX_K) then fault MALFORMED "invalid audit mode or k"
  if o.flags > 3 then fault MALFORMED "reserved verification flag is set"
  if o.flags == 2 then fault MALFORMED "payload comparison requires decode checking"
  pure o

/-- The `FINAL`'s identity read from a fixed-size but invalid AUDIT_OPEN (`_salvage_open`). -/
def salvageOpen (f : ByteArray) : Option Open :=
  match parseEnvelope f (some AUDIT_OPEN) with
  | .ok (_, body) => if body.size == OPEN_BYTES then some (openFields f body) else none
  | .error _ => none

/-- The identity a setup failure is reported under (`_recover_open`): the first well-framed AUDIT_OPEN in the session,
valid or salvaged. -/
def recoverOpen (frames : Array ByteArray) : Option Open := Id.run do
  for f in frames do
    if let .ok (type, _) := parseEnvelope f then
      if type == AUDIT_OPEN then
        return match parseOpen f with
          | .ok o => some o
          | .error _ => salvageOpen f
  return none

/-- CHALLENGE: `epoch ‖ audit id ‖ u16 round ‖ u64 index ‖ nonce`. -/
def parseChallenge (f : ByteArray) : W (ByteArray × ByteArray × Nat × Nat × ByteArray) := do
  let (_, body) ← parseEnvelope f (some CHALLENGE)
  if body.size != CHALLENGE_BYTES then fault MALFORMED "CHALLENGE length mismatch"
  let o := D + S
  pure (body.extract 0 D, body.extract D o, field body o (o + 2), field body (o + 2) (o + 10), body.extract (o + 10) body.size)

/-- RESPONSE: `epoch ‖ audit id ‖ u16 round ‖ u64 index ‖ u32 n ‖ block ‖ u16 h ‖ h siblings`, the block `blockBytes`. -/
def parseResponse (f : ByteArray) (blockBytes : Nat) :
    W (ByteArray × ByteArray × Nat × Nat × ByteArray × Array ByteArray) := do
  let (_, body) ← parseEnvelope f (some RESPONSE)
  let o := D + S
  if body.size < RESPONSE_FIXED then fault MALFORMED "truncated RESPONSE"
  let length := field body (o + 10) (o + 14)
  if length != blockBytes then fault MALFORMED "wrong block length"
  let stop := o + 14 + length
  if body.size < stop + 2 then fault MALFORMED "truncated block"
  let count := field body stop (stop + 2)
  if body.size - (stop + 2) != D * count then fault MALFORMED "wrong path_count or trailing bytes"
  pure (body.extract 0 D, body.extract D o, field body o (o + 2), field body (o + 2) (o + 10), body.extract (o + 14) stop,
    (Array.range count).map fun x => body.extract (stop + 2 + D * x) (stop + 2 + D * (x + 1)))

/-- A round's record: `u16 round ‖ u64 index ‖ u64 elapsed ‖ flags ‖ SHA512(challenge) ‖ SHA512(response)`. Fields too
wide for their bytes are truncated here; the Python raises (`to_bytes`). -/
def record (j index elapsed flags : Nat) (challengeHash responseHash : ByteArray) : ByteArray :=
  be j 2 ++ be index 8 ++ be elapsed 8 ++ ⟨#[flags.toUInt8]⟩ ++ challengeHash ++ responseHash

/-- `FINAL`: `epoch ‖ audit id ‖ verdict ‖ code ‖ u16 r ‖ u64 Δ ‖ u64 RTT ‖ transcript hash ‖ u16 r ‖ r records`, the
transcript hash `SHA512("verity/pous/p2/transcript/v4\0" ‖ AUDIT_OPEN ‖ every challenge and response read)`. -/
def final (o : Open) (code : Nat) (transcript records : Array ByteArray) : ByteArray :=
  let th := Flock.Sha512.hash (transcript.foldl (· ++ ·) TRANSCRIPT_TAG)
  let r := be records.size 2
  envelope FINAL (o.epoch ++ o.audit ++ ⟨#[if code == SUCCESS then 1 else 0, code.toUInt8]⟩ ++ r ++ be o.delta 8
    ++ be o.rtt 8 ++ th ++ r ++ records.foldl (· ++ ·) .empty)

/-- What the verifier holds besides the frames: its secret, round `j`'s beacon value in mode 1, and the `(epoch, audit id)`
pairs it has seen, when it keeps them. -/
structure Verifier where
  source : Option ByteArray
  beacons : Option (Array (Nat × ByteArray)) := none
  seen : Option (Array (ByteArray × ByteArray)) := none

/-- The rounds of an opened audit, from frame `cursor` on: the `FINAL` frame. -/
def rounds (setup : Setup) (o : Open) (frames : Array ByteArray) (cursor : Nat) (elapsed : Array Nat) (v : Verifier) :
    ByteArray := Id.run do
  let mut transcript := #[o.frame]
  let mut records : Array ByteArray := #[]
  let mut cursor := cursor
  if setup.authFailed then return final o AUTHENTICATION transcript records
  if o.epoch != setup.epoch then return final o IDENTITY transcript records
  if let some seen := v.seen then
    if seen.contains (o.epoch, o.audit) then return final o IDENTITY transcript records
  let src := v.source.getD .empty
  if src.isEmpty || o.audit != Challenge.auditId src o.epoch o.counter then return final o IDENTITY transcript records
  let key := if o.mode == 0 then some (Challenge.auditKey src o.audit o.epoch setup.root setup.salt) else none
  let q := setup.params
  let zero : ByteArray := ⟨Array.replicate D 0⟩
  for j in [0:o.k] do
    let mut expected := (0, ByteArray.empty)
    match key with
    | some k => expected := ((Challenge.index k q.blocks j).getD 0, Challenge.nonce k j)
    | none =>
      match (v.beacons.getD #[]).find? (·.1 == j) with
      | some (_, beacon) =>
        if beacon.size != S then return final o IDENTITY transcript records
        expected := ((Challenge.index (Challenge.roundKey beacon o.audit o.epoch setup.root j) q.blocks j).getD 0, beacon)
      | none => return final o IDENTITY transcript records
    let (expectedIndex, expectedNonce) := expected
    let some cf := frames[cursor]? | return final o TRANSPORT transcript records
    cursor := cursor + 1
    let ch := Flock.Sha512.hash cf
    transcript := transcript.push cf
    match parseChallenge cf with
    | .error e => return final o e.code transcript (records.push (record j expectedIndex 0 0 ch zero))
    | .ok (epoch, audit, rj, index, nonce) =>
      if epoch != o.epoch || audit != o.audit || rj != j || index != expectedIndex || nonce != expectedNonce
          || index ≥ q.blocks then
        return final o IDENTITY transcript (records.push (record j expectedIndex 0 0 ch zero))
      let el := elapsed[j]?.getD 0
      let some rf := frames[cursor]? | return final o TRANSPORT transcript (records.push (record j index el 0 ch zero))
      cursor := cursor + 1
      let rh := Flock.Sha512.hash rf
      transcript := transcript.push rf
      let fail (code flags : Nat) := final o code transcript (records.push (record j index el flags ch rh))
      match parseResponse rf q.widthBytes with
      | .error e => return fail e.code 0
      | .ok (rEpoch, rAudit, rj, rIndex, block, path) =>
        if !q.canonical (natOfBE block) || some path.size != Merkle.pathLength q.blocks index then
          return fail MALFORMED 0
        if rEpoch != o.epoch || rAudit != o.audit || rj != j || rIndex != index then return fail IDENTITY 0
        let (segment, blk) := (index / q.blocksPerSegment, index % q.blocksPerSegment)
        if !Merkle.verifyPath setup.root q.blocks index (Merkle.leaf setup.salt segment blk block) path then
          return fail AUTHENTICATION 0
        let mut flags := 0x01
        if el > o.delta + o.rtt then return fail LATE flags
        flags := flags ||| 0x02
        if o.flags &&& 0x01 != 0 then
          flags := flags ||| 0x04
          let p := q.p.toNat
          match Codec.keys q.primeId setup.commitment setup.salt q.widthBytes p segment blk with
          | none => return fail DECODE flags
          | some (kv, t) =>
            if Codec.decode p (natOfBE block) kv t >>> q.payloadBits != 0 then return fail DECODE flags
          flags := flags ||| 0x08
        -- comparison with a verifier-held payload is setup's, outside the audit: the reference refuses it
        if o.flags &&& 0x02 != 0 then return fail DECODE flags
        records := records.push (record j index el flags ch rh)
  return final o (if cursor != frames.size then IDENTITY else SUCCESS) transcript records

/-- `verify_audit`: a setup and one audit transcript to the exact `FINAL` frame, or the fault when the session has no
audit identity to report it under. `elapsed[j]` is the verifier's measured challenge-to-response time for round `j`. -/
def verifyAudit (frames : Array ByteArray) (elapsed : Array Nat) (v : Verifier) : W ByteArray :=
  if frames.isEmpty then fault TRANSPORT "empty transcript has no audit identity" else
  match parseSetup frames with
  | .error e =>
    match recoverOpen frames with
    | some o => pure (final o e.code #[o.frame] #[])
    | none => throw e
  | .ok (setup, cursor) =>
    match frames[cursor]? with
    | none => fault TRANSPORT "missing AUDIT_OPEN"
    | some f =>
      match parseOpen f with
      | .error e =>
        match salvageOpen f with
        | some s => pure (final s e.code #[f] #[])
        | none => throw e
      | .ok o => pure (rounds setup o frames (cursor + 1) elapsed v)

/-- The `FINAL`'s verdict, failure code and record count (`parse_final`). -/
def parseFinal (f : ByteArray) : W (Bool × Nat × Nat) := do
  let (_, body) ← parseEnvelope f (some FINAL)
  let fixed := D + S + 1 + 1 + 2 + 8 + 8 + D + 2
  if body.size < fixed || (body.size - fixed) % 147 != 0 then fault MALFORMED "FINAL length mismatch"
  let o := D + S
  pure (body.get! o != 0, (body.get! (o + 1)).toNat, field body (o + 2) (o + 4))

end PousCheck.Wire
