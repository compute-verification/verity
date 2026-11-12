module

public import MemoryChallenger.Verdict

/-!
# The streamed-label wipe's verdict

`verity/protocols/pous/verifier/streamed_wipe.py`'s formats and `verdict`, whose module docstring and `PROTOCOL.md` ("The
streamed-label wipe") are the spec: the commitment and its header, the batch record (`Batch.value`), the records root, the
signed message with its start mark, the start and end messages, the unit an egress count is published in
(`egressUnits`), and the check of a signed session against the commitment published when it opened. The reasons are the
Python's, word for word. The session signs no egress count, and the verdict carries none.

A record's fields and the fill's times are integers of either sign, as the Python holds them. One past its width in the
record (`to_bytes`'s `OverflowError`) leaves the session with no message, so its signature doesn't verify, and every other
check reads it as it is. The verdict refuses a reveal of more than one index a batch (`Params.covered`), which the
guarantee (`Pous.Guarantees.StreamedWipe`) doesn't cover.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace MemoryChallenger.StreamedWipe

open Flock

def PROTOCOL : String := "verity-pous/streamed-wipe/v2"
def SESSION_TAG : ByteArray := "verity-pous/streamed-wipe/session/v2".toUTF8.push 0
def START_TAG : ByteArray := "verity-pous/streamed-wipe/start/v1".toUTF8.push 0
def END_TAG : ByteArray := "verity-pous/streamed-wipe/end/v1".toUTF8.push 0
def RECORD_BINDING : ByteArray := "verity-pous/streamed-wipe/batch-records/v1".toUTF8.push 0
def RECORDS_TREE : String := "verity-pous/streamed-wipe/batch-records/frame-v3-sha512/v1"
def RECORD_SCHEMA : String := "pous/streamed-wipe/batch-record/v1"

/-- The clock a session's times are on, as its header names it: Linux's raw monotonic clock, in ns, which time
synchronisation doesn't slew. The wipe's own: the memory challenger's sessions name `MemoryChallenger.CLOCK`. -/
def CLOCK : String := "CLOCK_MONOTONIC_RAW ns"
def LABEL_BYTES : Nat := 64

/-- A start mark: the SHA-512 of the start certificate. -/
def MARK_BYTES : Nat := 64
def COMPLETE : String := "complete"

/-- The public egress count's unit: one MiB, in bits. -/
def EGRESS_UNIT_BITS : Int := 8 * 2 ^ 20

/-- The count saturates at 15 bits: the wipe's public unit stays in the least public size class, 16 bits, the verdict's
one and the count's 15. -/
def EGRESS_MAX_UNITS : Int := 2 ^ 15 - 1

/-- `WipeParams`: `labels` (`N`), the `k` challenges revealed `batch` at a time, each batch's deadline, and the memory
tiers as `(name, bytes)`, which hold the fill and may declare memory past it. -/
structure Params where
  labels : Nat
  k : Nat
  batch : Nat
  limitNs : Nat
  tiers : Array (String × Nat)
  deriving BEq, Inhabited

/-- `WipeParams.__post_init__`: refused as the Python refuses it. -/
def Params.check (p : Params) : E Unit := do
  if p.labels < 1 || p.batch < 1 || p.batch > p.k || p.k ≥ 2 ^ 32 || p.limitNs == 0 then
    throw "a wipe has labels, 1 <= batch <= k < 2^32 challenges and a positive deadline"
  if p.tiers.isEmpty || p.tiers.any (·.2 == 0) || (p.tiers.toList.map (·.1)).eraseDups.length != p.tiers.size then
    throw "a wipe declares its tiers, each named once with a positive size"
  let total := p.tiers.foldl (· + ·.2) 0
  if total < p.labels * LABEL_BYTES then
    throw s!"the tiers total {total} bytes, less than the fill's {p.labels * LABEL_BYTES}: the declared tiers hold the fill"

/-- `WipeParams.covered`: the sequential reveal, one index a batch, is what the guarantee covers. `verdict` refuses any
other. -/
def Params.covered (p : Params) : Bool := p.batch == 1

/-- `WipeParams.batches`: `⌈k / batch⌉`. -/
def Params.batches (p : Params) : Nat := (p.k + p.batch - 1) / p.batch

/-- `WipeCommitment`: what the challenger publishes when it opens the session. -/
structure Commitment where
  server : ByteArray
  session : ByteArray
  params : Params
  deriving BEq, Inhabited

private def tiersJson (tiers : Array (String × Nat)) : String :=
  "[" ++ ",".intercalate (tiers.toList.map fun (t, b) => "[" ++ pyJsonStr t ++ "," ++ toString b ++ "]") ++ "]"

/-- `WipeCommitment.header()`'s fields, each value as `json.dumps` writes it. -/
def Commitment.headerFields (c : Commitment) : List (String × String) :=
  let p := c.params
  [("protocol", pyJsonStr PROTOCOL), ("records_tree", pyJsonStr RECORDS_TREE), ("server", pyJsonStr (hexOf c.server)),
   ("session", pyJsonStr (hexOf c.session)), ("labels", toString p.labels), ("label_bytes", toString LABEL_BYTES),
   ("k", toString p.k), ("batch", toString p.batch), ("limit_ns", toString p.limitNs), ("tiers", tiersJson p.tiers),
   ("clock", pyJsonStr CLOCK)]

/-- One batch's record. -/
structure Batch where
  b : Int
  first : Int
  count : Int
  sendNs : Int
  receiptNs : Int
  wrong : Int
  digest : ByteArray
  deriving BEq, Inhabited

def Batch.elapsedNs (r : Batch) : Int := r.receiptNs - r.sendNs

/-- `x.to_bytes(n, "big")` of an integer of either sign: refused below 0 and at `2^(8n)` and above (`OverflowError`). -/
def beInt (n : Nat) (x : Int) (what : String) : E ByteArray :=
  if x < 0 then throw s!"{what} = {x} is negative" else beN n x.toNat what

/-- `BatchRecord.value`: `u32 b ‖ u32 first ‖ u32 count ‖ u64 send_ns ‖ u64 receipt_ns ‖ u32 wrong ‖ digest`. -/
def Batch.value (r : Batch) : E ByteArray := do
  pure ((← beInt 4 r.b "b") ++ (← beInt 4 r.first "first") ++ (← beInt 4 r.count "count")
    ++ (← beInt 8 r.sendNs "send_ns") ++ (← beInt 8 r.receiptNs "receipt_ns") ++ (← beInt 4 r.wrong "wrong") ++ r.digest)

/-- `records_root`: the frame-v3-sha512 root over the batch records, the empty root for none. -/
def recordsRoot (session : ByteArray) (batches : Array Batch) : E ByteArray := do
  let d := domain RECORD_BINDING session batches.size
  let values ← batches.mapM Batch.value
  pure (FrameV3Sha512.root d ((Array.range values.size).map fun r => d.leaf r r RECORD_SCHEMA values[r]!))

/-- The signed header: the commitment's header with how the session ended, the start mark as hex (the empty string for
none), the fill's times and the number of batches, as `json.dumps(sort_keys=True, separators=(",", ":"))`. -/
def header (c : Commitment) (ended : String) (startMark : ByteArray) (fillStartNs fillEndNs : Int) (batches : Nat) :
    String :=
  let fields := (c.headerFields ++ [("ended", pyJsonStr ended), ("fill_start_ns", toString fillStartNs),
    ("fill_end_ns", toString fillEndNs), ("batches", toString batches),
    ("start_mark", pyJsonStr (hexOf startMark))]).toArray.qsort (·.1 < ·.1)
  "{" ++ ",".intercalate (fields.toList.map fun (k, v) => pyJsonStr k ++ ":" ++ v) ++ "}"

/-- `message`: what the session's one signature signs, `SHA-512(SESSION_TAG ‖ u64 len ‖ header ‖ records root)`. -/
def message (c : Commitment) (ended : String) (startMark : ByteArray) (fillStartNs fillEndNs : Int)
    (batches : Array Batch) : E ByteArray := do
  let head := (header c ended startMark fillStartNs fillEndNs batches.size).toUTF8
  let root ← recordsRoot c.session batches
  pure (Sha512.hash (SESSION_TAG ++ be64 head.size.toUInt64 ++ head ++ root))

/-- The commitment's header alone, as `json.dumps(sort_keys=True, separators=(",", ":"))`: what the start message
hashes. -/
def Commitment.headerJson (c : Commitment) : String :=
  let fields := c.headerFields.toArray.qsort (·.1 < ·.1)
  "{" ++ ",".intercalate (fields.toList.map fun (k, v) => pyJsonStr k ++ ":" ++ v) ++ "}"

/-- `start_message`: what the challenger hands the network certifier before the fill,
`SHA-512(START_TAG ‖ u64 len ‖ commitment header)`. Not signed. -/
def startMessage (c : Commitment) : ByteArray :=
  let head := c.headerJson.toUTF8
  Sha512.hash (START_TAG ++ be64 head.size.toUInt64 ++ head)

/-- `SignedWipe`: a closed session, its start mark (the SHA-512 of the start certificate the challenger held before the
fill, empty when it had no mark source) and the one signature. It holds no egress count. -/
structure Signed where
  commitment : Commitment
  ended : String
  startMark : ByteArray
  fillStartNs : Int
  fillEndNs : Int
  batches : Array Batch
  scheme : String
  publicKey : ByteArray
  signature : ByteArray
  deriving BEq, Inhabited

def Signed.message (s : Signed) : E ByteArray :=
  StreamedWipe.message s.commitment s.ended s.startMark s.fillStartNs s.fillEndNs s.batches

/-- `end_message`: what the challenger hands the network certifier after its last read,
`SHA-512(END_TAG ‖ u64 len ‖ message ‖ u64 len ‖ signature)`, the session's signed message and its signature. Not
signed. -/
def endMessage (s : Signed) : E ByteArray := do
  let m ← s.message
  pure (Sha512.hash (END_TAG ++ be64 m.size.toUInt64 ++ m ++ be64 s.signature.size.toUInt64 ++ s.signature))

/-- `egress_units`: a count of egress bits in whole MiB, rounded up (Python's floor division), saturating at
`EGRESS_MAX_UNITS`: the unit the wipe's check takes its egress in (`WipeProtocol.Evidence`). -/
def egressUnits (egressBits : Int) : Int := min (-(Int.fdiv (-egressBits) EGRESS_UNIT_BITS)) EGRESS_MAX_UNITS

/-- What the check found besides its verdict: the reasons it rejects (empty when it passes), which are witness. It
holds no amount. -/
structure Findings where
  reasons : Array String
  deriving BEq, Inhabited

/-- Whether the signed session passes, as `Core.Verdict` (`accept` exactly when there is no reason, never `undecided`),
with its findings. The verdict is what `Verdict.public()` publishes. -/
abbrev Verdict := Findings × Core.Verdict

def Verdict.accepted (v : Verdict) : Bool := v.2 == Core.Verdict.accept
def Verdict.reasons (v : Verdict) : Array String := v.1.reasons

/-- `verdict(signed, commitment, trusted=trusted, schemes=schemes)`. It fails closed: an unknown scheme, an untrusted key
or a bad signature rejects whatever the batches say, and a commitment whose batch size the guarantee doesn't cover
(`Params.covered`) rejects the session however it ran. -/
def verdict (schemes : Schemes) (trusted : List ByteArray) (s : Signed) (c : Commitment) : Verdict := Id.run do
  let mut reasons : Array String := #[]
  match schemes.lookup s.scheme with
  | none => reasons := reasons.push s!"signature scheme {pyRepr s.scheme} is not accepted"
  | some p =>
    if !trusted.contains s.publicKey then
      reasons := reasons.push "the session is signed by a key that is not a trusted memory challenger's"
    else
      let ok := match s.message >>= fun m => lmsVerify p s.publicKey s.commitment.session m s.signature with
        | .ok b => b
        | .error _ => false
      if !ok then reasons := reasons.push "the signature does not verify"
  if !(s.commitment == c) then reasons := reasons.push "the signed header is not the session's commitment"
  let p := c.params
  if !p.covered then
    reasons := reasons.push s!"batch size {p.batch} is uncovered: the guarantee covers the sequential reveal, batch size 1"
  if s.ended != COMPLETE || s.batches.size != p.batches then
    reasons := reasons.push s!"the session ended {s.ended} after {s.batches.size} of {p.batches} batches"
  for h : b in [0:s.batches.size] do
    let r := s.batches[b]
    if r.b != (b : Int) || r.first != ((b * p.batch : Nat) : Int) || r.count != min (p.batch : Int) ((p.k : Int) - r.first) then
      reasons := reasons.push s!"batch {b} is not the session's batch {b}"
    if r.wrong != 0 then reasons := reasons.push s!"batch {b} has {r.wrong} wrong answers"
    if !(0 ≤ r.elapsedNs && r.elapsedNs ≤ (p.limitNs : Int)) then
      reasons := reasons.push s!"batch {b} is past its deadline of {p.limitNs} ns"
    if b == 0 && r.sendNs < s.fillEndNs then reasons := reasons.push "the first batch was sent before the fill completed"
  return (Findings.mk reasons, acceptIf reasons.isEmpty)

end MemoryChallenger.StreamedWipe
