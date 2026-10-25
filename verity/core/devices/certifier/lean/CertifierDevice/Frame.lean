import Flock.Bytes

/-!
# The link's frames and the server's events

Executable forms of `active.py`'s framing: a link frame is B bytes, a header (kind u8, stream u32be, length u32be), the
payload and zero fill (`encodeFrame`, `decodeFrame`), and a salt frame carries its PoUW window and its salt or the
placeholder (`saltFrame`); a server's byte stream is cut after every `\n\n` (`Events.feed`), an
event's `id:` field is its canonical position (`eventPosition`), and what follows that line is the frame's payload
(`eventBody`). On the link every request is its length (u32be) and its bytes over `DATA` frames, and `Requests.feed` cuts
the released payloads back into requests, the k-th on stream k. A frame's bits (`bitsOf`) are the payload the Program
carries. Under `ordinals` a stream's server connection opens with `ordinalHeader`; under `compact` the server's bytes
are an HTTP response head, then (position, record) units (`CompactUnits.feed`), and a frame is the record zero-filled
(`compactCell`), a record naming its request by ordinal (`compactOrdinal`).
-/

namespace CertifierDevice

open Flock

/-- The bits of a frame's bytes, most significant first: the payload `NetworkCertifier`'s Program carries. -/
def bitsOf (b : ByteArray) : List Bool := b.toList.flatMap fun x => (List.range 8).map fun i => x.toNat.testBit (7 - i)

/-- `HEADER.size`: kind u8, stream u32, length u32. -/
def headerSize : Nat := 9

def kindPad : Nat := 0
def kindData : Nat := 1

/-- `CANONICAL_STREAM`: the stream every frame of the canonical framing carries in its header. -/
def canonicalStream : Nat := 0

/-- Big-endian u32 at offset `o`. -/
def be32At (b : ByteArray) (o : Nat) : Nat :=
  (List.range 4).foldl (fun acc i => acc * 256 + (b.get! (o + i)).toNat) 0

/-- `encode_frame`: the header, the payload, then zero fill to `B` bytes; none where Python raises (the payload doesn't
fit). -/
def encodeFrame (kind stream : Nat) (payload : ByteArray) (B : Nat) : Option ByteArray :=
  if headerSize + payload.size > B then none
  else some (⟨#[kind.toUInt8]⟩ ++ be32 stream.toUInt32 ++ be32 payload.size.toUInt32 ++ payload
    ++ ⟨Array.replicate (B - headerSize - payload.size) 0⟩)

def kindSalt : Nat := 3

/-- `SALT_STREAM`: the stream a salt frame's header carries. -/
def saltStream : Nat := 0xFFFFFFFF

/-- `PLACEHOLDER`: what the certifier's own frame on the link carries in its salt's place. -/
def placeholder : ByteArray := ⟨Array.replicate 64 0⟩

/-- `SALT_FRAME_BYTES`: the header, the PoUW window (u64be) and the placeholder. -/
def saltFrameBytes : Nat := headerSize + 8 + placeholder.size

/-- `salt_frame`: PoUW window `p`'s salt frame of `B` bytes, carrying `value` after the window (u64be). -/
def saltFrame (p : Nat) (value : ByteArray) (B : Nat) : Option ByteArray :=
  encodeFrame kindSalt saltStream (be64 p.toUInt64 ++ value) B

/-- `decode_frame`: (kind, stream, payload) of a frame of at least `headerSize` bytes; the payload is cut at the frame's
end, as Python's slice is. -/
def decodeFrame (frame : ByteArray) : Nat × Nat × ByteArray :=
  let n := be32At frame 5
  ((frame.get! 0).toNat, be32At frame 1, frame.extract headerSize (headerSize + n))

/-- The ASCII whitespace `bytes.strip()` removes. -/
def isSpace (c : UInt8) : Bool := c == 32 || c == 9 || c == 10 || c == 13 || c == 11 || c == 12

def isDigit (c : UInt8) : Bool := 48 ≤ c && c ≤ 57

/-- `bytes.strip()`. -/
def strip (b : ByteArray) : ByteArray :=
  let l := b.toList
  let l := (l.dropWhile isSpace).reverse.dropWhile isSpace |>.reverse
  ⟨l.toArray⟩

/-- `bytes.split(b"\n")`. -/
def lines (b : ByteArray) : List ByteArray :=
  (b.toList.splitOn 10).map fun l => ⟨l.toArray⟩

def idPrefix : List UInt8 := "id:".toUTF8.toList

/-- `event_position`: the first line starting `id:` decides; its value, stripped, is the position if it is a non-empty
run of ASCII digits. -/
def eventPosition (event : ByteArray) : Option Nat :=
  match (lines event).find? (fun l => idPrefix.isPrefixOf l.toList) with
  | none => none
  | some l =>
    let v := strip (l.extract 3 l.size)
    if v.size > 0 && v.toList.all isDigit then some (v.toList.foldl (fun acc c => acc * 10 + (c - 48).toNat) 0)
    else none

/-- `event_body`: an event's lines after its first `id:` line (`eventPosition`'s), joined by `\n` again; the event
itself if it has none. -/
def eventBody (event : ByteArray) : ByteArray :=
  let ls := event.toList.splitOn 10
  match ls.findIdx? (fun l => idPrefix.isPrefixOf l) with
  | none => event
  | some i => ⟨(List.intercalate [10] (ls.drop (i + 1))).toArray⟩

/-- `Requests`: the stream of the request being cut, and how many of its bytes are still to come. -/
structure Requests where
  stream : Nat := 0
  left : Nat := 0

/-- The big-endian u32 at the start of `b`, zero past its end (`payload[:4].ljust(4, b"\0")`). -/
def be32Head (b : ByteArray) : Nat :=
  (List.range 4).foldl (fun acc i => acc * 256 + (b.toList.getD i 0).toNat) 0

/-- `Requests.feed`: a released `DATA` payload's stream, the request's bytes in it and whether it ends the request. A
request starts at a frame with its length; the frame that holds its last byte ends it, and the rest of it is fill. -/
def Requests.feed (r : Requests) (payload : ByteArray) : Requests × Nat × ByteArray × Bool :=
  let (r, payload) := if r.left = 0 then (({ stream := r.stream + 1, left := be32Head payload } : Requests),
    payload.extract 4 payload.size) else (r, payload)
  let data := payload.extract 0 r.left
  let left := r.left - data.size
  ({ r with left }, r.stream, data, left == 0)

/-- `ORDINAL`: the kind of the header (`HEADER`, with the stream and length 0) that opens a stream's server connection
under `ordinals`; never on a link. -/
def kindOrdinal : Nat := 4

/-- The header that opens stream `s`'s server connection under `ordinals`. -/
def ordinalHeader (s : Nat) : ByteArray := ⟨#[kindOrdinal.toUInt8]⟩ ++ be32 s.toUInt32 ++ be32 0

/-- `COMPACT_RECORD.size`: a compact record is its request's ingress ordinal (u32be) and its token field (u32be). -/
def compactRecordSize : Nat := 8

/-- `COMPACT_LAST`: the token field's flag on the response's last output (finish `length`, or `stop` with
`compactStop`). -/
def compactLast : Nat := 2 ^ 31

/-- `COMPACT_STOP`: the token field's flag, beside `compactLast`, on a sampled stop token, whose id the field still
carries. -/
def compactStop : Nat := 2 ^ 30

/-- `COMPACT_UNIT_BYTES`: what the server writes for each record, its canonical position (u32be), then the record. -/
def compactUnitBytes : Nat := 4 + compactRecordSize

/-- `compact_record_data`'s request ordinal: a compact cell or record is `compactRecordSize` bytes, zero after them, of
an ordinal ≥ 1 and a token field that is an id below `compactStop` with no flag, `compactLast`, or `compactLast` and
`compactStop`, so a field with `compactStop` and not `compactLast` is no record's; none for bytes that aren't one. The
device reads only the ordinal: the token field, finish included, is the server's. -/
def compactOrdinal (cell : ByteArray) : Option Nat :=
  if cell.size < compactRecordSize || (cell.toList.drop compactRecordSize).any (· != 0) then none else
  let request := be32At cell 0
  let field := be32At cell 4
  if request == 0 || field / compactStop == 1 then none else some request

/-- `compact_cell`: a compact record's egress frame, the record zero-filled to `B` bytes; none where Python raises (the
record doesn't fit). -/
def compactCell (record : ByteArray) (B : Nat) : Option ByteArray :=
  if record.size > B then none else some (record ++ ⟨Array.replicate (B - record.size) 0⟩)

/-- The index of the first `\r\n\r\n` in `b`. -/
def headEnd (b : ByteArray) : Option Nat :=
  (List.range b.size).find? fun i =>
    i + 3 < b.size && b.get! i == 13 && b.get! (i + 1) == 10 && b.get! (i + 2) == 13 && b.get! (i + 3) == 10

/-- `CompactUnits`: a server connection's bytes not yet cut, and whether its HTTP response head is past. -/
structure CompactUnits where
  buffer : ByteArray := ByteArray.empty
  head : Bool := false

/-- `CompactUnits.feed`: append; until the head is past, drop through the first `\r\n\r\n` once there is one; then cut
every whole `compactUnitBytes` unit, first to last, into (position, record). -/
def CompactUnits.feed (u : CompactUnits) (data : ByteArray) : CompactUnits × Array (Nat × ByteArray) :=
  let b := u.buffer ++ data
  let (b, head) := if u.head then (b, true) else
    match headEnd b with
    | some i => (b.extract (i + 4) b.size, true)
    | none => (b, false)
  if !head then ({ buffer := b }, #[]) else
  let n := b.size / compactUnitBytes
  let out := (List.range n).toArray.map fun k =>
    (be32At b (k * compactUnitBytes), b.extract (k * compactUnitBytes + 4) ((k + 1) * compactUnitBytes))
  ({ buffer := b.extract (n * compactUnitBytes) b.size, head := true }, out)

/-- `Events`: the bytes of a stream not yet cut into events. -/
structure Events where
  buffer : ByteArray := ByteArray.empty

/-- `Events.feed`: append, then cut after every `\n\n`, first to last. -/
def Events.feed (e : Events) (data : ByteArray) : Events × Array ByteArray := Id.run do
  let b := e.buffer ++ data
  let mut start := 0
  let mut out := #[]
  for i in [0:b.size] do
    if start ≤ i && i + 1 < b.size && b.get! i == 10 && b.get! (i + 1) == 10 then
      out := out.push (b.extract start (i + 2))
      start := i + 2
  return (⟨b.extract start b.size⟩, out)

end CertifierDevice
