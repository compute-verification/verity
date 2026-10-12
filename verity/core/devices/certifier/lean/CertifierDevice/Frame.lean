import Flock.Bytes

/-!
# The link's frames and the server's events

Executable forms of `active.py`'s framing: a link frame is B bytes, a header (kind u8, stream u32be, length u32be), the
payload and zero fill (`encodeFrame`, `decodeFrame`); a server's byte stream is cut after every `\n\n` (`Events.feed`), and
an event's `id:` field is its canonical position (`eventPosition`). A frame's bits (`bitsOf`) are the payload the Program
carries.
-/

namespace CertifierDevice

open Flock

/-- The bits of a frame's bytes, most significant first: the payload `NetworkCertifier`'s Program carries. -/
def bitsOf (b : ByteArray) : List Bool := b.toList.flatMap fun x => (List.range 8).map fun i => x.toNat.testBit (7 - i)

/-- `HEADER.size`: kind u8, stream u32, length u32. -/
def headerSize : Nat := 9

def kindPad : Nat := 0
def kindData : Nat := 1
def kindLast : Nat := 2

/-- Big-endian u32 at offset `o`. -/
def be32At (b : ByteArray) (o : Nat) : Nat :=
  (List.range 4).foldl (fun acc i => acc * 256 + (b.get! (o + i)).toNat) 0

/-- `encode_frame`: the header, the payload, then zero fill to `B` bytes; none where Python raises (the payload doesn't
fit). -/
def encodeFrame (kind stream : Nat) (payload : ByteArray) (B : Nat) : Option ByteArray :=
  if headerSize + payload.size > B then none
  else some (⟨#[kind.toUInt8]⟩ ++ be32 stream.toUInt32 ++ be32 payload.size.toUInt32 ++ payload
    ++ ⟨Array.replicate (B - headerSize - payload.size) 0⟩)

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
