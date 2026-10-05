/-!
# Canonical frames: the one reader between a party program and its environment

A frame is an 8-byte big-endian length followed by that many bytes. Every byte that crosses between a party's Lean
program and a worker, a peer or the store is a frame, and this is the only code that reads one. A frame longer than the
bound its request declares is refused before it is read.
-/

namespace RunModel.Frame

/-- `n` as 8 big-endian bytes (`n < 2^64`). -/
def u64be (n : Nat) : ByteArray :=
  ⟨(List.range 8).toArray.map fun i => (n >>> (8 * (7 - i)) % 256).toUInt8⟩

/-- The big-endian number in `b[off, off + 8)`. -/
def ofU64be (b : ByteArray) (off : Nat) : Nat :=
  (List.range 8).foldl (fun acc i => acc * 256 + (b.get! (off + i)).toNat) 0

def encode (b : ByteArray) : ByteArray := u64be b.size ++ b

/-- Exactly `n` bytes from `h`, or an error if the stream ends first. -/
partial def readExact (h : IO.FS.Handle) (n : Nat) (acc : ByteArray := .empty) : IO ByteArray := do
  if acc.size ≥ n then return acc
  let chunk ← h.read (n - acc.size).toUSize
  if chunk.isEmpty then throw (IO.userError s!"frame: the stream ended after {acc.size} of {n} bytes")
  readExact h n (acc ++ chunk)

/-- One frame from `h`, refused if it is longer than `max`. -/
def read (h : IO.FS.Handle) (max : Nat) : IO ByteArray := do
  let n := ofU64be (← readExact h 8) 0
  if n > max then throw (IO.userError s!"frame: {n} bytes, over the request's bound of {max}")
  readExact h n

def write (h : IO.FS.Handle) (b : ByteArray) : IO Unit := do
  h.write (encode b)
  h.flush

end RunModel.Frame
