/-!
# Frames: the one reader between the auditor's program and its workers

A frame is an 8-byte big-endian length followed by that many bytes. Every byte that crosses between `flock-audit` and a
worker is a frame, and this is the only code that reads one off a pipe. A frame longer than the bound its request declares
is refused before its body is read.
-/

namespace FlockAudit.Frame

/-- `n` as 8 big-endian bytes (`n < 2^64`). -/
def u64be (n : Nat) : ByteArray :=
  ⟨(Array.range 8).map fun i => (n >>> (8 * (7 - i)) % 256).toUInt8⟩

/-- The big-endian number in `b[off, off + 8)`. -/
def ofU64be (b : ByteArray) (off : Nat) : Nat :=
  (List.range 8).foldl (fun acc i => acc * 256 + (b.get! (off + i)).toNat) 0

def encode (b : ByteArray) : ByteArray := u64be b.size ++ b

/-- The frame at `off` of `b` and the offset after it, or `none` if `b` ends inside it. -/
def take (b : ByteArray) (off : Nat) : Option (ByteArray × Nat) :=
  if off + 8 > b.size then none
  else
    let n := ofU64be b off
    if off + 8 + n > b.size then none else some (b.extract (off + 8) (off + 8 + n), off + 8 + n)

/-- Exactly `n` bytes from `h`, or an error if the stream ends first. -/
partial def readExact (h : IO.FS.Handle) (n : Nat) (acc : ByteArray := .empty) : IO ByteArray := do
  if acc.size ≥ n then return acc
  let chunk ← h.read (n - acc.size).toUSize
  if chunk.isEmpty then throw (IO.userError s!"frame: the stream ended after {acc.size} of {n} bytes")
  readExact h n (acc ++ chunk)

/-- One frame from `h`, refused before its body is read if it is longer than `max`. -/
def read (h : IO.FS.Handle) (max : Nat) : IO ByteArray := do
  let n := ofU64be (← readExact h 8) 0
  if n > max then throw (IO.userError s!"frame: {n} bytes, over the request's bound of {max}")
  readExact h n

def write (h : IO.FS.Handle) (b : ByteArray) : IO Unit := do
  h.write (encode b)
  h.flush

end FlockAudit.Frame
