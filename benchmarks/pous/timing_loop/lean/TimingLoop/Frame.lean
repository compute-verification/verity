/-!
# Frames between the Lean driver and its Python worker

A frame is an 8-byte big-endian length followed by that many bytes, over the worker's stdin and stdout: the transport the
run-model draft's `run-model ping` measured (9.4 µs median, 0.35 ms at p99.9, for an 8-byte request), so the hop measured
here is that one.
-/

namespace TimingLoop.Frame

/-- `n` as 8 big-endian bytes (`n < 2^64`). -/
def u64be (n : Nat) : ByteArray :=
  ⟨(List.range 8).toArray.map fun i => (n >>> (8 * (7 - i)) % 256).toUInt8⟩

/-- The big-endian number in `b[off, off + 8)`. -/
def ofU64be (b : ByteArray) (off : Nat) : Nat :=
  (List.range 8).foldl (fun acc i => acc * 256 + (b.get! (off + i)).toNat) 0

/-- Exactly `n` bytes from `h`, or an error if the stream ends first. -/
def readExact (h : IO.FS.Handle) (n : Nat) : IO ByteArray := do
  let mut acc := ByteArray.empty
  for _ in [0:n + 1] do
    if acc.size ≥ n then break
    let chunk ← h.read (n - acc.size).toUSize
    if chunk.isEmpty then throw (IO.userError s!"frame: the stream ended after {acc.size} of {n} bytes")
    acc := acc ++ chunk
  return acc

/-- One frame from `h`, refused if it is longer than `max`. -/
def read (h : IO.FS.Handle) (max : Nat) : IO ByteArray := do
  let n := ofU64be (← readExact h 8) 0
  if n > max then throw (IO.userError s!"frame: {n} bytes, over the request's bound of {max}")
  readExact h n

def write (h : IO.FS.Handle) (b : ByteArray) : IO Unit := do
  h.write (u64be b.size ++ b)
  h.flush

end TimingLoop.Frame
