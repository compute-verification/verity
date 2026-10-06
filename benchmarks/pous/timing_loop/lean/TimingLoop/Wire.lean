import Std.Internal.UV.TCP

/-!
# The responder's wire, from Lean

`pous_native.cu`'s wire (`benchmarks/pous/band_gpu/memory_challenger.py`): a request is `<QQ` (op, argument); an answer is a 24-byte
header `<QQQ` (status, device ns, server ns) and one block. The socket is the toolchain's own (`Std.Internal.UV.TCP`, libuv
on Lean's event-loop thread): a receive resolves a promise there, and this thread waits on it.
-/

namespace TimingLoop.Wire

open Std.Internal.UV.TCP

def opChallenge : Nat := 1
def opPing : Nat := 2
def opClose : Nat := 3
def header : Nat := 24

/-- `n` as 8 little-endian bytes (`n < 2^64`). -/
def u64le (n : Nat) : ByteArray :=
  ⟨(List.range 8).toArray.map fun i => (n >>> (8 * i) % 256).toUInt8⟩

/-- The little-endian number in `b[off, off + 8)`. -/
def ofU64le (b : ByteArray) (off : Nat) : Nat :=
  (List.range 8).foldr (fun i acc => acc * 256 + (b.get! (off + i)).toNat) 0

def await {α : Type} (p : IO.Promise (Except IO.Error α)) : IO α := do
  IO.ofExcept (← IO.wait p.result!)

/-- A connection to the responder on loopback port `port`, with Nagle off. -/
def connect (port : Nat) : IO Socket := do
  let s ← Socket.new
  await (← s.connect (.v4 ⟨Std.Net.IPv4Addr.ofParts 127 0 0 1, port.toUInt16⟩))
  s.noDelay
  return s

/-- Exactly `n` bytes from `s`, and how many receives they took. -/
def recvExact (s : Socket) (n : Nat) : IO (ByteArray × Nat) := do
  let mut acc := ByteArray.empty
  let mut calls := 0
  for _ in [0:n + 1] do
    if acc.size ≥ n then break
    let some chunk ← await (← s.recv? (n - acc.size).toUInt64)
      | throw (IO.userError s!"wire: the responder closed after {acc.size} of {n} bytes")
    acc := acc ++ chunk
    calls := calls + 1
  return (acc, calls)

/-- One answer: its bytes (header and block), the clock just before the request is sent and just after the whole answer is
received, and how many receives it took. -/
structure Answer where
  bytes : ByteArray
  sentNs : Nat
  receivedNs : Nat
  recvs : Nat

def Answer.status (a : Answer) : Nat := ofU64le a.bytes 0
def Answer.deviceNs (a : Answer) : Nat := ofU64le a.bytes 8
def Answer.serverNs (a : Answer) : Nat := ofU64le a.bytes 16

def roundtrip (s : Socket) (block op arg : Nat) : IO Answer := do
  let req := u64le op ++ u64le arg
  let t0 ← IO.monoNanosNow
  let sent ← s.send #[req]
  let (bytes, recvs) ← recvExact s (header + block)
  let t1 ← IO.monoNanosNow
  await sent
  return ⟨bytes, t0, t1, recvs⟩

def close (s : Socket) : IO Unit := do
  await (← s.send #[u64le opClose ++ u64le 0])

end TimingLoop.Wire
