/-!
# The outside world, as a core device or challenger reaches it

A device (the network certifier) or a challenger (PoUS's memory challenger) is a pure state machine: a `step` from its
state and one event to its next state and the actions it takes. Its properties are about `step`. A small driver runs
`step` over a `World`, the functions here, as the Python's asyncio tasks do with the `clock`, `coins` and `sleep` they are
given.

The analysis states its assumptions about the traces a `World` produces, not about these functions: to Lean's logic an
`IO` action is opaque. Those are the verifier's coins (A3, `UniformRandomBytes`, and A5 for the window secret), the clock's
drift against real time, and `trusted-nebius-host` for a signer's state (an LMS key's next leaf and the session ids it
has started), which is state the device keeps on its host. Everything else a `World` returns (link bytes, a refused
connection, an unpublished record) is adversarial, and a device fails closed on it.
-/

namespace Core

/-- One byte stream: the boundary link, or a connection to the server for one stream. `read` returns the next bytes as
they arrive and an empty array once the peer has closed its side; the device frames them itself. `readUntil t` is the
same read, but returns `none` once `clock` reads at least `t` with nothing arrived, so a driver waits on a deadline in a
plain loop. `backlog` is the bytes written and not yet sent (asyncio's `get_write_buffer_size`), which a device reads to
refuse an overrun. -/
structure Conn where
  read : IO ByteArray
  readUntil : Nat → IO (Option ByteArray)
  write : ByteArray → IO Unit
  backlog : IO Nat
  close : IO Unit

/-- What a device or challenger may do outside its own state. `Out` is what it publishes (the certifier's closed
window, its record and grid parameters).

- `coins n`: `n` bytes of the verifier's own randomness.
- `clock`: monotonic nanoseconds. A property reads only differences of it, against a deadline or a tick's slack.
- `sleepUntil t`: wait until `clock` reads at least `t`. It may return late, never early, and the device sees how late on
  `clock`.
- `link`: the boundary link.
- `connect`: a new connection upstream, or the reason it was refused.
- `publish`: hand an output to the service.
- `reserve`: the next unused LMS leaf, which the leaf state has advanced past on disk before this returns, or the reason
  the signer refuses (missing, another key's, behind its journal, held by another signer).
- `start s`: session `s` recorded as started in the signer's state on disk before this returns, or the reason the signer
  refuses (`s` already started, or the same refusals as `reserve`). -/
structure World (Out : Type) where
  coins : Nat → IO ByteArray
  clock : IO Nat
  sleepUntil : Nat → IO Unit
  link : Conn
  connect : IO (Except String Conn)
  publish : Out → IO Unit
  reserve : IO (Except String Nat)
  start : ByteArray → IO (Except String Unit)

/-- The operating system's randomness, as A3 states it: `IO.getRandomBytes`. -/
def World.coinsOS (n : Nat) : IO ByteArray := IO.getRandomBytes n.toUSize

/-- The operating system's monotonic clock, in nanoseconds. -/
def World.clockOS : IO Nat := IO.monoNanosNow

/-- `IO.sleep` takes whole milliseconds, so this waits the remaining time rounded up to one: at most a millisecond late,
plus the scheduler's own lateness. -/
def World.sleepUntilOS (t : Nat) : IO Unit := do
  let now ← IO.monoNanosNow
  if now < t then IO.sleep ((t - now + 999999) / 1000000).toUInt32

end Core
