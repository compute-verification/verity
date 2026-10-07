import CertifierDevice.Device
import Std.Sync.Channel

/-!
# A local world, and the driver that runs the device in it

`World` is what the device needs from its host, kept minimal so the shared world can replace it: a monotonic clock and a
sleep (ns), the boundary link (`Conn`: a blocking read, of no bytes at its end, a write, and its unwritten backlog), a
server connection per stream (`connect`), and where records go (`publish`).

`run` feeds `step`. Each blocking read runs in a task of its own (the link's, and each stream's once it connects), and
sends what it read, with the clock's reading, to one channel; so does each connection attempt. The driver takes those
in order, and when a grid's tick is due (`origin + t·τ`) it ticks it with the clock's reading then and, on egress, the
link's backlog. It performs each action as the device returns it: a write with `drain` is delivered once the write
returns. It returns when both grids have run all their ticks, leaving the reading tasks blocked where they are.
-/

namespace CertifierDevice

/-- A refusal: a connection the server refused. -/
structure Refused where
  reason : String

/-- A byte connection: a read blocks until bytes come and gives none at the end; `backlog` is what is still unwritten. -/
structure Conn where
  read : IO ByteArray
  write : ByteArray → IO Unit
  backlog : IO Nat

/-- What the device needs from its host. -/
structure World where
  clock : IO Nat
  sleep : Nat → IO Unit
  link : Conn
  connect : Nat → IO (Except Refused Conn)
  publish : Record → Params → IO Unit

/-- An event from a task, with the connection it opened. -/
abbrev Message := Event × Option Conn

/-- Read `conn` until its end, sending each read as `wrap now bytes`. -/
def pumpReads (w : World) (conn : Conn) (ch : Std.Channel.Sync Message) (wrap : Nat → ByteArray → Event) : IO Unit := do
  repeat
    let bytes ← conn.read
    ch.send (wrap (← w.clock) bytes, none)
    if bytes.isEmpty then break

/-- How long the driver sleeps at most while it waits for a tick, so that a read is handled within it (ns). -/
def pollNs : Nat := 200000

/-- **The driver.** -/
def run (w : World) (c : Config) : IO Unit := do
  match c.check with
  | .error e => throw (IO.userError e)
  | .ok () => pure ()
  let ch ← Std.Channel.new (α := Message)
  let ch := ch.sync
  let _ ← IO.asTask (prio := .dedicated) (pumpReads w w.link ch Event.link)
  let mut st := init c
  let mut conns : Std.HashMap Nat Conn := {}
  let mut queue : Array Event := #[]
  while !st.done do
    let ev? ← if h : 0 < queue.size then
        let ev := queue[0]
        queue := queue.extract 1 queue.size
        pure (some ev)
      else match ← ch.tryRecv with
        | some (ev, conn?) =>
          if let (.connected s _, some conn) := (ev, conn?) then conns := conns.insert s conn
          pure (some ev)
        | none =>
          let due := fun (p : Params) (t : Nat) => c.origin + t * p.bucketMs * nsPerMs
          let e := if st.shaper.t < st.ticks c.egress then some (due c.egress st.shaper.t) else none
          let i := if st.bucketer.t < st.ticks c.ingress then some (due c.ingress st.bucketer.t) else none
          let now ← w.clock
          match e, i with
          | some de, some di =>
            if de ≤ di && de ≤ now then pure (some (.tick .egress now (← w.link.backlog)))
            else if di ≤ now then pure (some (.tick .ingress now 0))
            else w.sleep (min (min de di - now) pollNs) *> pure none
          | some de, none =>
            if de ≤ now then pure (some (.tick .egress now (← w.link.backlog))) else w.sleep (min (de - now) pollNs) *> pure none
          | none, some di =>
            if di ≤ now then pure (some (.tick .ingress now 0)) else w.sleep (min (di - now) pollNs) *> pure none
          | none, none => pure none
    let some ev := ev? | continue
    let (st', acts) := step st ev
    st := st'
    for a in acts do
      match a with
      | .writeLink bytes => w.link.write bytes
      | .connect s =>
        let _ ← IO.asTask (prio := .dedicated) do
          match ← w.connect s with
          | .error _ => ch.send (.refused s (← w.clock), none)
          | .ok conn =>
            ch.send (.connected s (← w.clock), some conn)
            pumpReads w conn ch (Event.upstream s)
      | .writeUpstream s bytes drain =>
        if let some conn := conns.get? s then
          conn.write bytes
          if drain then queue := queue.push (.delivered s (← w.clock))
      | .publish record params => w.publish record params

end CertifierDevice
