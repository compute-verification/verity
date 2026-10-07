import CertifierDevice.Device
import World
import Std.Sync.Channel

/-!
# The driver that runs the device in the world

The device's host is `core_world`'s `Core.World`, whose outputs are the closed windows (`Out`: a record and its grid's
parameters): its clock and `sleepUntil`, the boundary link, a server connection per stream (`connect`, called once for
each), `publish`, and the signer's leaf state (`reserve`, which `certifyWith` takes a leaf from).

`run` feeds `step`. Each blocking read runs in a task of its own (the link's, and each stream's once it connects), and
sends what it read, with the clock's reading, to one channel; so does each connection attempt. The driver takes those
in order, and when a grid's tick is due (`origin + t·τ`) it ticks it with the clock's reading then and, on egress, the
link's backlog. It performs each action as the device returns it: a write with `drain` is delivered once the write
returns. It closes a stream's connection when it takes the read that ends it, as `active.py` closes its writer then, and
when both grids have run all their ticks it closes the connections still open and the link, as `Proxy.run` does, and
returns.
-/

namespace CertifierDevice

open Core (Conn)

/-- What the device publishes: a closed window's record, with its grid's parameters. -/
abbrev Out := Record × Params

/-- An event from a task, with the connection it opened. -/
abbrev Message := Event × Option Conn

/-- Read `conn` until its end, sending each read as `wrap now bytes`. -/
def pumpReads (w : Core.World Out) (conn : Conn) (ch : Std.Channel.Sync Message) (wrap : Nat → ByteArray → Event) :
    IO Unit := do
  repeat
    let bytes ← conn.read
    ch.send (wrap (← w.clock) bytes, none)
    if bytes.isEmpty then break

/-- How long the driver sleeps at most while it waits for a tick, so that a read is handled within it (ns). -/
def pollNs : Nat := 200000

/-- **The driver.** -/
def run (w : Core.World Out) (c : Config) : IO Unit := do
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
            else w.sleepUntil (min (min de di) (now + pollNs)) *> pure none
          | some de, none =>
            if de ≤ now then pure (some (.tick .egress now (← w.link.backlog)))
            else w.sleepUntil (min de (now + pollNs)) *> pure none
          | none, some di =>
            if di ≤ now then pure (some (.tick .ingress now 0)) else w.sleepUntil (min di (now + pollNs)) *> pure none
          | none, none => pure none
    let some ev := ev? | continue
    if let .upstream s _ bytes := ev then
      if bytes.isEmpty then
        if let some conn := conns.get? s then
          conn.close
          conns := conns.erase s
    let (st', acts) := step st ev
    st := st'
    for a in acts do
      match a with
      | .writeLink bytes => w.link.write bytes
      | .connect s =>
        let _ ← IO.asTask (prio := .dedicated) do
          match ← w.connect with
          | .error _ => ch.send (.refused s (← w.clock), none)
          | .ok conn =>
            ch.send (.connected s (← w.clock), some conn)
            pumpReads w conn ch (Event.upstream s)
      | .writeUpstream s bytes drain =>
        if let some conn := conns.get? s then
          conn.write bytes
          if drain then queue := queue.push (.delivered s (← w.clock))
      | .publish record params => w.publish (record, params)
  for (_, conn) in conns do conn.close
  w.link.close

end CertifierDevice
