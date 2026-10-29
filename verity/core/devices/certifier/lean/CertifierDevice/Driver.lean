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
link's backlog; when both are due, ingress first (`active.TICK_ORDER`). It performs each action as the device returns
it: a write with `drain` is delivered once the write returns, as is each salt frame's write on the salt connection,
which `connectSalt` opens with `connect`. It closes a stream's connection when it takes the read that ends it, as
`active.py` closes its writer then, and when both grids have run all their ticks it closes the connections still open
and the link, as `Proxy.run` does, and returns. With the salt on, it draws every PoUW window's salt from the world's
coins before its first tick; the device reads one only once its frame is released.
-/

namespace CertifierDevice

open Core (Conn)

/-- What the device publishes: a closed window's record, with its grid's parameters. -/
abbrev Out := BlockGrid × Params

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

/-- The salt source of `c`'s salt, if on: one salt of its bits for every PoUW window of the run, from the device's own
coins. -/
def drawSalts (w : Core.World Out) (c : Config) : IO Config := do
  let some s := c.salt | return c
  let k := max s.pouwWindow 1
  let values ← (List.range ((c.windows + k - 1) / k)).toArray.mapM fun _ => w.coins (s.bits / 8)
  return { c with salt := some { s with values } }

/-- **The driver**, for a proxy that starts at window `start` with its first egress frame at `position`. -/
def run (w : Core.World Out) (c : Config) (start position : Nat := 0) : IO Unit := do
  let c ← drawSalts w c
  match c.check start with
  | .error e => throw (IO.userError e)
  | .ok () => pure ()
  let ch ← Std.Channel.new (α := Message)
  let ch := ch.sync
  let _ ← IO.asTask (prio := .dedicated) (pumpReads w w.link ch Event.link)
  let mut st := initAt c start position
  let mut conns : Std.HashMap Nat Conn := {}
  let mut saltConn : Option Conn := none
  let mut queue : Array Event := #[]
  while !st.done do
    let ev? ← if h : 0 < queue.size then
        let ev := queue[0]
        queue := queue.extract 1 queue.size
        pure (some ev)
      else match ← ch.tryRecv with
        | some (ev, conn?) =>
          if let (.connected s _, some conn) := (ev, conn?) then conns := conns.insert s conn
          if let (.saltConnected _, some conn) := (ev, conn?) then saltConn := some conn
          pure (some ev)
        | none =>
          let due := fun (p : Params) (t : Nat) => c.origin + t * p.bucketMs * nsPerMs
          let e := if st.shaper.t < st.ticks c.egress then some (due c.egress st.shaper.t) else none
          let i := if st.bucketer.t < st.ticks c.ingress then some (due c.ingress st.bucketer.t) else none
          let now ← w.clock
          match e, i with
          | some de, some di =>
            if di ≤ de && di ≤ now then pure (some (.tick .ingress now 0))
            else if de ≤ now then pure (some (.tick .egress now (← w.link.backlog)))
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
      | .connectSalt =>
        let _ ← IO.asTask (prio := .dedicated) do
          match ← w.connect with
          | .error _ => ch.send (.saltRefused (← w.clock), none)
          | .ok conn => ch.send (.saltConnected (← w.clock), some conn)
      | .writeSalt bytes =>
        if let some conn := saltConn then
          conn.write bytes
          queue := queue.push (.saltDelivered (← w.clock))
  for (_, conn) in conns do conn.close
  if let some conn := saltConn then conn.close
  w.link.close

end CertifierDevice
