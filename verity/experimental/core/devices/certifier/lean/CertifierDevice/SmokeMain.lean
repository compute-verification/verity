import CertifierDevice.Issuer

/-! A smoke run of `CertifierDevice.run` in a `Core.World` on the operating system's clock and timer, with an in-memory
loopback link and server: a peer that has written two requests on the link, and a server that answers each request
(`GET /s`) with one event, at position s − 1, once it is written. Two windows of five 20 ms buckets; it fails unless all
four records are published complete, the egress grids carry both events, and the link got every bucket's bytes. Then it
certifies the four records with a toy key of four leaves, reserved from the world's leaf counter, and fails unless each
certificate verifies and a fifth reservation is refused and listed. Run by hand (`lake exe certifier-device-smoke`); it
reads the wall clock, so no test runs it. -/

open CertifierDevice NetworkCertifier

def ms : Nat := 1000000

def egressParams : Params := { buckets := 5, allocation := 1, frameBytes := 64, bucketMs := 20, queueCapacity := 16 }
def ingressParams : Params := { buckets := 5, allocation := 2, frameBytes := 32, bucketMs := 20, queueCapacity := 16 }

def syncSet : SyncSet := ⟨[2], [1], by decide, by decide, by decide, by decide, by decide⟩

def sync : ClockSync := ⟨2, 1, by decide⟩

/-- A server connection: once a request `GET /s` is written, one event at position `s − 1`, then its end. -/
def server : IO Core.Conn := do
  let reads ← Std.Channel.new (α := ByteArray)
  let reads := reads.sync
  return { read := reads.recv, readUntil := fun _ => some <$> reads.recv, backlog := pure 0, close := pure (),
           write := fun request => do
             let s := (((String.fromUTF8? request).getD "").drop 5).toString.toNat!
             reads.send s!"id: {s - 1}\ndata: hello {s}\n\n".toUTF8
             reads.send ByteArray.empty }

def main : IO UInt32 := do
  let link ← Std.Channel.new (α := ByteArray)
  let link := link.sync
  for s in [1, 2] do
    let frame := (encodeFrame kindLast s s!"GET /{s}".toUTF8 ingressParams.frameBytes).getD ByteArray.empty
    link.send frame
  let wrote ← IO.mkRef ByteArray.empty
  let records ← IO.mkRef (#[] : Array Record)
  let next ← IO.mkRef 0
  let now ← IO.monoNanosNow
  let config : Config := { egress := egressParams, ingress := ingressParams, syncSet, syncs := #[sync, sync],
                           windows := 2, origin := now + 20 * ms, slack := 15 * ms }
  let world : Core.World Out := {
    coins := Core.World.coinsOS, clock := Core.World.clockOS, sleepUntil := Core.World.sleepUntilOS,
    link := { read := link.recv, readUntil := fun _ => some <$> link.recv, write := fun b => wrote.modify (· ++ b),
              backlog := pure 0, close := pure () },
    connect := .ok <$> server,
    publish := fun (r, _) => records.modify (·.push r),
    reserve := do
      let q ← next.get
      if q ≥ 4 then return .error "all 2^2 leaves are used"
      next.set (q + 1)
      return .ok q }
  run world config
  let records ← records.get
  let wrote ← wrote.get
  for r in records do
    let frames := r.grid.toList.flatMap fun row => row.toList.filterMap id
    let status := match Status.reason? r.status with | none => "complete" | some x => reasonValue x
    IO.println s!"{r.link} window {r.window}: {status}, {frames.length} frames"
  let events := records.filter (·.link == config.egressLink) |>.toList.flatMap fun r =>
    r.grid.toList.flatMap fun row => row.toList.filterMap fun slot => slot.map fun f => (decodeFrame f).2.2
  IO.println s!"link: {wrote.size} bytes written; egress events: {events.map String.fromUTF8!}"
  let ok := records.size == 4 && records.all (Status.reason? ·.status |>.isNone) && events.length == 2 &&
    wrote.size == 2 * egressParams.buckets * egressParams.allocation * egressParams.frameBytes
  let lms ← IO.ofExcept (Lms.Params.make 8 4 2 true)
  let key ← IO.ofExcept (Lms.keygen lms "smoke!!!".toUTF8)
  let mut issuer : Issuer := { key }
  let mut verified := 0
  for r in records ++ records.extract 0 1 do
    let (direction, params, ss) := if r.link == config.egressLink then (.egress, egressParams, some syncSet)
      else (.ingress, ingressParams, none)
    let (i, c?) ← certifyWith world issuer r direction params (← IO.ofExcept (parameters params ss))
    issuer := i
    if let some c := c? then
      if c.verify lms key.publicKey then verified := verified + 1
  IO.println s!"certified: {verified} verified, {issuer.refused.size} refused"
  let ok := ok && verified == 4 && issuer.refused.size == 1
  IO.println (if ok then "smoke: PASS" else "smoke: FAIL")
  IO.Process.exit (if ok then 0 else 1)
