import CertifierDevice.World

/-! A smoke run of `CertifierDevice.run` on the host's clock, in an in-memory loopback world: a peer that has written two
requests on the link, and a server that answers each request with one event once it is written. Two windows of five 20 ms
buckets; it fails unless all four records are published complete, the egress grids carry both events, and the link got
every bucket's bytes. Run by hand (`lake exe certifier-device-smoke`); it reads the wall clock, so no test runs it. -/

open CertifierDevice NetworkCertifier

def ms : Nat := 1000000

def egressParams : Params := { buckets := 5, allocation := 1, frameBytes := 64, bucketMs := 20, queueCapacity := 16 }
def ingressParams : Params := { buckets := 5, allocation := 2, frameBytes := 32, bucketMs := 20, queueCapacity := 16 }

def syncSet : SyncSet := ⟨[2], [1], by decide, by decide, by decide, by decide, by decide⟩

def sync : ClockSync := ⟨2, 1, by decide⟩

/-- Stream `s`'s server: once its request is written, one event at position `s − 1`, then its end. -/
def server (s : Nat) : IO Conn := do
  let reads ← Std.Channel.new (α := ByteArray)
  let reads := reads.sync
  return { read := reads.recv, backlog := pure 0,
           write := fun _ => do
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
  let now ← IO.monoNanosNow
  let config : Config := { egress := egressParams, ingress := ingressParams, syncSet, syncs := #[sync, sync],
                           windows := 2, origin := now + 20 * ms, slack := 15 * ms }
  let world : World := {
    clock := IO.monoNanosNow,
    sleep := fun ns => IO.sleep ((ns + ms - 1) / ms).toUInt32,
    link := { read := link.recv, write := fun b => wrote.modify (· ++ b), backlog := pure 0 },
    connect := fun s => .ok <$> server s,
    publish := fun r _ => records.modify (·.push r) }
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
  IO.println (if ok then "smoke: PASS" else "smoke: FAIL")
  IO.Process.exit (if ok then 0 else 1)
