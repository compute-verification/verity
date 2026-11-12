module

public import NetworkCheck.ProbeOkBit

/-! The `network-okbit-probe` executable, a probe and not for landing: `NetworkCheck.ProbeOkBit`'s three forms of the
rule on a registration, each with its verdict and its time a call, and the time of one constant.

    network-okbit-probe rule REGISTRATION.json T REPS
    network-okbit-probe const LEAVES REPS -/

@[expose] public section

namespace NetworkCheck.ProbeOkBit

open Lean (Json)
open Flock

/-- `f k` for each `k < reps`: nanoseconds a call, and how many returned true. Each call reads `k`, so none is shared. -/
def timed (reps : Nat) (f : Nat → Bool) : IO (Nat × Nat) := do
  let t0 ← IO.monoNanosNow
  let mut ok := 0
  for k in [0:reps] do
    if f k then ok := ok + 1
  let t1 ← IO.monoNanosNow
  return ((t1 - t0) / reps, ok)

def ruleMain (path : String) (T reps : Nat) : IO UInt32 := do
  let .ok reg := Json.parse (← IO.FS.readFile path)
    | do IO.eprintln s!"{path}: not JSON"; return 2
  let .ok es := entries reg
    | do IO.eprintln s!"{path}: {verdict (entries reg)}"; return 2
  let pub := es.filter fun e => publicO0 (fun _ => false) e.port
  IO.println s!"OKBIT registration roots={es.size} public_o0={pub.size} leaves={pub.foldl (fun n e => n + e.leaves) 0} T={T}"
  for e in pub do
    IO.println s!"OKBIT entry port={e.port} leaves={e.leaves} domain={hexOf (e.domain.extract 0 8)} root={hexOf (e.root.extract 0 8)} \
      ok_root={hexOf ((okRoot e.leaves e.domain).extract 0 8)} calls={okRootCalls e.leaves}"
  let (ns, n) ← timed reps fun k => accepts (okWalk reg (T + k - k))
  IO.println s!"OKBIT walk verdict={verdict (okWalk reg T)} ns={ns} accepted={n}/{reps} \
    calls={pub.foldl (fun n e => n + okRootCalls e.leaves) 0}"
  let (ns, n) ← timed reps fun k => accepts (okCached reg (T + k - k))
  IO.println s!"OKBIT cached verdict={verdict (okCached reg T)} ns={ns} accepted={n}/{reps}"
  match okCached reg T with
  | .error _ => pure ()
  | .ok cs =>
    for c in cs do
      IO.println s!"OKBIT const leaves={c.leaves} domain={hexOf (c.domain.extract 0 8)} root={hexOf c.root} calls={okRootCalls c.leaves}"
    let (ns, n) ← timed reps fun k => accepts (checkCarried (cs.extract 0 (cs.size + k - k)))
    IO.println s!"OKBIT carried-once verdict={verdict (checkCarried cs)} ns={ns} accepted={n}/{reps} constants={cs.size} \
      calls={cs.foldl (fun n c => n + okRootCalls c.leaves) 0}"
    let many := 1000 * reps
    let (ns, n) ← timed many fun k => accepts (okCarried cs reg (T + k - k))
    IO.println s!"OKBIT carried-compare verdict={verdict (okCarried cs reg T)} ns={ns} accepted={n}/{many} calls=0"
  return 0

def constMain (leaves reps : Nat) : IO UInt32 := do
  let domain := Sha512.hash "a probe's domain".toUTF8
  let (ns, n) ← timed reps fun k => (okRoot (leaves + k - k) domain).size == 64
  IO.println s!"OKBIT const-time leaves={leaves} ns={ns} done={n}/{reps} calls={okRootCalls leaves} \
    root={hexOf ((okRoot leaves domain).extract 0 8)}"
  let msg := ByteArray.mk (Array.replicate 320 0)
  let (ns, _) ← timed (100 * reps) fun k => (Sha512.hash (msg.push k.toUInt8)).size == 64
  IO.println s!"OKBIT sha512 bytes=321 ns={ns}"
  return 0

end NetworkCheck.ProbeOkBit

def main (args : List String) : IO UInt32 :=
  match args with
  | ["rule", path, t, reps] => NetworkCheck.ProbeOkBit.ruleMain path t.toNat! reps.toNat!
  | ["const", n, reps] => NetworkCheck.ProbeOkBit.constMain n.toNat! reps.toNat!
  | _ => do
    IO.eprintln "usage: network-okbit-probe rule REGISTRATION.json T REPS | const LEAVES REPS"
    return 2
