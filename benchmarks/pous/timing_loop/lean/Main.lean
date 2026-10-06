import Flock.Hash
import TimingLoop.Frame
import TimingLoop.Wire

/-!
`timing-loop`: PoUS's sequential-reveal timing loop with Lean holding the clock (`IO.monoNanosNow`), two ways.

* `own --port P --challenges FILE --store FILE --block B --k K --pings N --limit-ns L --out FILE [--check digest --leaves
  FILE --prefixes FILE]`: Lean owns the socket. It connects to the responder, sends each index of `FILE` (8 little-endian
  bytes each, `k` to an audit, drawn by the Python side as `audit.MemoryChallenger` draws them), times each round from just
  before the send to the whole answer, and checks the answer byte for byte against block `i` of the store, or with `--check
  digest` as `VerifierKey.check` does: `Flock.Sha256` of leaf `i`'s 35-byte prefix and the block against leaf `i`.
* `drive --k K --audits A --pings N --limit-ns L --out FILE -- CMD ARGS…`: Lean drives a Python worker (`CMD ARGS…`,
  `helper.py`) that keeps the socket and the verifier. Each round asks it for the challenge (`D`), then times the reveal
  (`R`: the worker sends it and replies with the answer's header once the whole answer is in), then asks for the verdict
  (`V`). Every request is one frame (`TimingLoop.Frame`).

Both write their rounds to `--out` as little-endian u64 rows (`bench.py` reads them) and print one `RESULT {json}` line:
audits, the audits with every answer correct and within `L` on Lean's clock, and the late and wrong answers.
-/

open TimingLoop TimingLoop.Wire

def argOf (args : List String) (k : String) : Option String :=
  match args.dropWhile (· != k) with
  | _ :: v :: _ => some v
  | _ => none

def natOf (args : List String) (k : String) (d : Nat) : Nat := ((argOf args k).bind String.toNat?).getD d

def workerCmd (args : List String) : Option (String × Array String) :=
  match args.dropWhile (· != "--") with
  | _ :: cmd :: rest => some (cmd, rest.toArray)
  | _ => none

def rows (xs : Array Nat) : ByteArray := Id.run do
  let mut out := ByteArray.empty
  for x in xs do
    out := out ++ u64le x
  return out

def result (audits accepted late wrong : Nat) : String :=
  s!"RESULT \{\"audits\": {audits}, \"accepted\": {accepted}, \"late\": {late}, \"wrong\": {wrong}}"

/-- The pings, then the rounds: per ping `t0 t1 server device recvs`; per round `index t0 t1 server device ok recvs`. -/
def ownCmd (args : List String) : IO UInt32 := do
  let (some chal, some storePath, some out) := (argOf args "--challenges", argOf args "--store", argOf args "--out")
    | IO.eprintln "own: --challenges FILE --store FILE --out FILE"; return 2
  let (port, block, k, pings, limit) :=
    (natOf args "--port" 0, natOf args "--block" 65536, natOf args "--k" 111, natOf args "--pings" 0, natOf args "--limit-ns" 0)
  let digest := argOf args "--check" == some "digest"
  let store ← IO.FS.readBinFile storePath
  let leaves ← if digest then IO.FS.readBinFile ((argOf args "--leaves").getD "") else pure .empty
  let prefixes ← if digest then IO.FS.readBinFile ((argOf args "--prefixes").getD "") else pure .empty
  let raw ← IO.FS.readBinFile chal
  let n := raw.size / 8
  let s ← connect port
  for _ in [0:50] do
    discard <| roundtrip s block opPing 0
  let mut ps : Array Nat := Array.emptyWithCapacity (5 * pings)
  for _ in [0:pings] do
    let a ← roundtrip s block opPing 0
    ps := ps ++ #[a.sentNs, a.receivedNs, a.serverNs, a.deviceNs, a.recvs]
  let mut rs : Array Nat := Array.emptyWithCapacity (7 * n)
  let mut accepted := 0
  let mut late := 0
  let mut wrong := 0
  let mut good := true
  for j in [0:n] do
    let i := ofU64le raw (8 * j)
    let a ← roundtrip s block opChallenge i
    let answer := a.bytes.extract header (header + block)
    let ok := a.status == 0 && a.bytes.size == header + block &&
      if digest then Flock.Sha256.hash (prefixes.extract (35 * i) (35 * i + 35) ++ answer) == leaves.extract (32 * i) (32 * i + 32)
      else answer == store.extract (i * block) ((i + 1) * block)
    let onTime := a.receivedNs - a.sentNs ≤ limit
    if !ok then wrong := wrong + 1
    if !onTime then late := late + 1
    good := good && ok && onTime
    if (j + 1) % k == 0 then
      if good then accepted := accepted + 1
      good := true
    rs := rs ++ #[i, a.sentNs, a.receivedNs, a.serverNs, a.deviceNs, if ok then 1 else 0, a.recvs]
  close s
  IO.FS.writeBinFile out (u64le pings ++ rows ps ++ u64le n ++ rows rs)
  IO.println (result (n / k) accepted late wrong)
  return 0

def ask (w : IO.Process.Child ⟨.piped, .piped, .inherit⟩) (body : String) (max : Nat) : IO ByteArray := do
  Frame.write w.stdin body.toUTF8
  Frame.read w.stdout max

/-- The pings, then the rounds: per ping `t0 t1 server device worker_rtt`; per round `index draw0 draw1 t0 t1 verdict1
server device worker_rtt correct worker_on_time`. -/
def driveCmd (args : List String) : IO UInt32 := do
  let some (cmd, wargs) := workerCmd args | IO.eprintln "drive: the worker's command follows --"; return 2
  let some out := argOf args "--out" | IO.eprintln "drive: --out FILE"; return 2
  let (k, audits, pings, limit) := (natOf args "--k" 111, natOf args "--audits" 1, natOf args "--pings" 0, natOf args "--limit-ns" 0)
  let w ← IO.Process.spawn { cmd, args := wargs, stdin := .piped, stdout := .piped, stderr := .inherit }
  for _ in [0:50] do
    discard <| ask w "P" 32
  let mut ps : Array Nat := Array.emptyWithCapacity (5 * pings)
  for _ in [0:pings] do
    let t0 ← IO.monoNanosNow
    let r ← ask w "P" 32
    let t1 ← IO.monoNanosNow
    ps := ps ++ #[t0, t1, ofU64le r 16, ofU64le r 8, ofU64le r 24]
  let mut rs : Array Nat := Array.emptyWithCapacity (11 * k * audits)
  let mut accepted := 0
  let mut late := 0
  let mut wrong := 0
  for _ in [0:audits] do
    discard <| ask w "A" 0
    let mut good := true
    for _ in [0:k] do
      let d0 ← IO.monoNanosNow
      let i := ofU64le (← ask w "D" 8) 0
      let d1 ← IO.monoNanosNow
      let t0 ← IO.monoNanosNow
      let r ← ask w "R" 32
      let t1 ← IO.monoNanosNow
      let v ← ask w "V" 2
      let v1 ← IO.monoNanosNow
      let ok := ofU64le r 0 == 0 && v.get! 0 == 1
      let onTime := t1 - t0 ≤ limit
      if !ok then wrong := wrong + 1
      if !onTime then late := late + 1
      good := good && ok && onTime
      rs := rs ++ #[i, d0, d1, t0, t1, v1, ofU64le r 16, ofU64le r 8, ofU64le r 24, (v.get! 0).toNat, (v.get! 1).toNat]
    if good then accepted := accepted + 1
  discard <| ask w "Q" 0
  discard <| w.wait
  IO.FS.writeBinFile out (u64le pings ++ rows ps ++ u64le (k * audits) ++ rows rs)
  IO.println (result audits accepted late wrong)
  return 0

def main (args : List String) : IO UInt32 :=
  match args with
  | "own" :: rest => ownCmd rest
  | "drive" :: rest => driveCmd rest
  | _ => do IO.eprintln "usage: timing-loop own|drive …"; return 2
