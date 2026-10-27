import VStar.Trace
import Flock.ProvedScope
import Flock.RecCoins
import Flock.With.Setup
import Flock.With.Log

/-!
# `vstar-trace`: the trace assigner's side of its difftest

`vstar-trace CIRCUIT INST` reads a staged V\* part (its `circuit.txt`, parsed as the setup parses it,
`ProvedScope.circuitOf` under `Tags.circuit`, and its `inst.bin` with shared rows, decoded to its port inputs) and prints
`VStar.Trace.digests` under M0's zero mask as one JSON line: `m`, `k_log`, whether every port's row is whole 16-bit
words (`full_rows`, `VStar.Trace.fullRows`, which the port lemma's row width reads), each block's SHA-512 (`blocks`) and each block's
ranges' (`ranges`), hex. `flock-trace` (`verity/ml/flock/live/src/bin/flock-trace.rs`) prints the same of the witness the
deployed prover assigns; `tests/test_vstar_trace_difftest.py` holds the two equal.

`vstar-trace --sat --circuit F --public F --instances F [--partition F] [--program F]` checks that trace against the
table `verify --zk` sets up on the verifier's files (`TablesSat`, below), the witness run `HonestWitSatisfies` cites;
`--session F` sets it up at the draw `verify` reads from that session's record (`RecCoins.recordDraw`).

`vstar-trace --nocoll [--plant] NAME=DIR=SESSION ...` checks `RecDeploy.NoCollAt` at each part and over all of them:
no SHA-512 output with two different inputs among the hm96 calls the part's trace computes and the SHA-512 inputs its
setup hashes (`partCalls`); `--plant` adds one such pair, the control.
-/

namespace VStar.TraceCli

open Lean (Json)
open Flock

/-- A staged `inst.bin` with shared rows (`Instances::load`) as port inputs: the header line, each port's table rows in the
circuit's port order (`2 · words` bytes each, `shared_rows[name]` of them), their salts, their `b ‖ c`, then each instance's
refs (u32 little-endian, instance-major). A file with public words or without shared rows is refused. -/
def partInOf (c : Circuit) (file : ByteArray) : Except String VStar.In.PartIn := do
  let some nl := findByte file 10 | throw "inst.bin: no header line"
  let some text := String.fromUTF8? (file.extract 0 nl) | throw "inst.bin: the header is not UTF-8"
  let head ← Json.parse text
  let n ← head.getObjValAs? Nat "instances"
  let shared ← head.getObjVal? "shared_rows"
  let np := c.ports.size
  let counts ← c.ports.mapM fun p => shared.getObjValAs? Nat p.name
  let total := counts.foldl (· + ·) 0
  let starts := (counts.foldl (fun (acc : Array Nat × Nat) k => (acc.1.push acc.2, acc.2 + k)) (#[], 0)).1
  let tableAt := (Array.range np).foldl (fun (acc : Array Nat × Nat) p =>
    (acc.1.push acc.2, acc.2 + 2 * (c.ports.getD p default).words * counts[p]!)) (#[], nl + 1)
  let saltAt := tableAt.2
  let refsAt := saltAt + (HmRow.SALT_BYTES + HmRow.BC_BYTES) * total
  if refsAt + 4 * n * np != file.size then throw "inst.bin: not the shared-row layout without public words"
  let u32At := fun (at_ : Nat) => (List.range 4).foldl (fun v j => v + (file.get! (at_ + j)).toNat * 256 ^ j) 0
  pure ((Array.range np).map fun p =>
    let port := c.ports.getD p default
    let w := 2 * port.words
    let rows := (Array.range counts[p]!).map fun r =>
      let at_ := tableAt.1[p]! + w * r
      let s := saltAt + HmRow.SALT_BYTES * (starts[p]! + r)
      ({ words := VStar.In.wordsOf (file.extract at_ (at_ + w)), salt := file.extract s (s + HmRow.SALT_BYTES) } : VStar.In.Row)
    (port.name, { rows, refs := (Array.range n).map fun i => u32At (refsAt + 4 * (i * np + p)) }))

def run (circuitPath instPath : String) : IO UInt32 := do
  let cb ← IO.FS.readBinFile circuitPath
  let ib ← IO.FS.readBinFile instPath
  match ProvedScope.circuitOf Tags.circuit cb [] with
  | .error e => IO.eprintln s!"circuit: {e}"; return 2
  | .ok c =>
    match partInOf c ib with
    | .error e => IO.eprintln e; return 2
    | .ok pin =>
      match Trace.digests c pin (fun _ _ => false) with
      | none => IO.eprintln "refused: the circuit is outside what the trace assigner mirrors (VStar.Trace.supported)"; return 3
      | some ds =>
        IO.println (Json.compress (Json.mkObj [
          ("m", (Trace.m c pin : Json)), ("k_log", (c.kLog : Json)), ("instances", (Trace.count pin : Json)),
          ("full_rows", Json.bool (Trace.fullRows c)),
          ("blocks", Json.arr (ds.map fun d => Json.str (hexOf d.1))),
          ("ranges", Json.arr (ds.map fun d => Json.arr (d.2.map fun r => Json.str (hexOf r))))]))
        return 0

/-! ## `vstar-trace --sat`: the trace against the verifier's tables

`TablesSat` (`RecDeploy`, the premise `HonestWitSatisfies` reads), executed: each table `verify --zk` sets up on the
statement's files (`Stmt.setupTables` at `zkTags`, `zkCoins`), read as `Readable.Tab.ofStmt` (the level-0 matrices
`FlockLevel3.placedA`, `placedB` and the regions `Refine.regionsOf`), must fit (`Tab.Fits`), and the trace
(`VStar.Trace.assign`) must hold in every block (`Tab.Holds`, `tab_iff`'s side of `Satisfies`): each row
`(A₀·x) ∧ (B₀·x) = x`, the pin set, and every region bit its public value. -/

/-- Bit `t` of block `b`'s mask under seed `seed` (SplitMix64's finalizer): a mask as `--zk`'s uniform one varies. -/
def mixMask (seed : UInt64) (b t : Nat) : Bool :=
  let x := seed + b.toUInt64 * 0x9E3779B97F4A7C15 + t.toUInt64 * 0xD1B54A32D192ED03
  let x := (x ^^^ (x >>> 30)) * 0xBF58476D1CE4E5B9
  let x := (x ^^^ (x >>> 27)) * 0x94D049BB133111EB
  (x ^^^ (x >>> 31)) &&& 1 == 1

@[inline] def byteAt (b : ByteArray) (k : Nat) : UInt8 := if h : k < b.size then b[k] else 0

/-- A sparse matrix's rows as `FlockLevel3.sparseEntry` reads them: row `i < rows`'s columns, each counted mod 2. -/
def sparseRows (m : Sparse) : Array (Array Nat) := (Array.range m.rows).map m.row

/-- Table `st`'s level-0 matrices and regions, laid out for one pass over a block. `cov[j]`: the index of the last range
covering column `j` (`FlockLevel3.lastCover`), `255` where none does. `rowsA[g]`, `rowsB[g]`: range `g`'s local rows
(`slotEntryA`, `slotEntryB`), `none` for a mask's identity. `da[r]`, `db[r]`: Δ's columns at row `r`. `regs`: each
region's bits `(i, inBlock i)` that land in the block (`Tab.fixed`). -/
structure Placed where
  bs : Nat
  ranges : Array Range
  sizes : Array Nat
  cov : ByteArray
  rowsA : Array (Option (Array (Array Nat)))
  rowsB : Array (Option (Array (Array Nat)))
  da : Std.HashMap Nat (Array Nat)
  db : Std.HashMap Nat (Array Nat)
  regs : Array (Region × Array (Nat × Nat))

/-- Region `reg`'s bits in a block of `2^kl` bits (`Refine.regionOf`, `Region.inBlock`): bit `i < 2^|free|` sits at the
free bits from `i` and the other bits fixed where `mask` and `fixed` are both set, kept where that is in the block. -/
def regionAddrs (kl : Nat) (reg : Region) : Array (Nat × Nat) :=
  let free := (reg.free.toList.filter (· < kl)).toArray
  let c0 := (List.range kl).foldl (fun s j =>
    if !free.contains j && (reg.mask >>> j) % 2 == 1 && (reg.fixed >>> j) % 2 == 1 then s + 2 ^ j else s) 0
  (Array.range (2 ^ free.size)).filterMap fun i =>
    let a := (List.range free.size).foldl (fun s t => if i.testBit t then s + 2 ^ free[t]! else s) c0
    if a < 2 ^ kl then some (i, a) else none

def byRow (bs : Nat) (d : Array (Nat × Nat)) : Std.HashMap Nat (Array Nat) :=
  d.foldl (fun m (i, j) => if i < bs && j < bs then m.insert i ((m.getD i #[]).push j) else m) {}

def placed (st : Flock.Stmt) : Except String Placed := do
  let c := st.c
  let bs := 2 ^ c.kLog
  if c.nets.any (·.2.lookup.isSome) then throw "a net with a lookup table: outside what --sat places"
  if c.ranges.size ≥ 255 then throw "255 ranges or more: outside what --sat places"
  let netA := c.nets.map fun n => sparseRows n.2.a
  let netB := c.nets.map fun n => sparseRows n.2.b
  let rowsOf := fun (net : Array (Array (Array Nat))) (comp : Sparse) (r : Range) => match r.kind with
    | .net n => some (net.getD n #[])
    | .mask => none
    | .comp => some (sparseRows comp)
  let covers := fun (r : Range) (j : Nat) => r.pos0 ≤ j >>> r.slotLog && j >>> r.slotLog < r.pos0 + r.count
  let cov := Id.run do
    let mut out := ByteArray.emptyWithCapacity bs
    for j in [0:bs] do
      let mut g : UInt8 := 255
      for k in [0:c.ranges.size] do
        if covers c.ranges[k]! j then g := k.toUInt8
      out := out.push g
    return out
  pure { bs, ranges := c.ranges, sizes := c.ranges.map (2 ^ ·.slotLog), cov,
         rowsA := c.ranges.map (rowsOf netA st.leaf.comp.a), rowsB := c.ranges.map (rowsOf netB st.leaf.comp.b),
         da := byRow bs st.da, db := byRow bs st.db, regs := st.regions.map fun r => (r, regionAddrs c.kLog r) }

/-- Row `r` of one side of the level-0 matrix applied to block `x` (bit `j` is `x[j] = 1`): over GF(2), each range's
entries in `r`'s slot at the columns it covers last (`rangesEntry`), then Δ's (`deltaEntry`). -/
def side (p : Placed) (rows : Array (Option (Array (Array Nat)))) (dl : Std.HashMap Nat (Array Nat)) (x : ByteArray)
    (r : Nat) : Bool := Id.run do
  let mut acc := false
  for g in [0:p.ranges.size] do
    let rg := p.ranges[g]!
    let s := r >>> rg.slotLog
    if rg.pos0 ≤ s && s < rg.pos0 + rg.count then
      let base := s <<< rg.slotLog
      let sz := p.sizes[g]!
      let cols := match rows[g]! with
        | none => #[r - base]
        | some rs => rs.getD (r - base) #[]
      for i' in cols do
        let j := base + i'
        if i' < sz && j < p.bs && (byteAt p.cov j).toNat == g && byteAt x j == 1 then acc := !acc
  for j in dl.getD r #[] do
    if byteAt x j == 1 then acc := !acc
  return acc

structure Bad where
  rows : Nat := 0
  pin : Nat := 0
  region : Nat := 0
  /-- the first failure: block, kind, row or region bit -/
  first : Option (Nat × String × Nat) := none
  deriving Inhabited

def Bad.add (a b : Bad) : Bad :=
  { rows := a.rows + b.rows, pin := a.pin + b.pin, region := a.region + b.region, first := a.first <|> b.first }

/-- Block `o` of the trace (`bits`), a byte a bit. -/
def blockOf (bits : Nat → Bool) (bs o : Nat) : ByteArray := Id.run do
  let mut x := ByteArray.emptyWithCapacity bs
  for j in [0:bs] do
    x := x.push (if bits (o * bs + j) then 1 else 0)
  return x

/-- Block `o` of the trace (`x`) against table `p`: every row, the pin and every region bit. -/
def checkBlock (p : Placed) (pin : Nat) (x : ByteArray) (o : Nat) : Bad := Id.run do
  let mut bad : Bad := {}
  for r in [0:p.bs] do
    if (side p p.rowsA p.da x r && side p p.rowsB p.db x r) != (byteAt x r == 1) then
      bad := { bad with rows := bad.rows + 1, first := bad.first <|> some (o, "row", r) }
  if byteAt x pin != 1 then bad := { bad with pin := bad.pin + 1, first := bad.first <|> some (o, "pin", pin) }
  for (reg, addrs) in p.regs do
    let known := reg.bytes o
    for (i, a) in addrs do
      if (byteAt x a == 1) != ((byteAt known (i / 8) >>> (i % 8).toUInt8) % 2 == 1) then
        bad := { bad with region := bad.region + 1, first := bad.first <|> some (o, s!"region {reg.name}", i) }
  return bad

/-- `checkBlock` on a layout of the task's own: objects shared between tasks count their references atomically, and the
row loop's reads of a shared layout contend on them. -/
@[noinline] def checkOwn (st : Flock.Stmt) (x : ByteArray) (o : Nat) : Bad :=
  match placed st with
  | .ok p => checkBlock p st.pin x o
  | .error e => { rows := 1, first := some (o, e, 0) }

/-- `--draw F`, `--partition F`: JSON files. -/
def jsonArg (args : List String) (k : String) : IO (Except String (Option Json)) := do
  match args.dropWhile (· != k) with
  | _ :: f :: _ => pure ((Json.parse (← IO.FS.readFile f)).map some |>.mapError (s!"{k}: " ++ ·))
  | _ => pure (.ok none)

def INFLIGHT : Nat := 256

def satUsage : String :=
  "usage: vstar-trace --sat --circuit F --public F --instances F [--partition F] [--program F] [--draw F | --session F] [--mask zero|SEED] [--flip Q,Q,...]"

def sat (args : List String) : IO UInt32 := do
  let arg : String → Option String := fun k => match args.dropWhile (· != k) with | _ :: v :: _ => some v | _ => none
  let (some cf, some pf, some inf) := (arg "--circuit", arg "--public", arg "--instances") | IO.eprintln satUsage; return 1
  let cb ← IO.FS.readBinFile cf
  let pb ← IO.FS.readBinFile pf
  let ib ← IO.FS.readBinFile inf
  let program ← (arg "--program").mapM IO.FS.readBinFile
  let (.ok partition, .ok draw) := (← jsonArg args "--partition", ← jsonArg args "--draw") | IO.eprintln satUsage; return 1
  let draw ← match arg "--session" with
    | some f => pure (RecCoins.recordDraw (← IO.FS.readFile f))
    | none => pure draw
  let mask : Nat → Nat → Bool ← match arg "--mask" with
    | none | some "zero" => pure fun _ _ => false
    | some s => match s.toNat? with
      | some n => pure (mixMask n.toUInt64)
      | none => IO.eprintln satUsage; return 1
  let some (flips : List Nat) := ((((arg "--flip").getD "").splitOn ",").filter (· ≠ "")).mapM String.toNat?
    | IO.eprintln satUsage; return 1
  let coins := Coins.tree (Zk.coinSpec (Tags.circuit.tableNames 1))
  let tags := Zk.tags (Tags.circuit.withCoins coins) (Zk.maskLogOf cb) 1
  let sts ← match Stmt.setupTables tags cb pb coins 1 [] draw partition program with
    | .ok (_, sts) => pure sts
    | .error e => IO.eprintln s!"setup: {e}"; return 2
  let some st := sts[0]? | IO.eprintln "setup: no table"; return 2
  if sts.size != 1 then IO.eprintln s!"setup: {sts.size} tables; --sat checks a one-table session"; return 3
  let c ← match ProvedScope.circuitOf Tags.circuit cb [] with
    | .ok c => pure c
    | .error e => IO.eprintln s!"circuit: {e}"; return 2
  let pin ← match partInOf c ib with
    | .ok p => pure p
    | .error e => IO.eprintln e; return 2
  if !Trace.supported c then
    IO.eprintln "refused: the circuit is outside what the trace assigner mirrors (VStar.Trace.supported)"; return 3
  let kl := st.c.kLog
  let fits := 6 ≤ kl && kl ≤ st.m && st.pin < 2 ^ kl
  let p ← match placed st with
    | .ok p => pure p
    | .error e => IO.eprintln s!"refused: {e}"; return 3
  let shapeOk := (Zk.shapeOk st).toBool
  let tm := Trace.m c pin
  let head := [("m", (st.m : Json)), ("k_log", (kl : Json)), ("pin", (st.pin : Json)), ("trace_m", (tm : Json)), ("trace_k_log", (c.kLog : Json)),
    ("drawn", Json.bool draw.isSome), ("fits", Json.bool fits), ("shape_ok", Json.bool shapeOk), ("ranges", (st.c.ranges.size : Json)),
    ("regions", (st.regions.size : Json)), ("region_bits", ((p.regs.foldl (fun (n : Nat) r => n + r.2.size) 0 : Nat) : Json)),
    ("delta", ((st.da.size + st.db.size : Nat) : Json)), ("flips", (flips.length : Json))]
  if !fits || tm != st.m || c.kLog != kl then
    IO.println (Json.compress (Json.mkObj (("ok", Json.bool false) :: head))); return 0
  let t0 ← IO.monoMsNow
  let z := Trace.slots c pin
  let f := Trace.bitAtT z mask (Trace.bitTables c)
  let fl : Std.HashSet Nat := flips.foldl (·.insert ·) {}
  let bits := if flips.isEmpty then f else fun q => f q != fl.contains q
  -- the trace's blocks are built here, so no task reads `z`; at most `INFLIGHT` blocks wait for their check
  let mut tasks : Array (Task Bad) := #[]
  let mut bad : Bad := {}
  let mut next := 0
  for o in [0:2 ^ (st.m - kl)] do
    let x := blockOf bits (2 ^ kl) o
    tasks := tasks.push (Task.spawn fun _ => checkOwn st x o)
    if tasks.size - next ≥ INFLIGHT then
      bad := bad.add tasks[next]!.get
      next := next + 1
  for t in tasks.extract next tasks.size do
    bad := bad.add t.get
  let first := match bad.first with
    | some (o, k, i) => Json.mkObj [("block", (o : Json)), ("kind", Json.str k), ("at", (i : Json))]
    | none => Json.null
  IO.println (Json.compress (Json.mkObj ([("ok", Json.bool (bad.rows + bad.pin + bad.region == 0 && shapeOk))] ++ head ++
    [("blocks", ((2 ^ (st.m - kl) : Nat) : Json)), ("rows_bad", (bad.rows : Json)), ("pin_bad", (bad.pin : Json)),
     ("region_bad", (bad.region : Json)), ("first", first), ("ms", (((← IO.monoMsNow) - t0 : Nat) : Json))])))
  return 0

/-! ## `--nocoll`: no SHA-512 digest with two inputs (`RecDeploy.NoCollAt`) -/

/-- SHA-512's chaining words as its digest's bytes (big-endian). -/
def beBytes (ws : Array UInt64) : ByteArray :=
  ⟨ws.flatMap fun w => (Array.range 8).map fun j => (w >>> (8 * (7 - j)).toUInt64).toUInt8⟩

/-- Instance `i`'s port-`p` hm96 calls as its slots compute them: the salt's `SHA-512(saltPrefix ‖ y)` and the row's
`SHA-512(prefix ‖ row)` (`HmRow.prefixOf`, the message `Trace.chain` compresses), with whether the row's digest is the
chain's last compression, the one the port's `hm96` slot computes. -/
def hmCallsAt (c : Circuit) (pin : In.PartIn) (p i : Nat) : Array (ByteArray × ByteArray) × Bool :=
  let port := c.ports.getD p default
  let (msg, cvs) := Trace.chain c pin p i
  let rowIn := HmRow.prefixOf port ++ msg.extract 0 port.bytes
  let d := Sha512.hash rowIn
  let sIn := Hm96.Default512.saltPrefix ++ (Trace.portIn c pin p i).2
  (#[(sIn, Sha512.hash sIn), (rowIn, d)], (beBytes (Sha512.compress cvs.back! msg (128 * port.chunks))).data == d.data)

/-- `(input, output)` pairs by output (hex): `findColl`'s search, the first output with two different inputs kept. -/
structure Calls where
  calls : Nat := 0
  byOut : Std.HashMap String ByteArray := {}
  others : Nat := 0
  coll : Option (String × Nat × Nat) := none

def Calls.add (s : Calls) (u d : ByteArray) : Calls :=
  let k := hexOf d
  let s := { s with calls := s.calls + 1 }
  match s.byOut[k]? with
  | none => { s with byOut := s.byOut.insert k u }
  | some u' => if u'.data == u.data then s else { s with others := s.others + 1, coll := s.coll <|> some (k, u'.size, u.size) }

def Calls.json (s : Calls) : List (String × Json) :=
  [("calls", (s.calls : Json)), ("distinct_digests", (s.byOut.size : Json)),
   ("distinct_inputs", ((s.byOut.size + s.others : Nat) : Json)), ("coll", Json.bool s.coll.isSome),
   ("first_coll", match s.coll with
     | some (k, a, b) => Json.mkObj [("digest", (k : Json)), ("sizes", Json.arr #[(a : Json), (b : Json)])]
     | none => Json.null)]

def nocollUsage : String :=
  "usage: vstar-trace --nocoll [--plant] NAME=DIR=SESSION ...  (DIR: circuit.txt, pub.bin, inst.bin, partition.json, program.json)"

/-- One part's calls: every hm96 salt and row call its circuit computes at its trace's rows (every port of every
instance, padding ones included, a superset of the calls `layR` records there), then every SHA-512 input
`flock-verify`'s setup hashes at the part's statement and session's draw (`Stmt.setupTablesWith` under the logging
oracle, `vcallsAt`'s `D.setup`). -/
def partCalls (dir session : String) : IO (Except String (Array (ByteArray × ByteArray) × List (String × Json))) := do
  let cb ← IO.FS.readBinFile s!"{dir}/circuit.txt"
  let pb ← IO.FS.readBinFile s!"{dir}/pub.bin"
  let ib ← IO.FS.readBinFile s!"{dir}/inst.bin"
  let program ← IO.FS.readBinFile s!"{dir}/program.json"
  let some partition := (Json.parse (← IO.FS.readFile s!"{dir}/partition.json")).toOption
    | return .error "partition.json: not JSON"
  let draw := RecCoins.recordDraw (← IO.FS.readFile session)
  let coins := Coins.tree (Zk.coinSpec (Tags.circuit.tableNames 1))
  let tags := Zk.tags (Tags.circuit.withCoins coins) (Zk.maskLogOf cb) 1
  let t0 ← IO.monoMsNow
  let (res, log) := (Flock.Stmt.setupTablesWith (Flock.Hashes.logged Sha512.hash) tags cb pb coins 1 [] draw (some partition)
    (some program)).go #[]
  let t1 ← IO.monoMsNow
  let sts ← match res with
    | .ok (_, sts) => pure sts
    | .error e => return .error s!"setup: {e}"
  if sts.size != 1 then return .error s!"setup: {sts.size} tables"
  let c ← match ProvedScope.circuitOf Tags.circuit cb [] with
    | .ok c => pure c
    | .error e => return .error s!"circuit: {e}"
  let pin ← match partInOf c ib with
    | .ok p => pure p
    | .error e => return .error e
  if !Trace.supported c then return .error "the circuit is outside what the trace assigner mirrors"
  let n := 2 ^ (Trace.m c pin - c.kLog) * c.g
  let mut out : Array (ByteArray × ByteArray) := #[]
  let mut chainBad := 0
  for i in [0:n] do
    for p in [0:c.ports.size] do
      let (cs, ok) := hmCallsAt c pin p i
      out := out ++ cs
      unless ok do chainBad := chainBad + 1
  let hm := out.size
  out := out ++ log.map fun u => (u, Sha512.hash u)
  pure (.ok (out, [("instances", (Trace.count pin : Json)), ("slots", (n : Json)), ("ports", (c.ports.size : Json)),
    ("hm96_calls", (hm : Json)), ("setup_calls", (log.size : Json)), ("chain_bad", (chainBad : Json)),
    ("drawn", Json.bool draw.isSome), ("setup_ms", ((t1 - t0 : Nat) : Json))]))

/-- `--nocoll`: per part, and over all the parts together, no output of SHA-512 with two different inputs among
`partCalls`. `--plant` adds to the first part one call with a recorded output and an input one byte longer: the
control, which must be found. -/
def nocoll (args : List String) : IO UInt32 := do
  let plant := args.contains "--plant"
  let parts := args.filter fun a => !a.startsWith "--"
  if parts.isEmpty then IO.eprintln nocollUsage; return 1
  let mut all : Calls := {}
  let mut ok := true
  let mut first := true
  for a in parts do
    let (nm, dir, session) ← match a.splitOn "=" with
      | [nm, d, s] => pure (nm, d, s)
      | _ => IO.eprintln nocollUsage; return 1
    let (cs, info) ← match ← partCalls dir session with
      | .ok r => pure r
      | .error e => IO.eprintln s!"{nm}: {e}"; return 2
    let cs := if plant && first then match cs[0]? with
      | some (u, d) => cs.push (u.push 0, d)
      | none => cs
      else cs
    first := false
    let one := cs.foldl (fun s (u, d) => s.add u d) ({} : Calls)
    all := cs.foldl (fun s (u, d) => s.add u d) all
    let bad := info.any fun (k, v) => k == "chain_bad" && v != (0 : Json)
    ok := ok && one.coll.isNone && !bad
    IO.println (Json.compress (Json.mkObj ([("part", (nm : Json))] ++ one.json ++ info)))
  ok := ok && all.coll.isNone
  IO.println (Json.compress (Json.mkObj ([("part", Json.str "all"), ("ok", Json.bool ok), ("planted", Json.bool plant)] ++ all.json)))
  return 0

end VStar.TraceCli

def main (args : List String) : IO UInt32 :=
  match args with
  | "--sat" :: rest => VStar.TraceCli.sat rest
  | "--nocoll" :: rest => VStar.TraceCli.nocoll rest
  | [c, i] => VStar.TraceCli.run c i
  | _ => do IO.eprintln s!"usage: vstar-trace CIRCUIT INST\n       {VStar.TraceCli.satUsage}"; return 1
