import Flock
import Spot.HistoryTable

/-!
`table-root-time [--tables T] [--lengths L,...]`: the Lean verifier's SHA-512 (`Flock.Sha512.hash`) and the frame-v3
pieces a history table's root at its pinned capacity (`Spot.HistoryTable.CAPACITY`, `2^17` rows) is made of, each
timed, one `PHASE {json}` line each on stdout.

* `sha512`: the bare hash at a message length (`bytes`): a node frame's 277 bytes, a leaf frame's, a `b ‖ c` tree leaf's,
  and 1 MiB for the bulk rate;
* `node`, `leaf`, `tree_leaf`: `FV3.Sha512.node`, `FV3.Sha512.leafAs` under the table's `SCHEMA` and
  `Hm96.Default512.treeLeaf`, framing included;
* `root`: `FV3.root` over `2^17` frame leaves (its `2^17 - 1` nodes);
* `table`: `Spot.HistoryTable.rootAt`, one table's root from its rows' `b ‖ c` at a length `len ≤ 2^17`: each row's
  tree leaf and frame leaf, the placeholder's frame leaf at each rank from `len` up, and the root, `--tables` times.

Each phase's input varies with its loop index and its outputs fold into a checksum it prints, so no hash is hoisted out of
the timing. `table_root.sh` builds it against the checkout's verifier and runs it; it is a measurement, not a verifier.
-/

open Lean Flock

def CAP : Nat := Spot.HistoryTable.CAPACITY

def statusKb (field : String) : IO Nat := do
  let s ← IO.FS.readFile "/proc/self/status"
  for l in s.splitOn "\n" do
    if l.startsWith field then
      return ((String.ofList ((l.toList.drop field.length).filter Char.isDigit)).toNat?).getD 0
  return 0

/-- `n` bytes, the first eight `i`'s little-endian bytes and the rest `fill`. -/
def msg (n i : Nat) (fill : UInt8 := 0x5a) : ByteArray := Id.run do
  let mut b := ByteArray.mk (Array.replicate n fill)
  for k in [0:min 8 n] do b := b.set! k (i >>> (8 * k)).toUInt8
  return b

/-- A digest's first byte, folded into a checksum. -/
def fold (acc : UInt64) (d : ByteArray) : UInt64 := acc * 31 + (d.get! 0).toUInt64

def emit (row : List (String × Json)) : IO Unit := do
  let hwm ← statusKb "VmHWM:"
  IO.println s!"PHASE {(Json.mkObj (row ++ [("hwm_mb", toJson (hwm / 1024))])).compress}"
  (← IO.getStdout).flush

def nsNow : IO Nat := IO.monoNanosNow

def rate (calls ns : Nat) : List (String × Json) :=
  [("calls", toJson calls), ("ms", toJson (ns / 1000000)),
   ("ns_per_call", toJson (ns / max calls 1)),
   ("calls_per_s", toJson (calls * 1000000000 / max ns 1))]

def benchHash (bytes calls : Nat) : IO Unit := do
  let base := msg bytes 0
  let t0 ← nsNow
  let mut acc : UInt64 := t0.toUInt64
  for i in [0:calls] do
    acc := fold acc (Sha512.hash (base.set! 0 i.toUInt8 |>.set! 1 (i >>> 8).toUInt8 |>.set! 2 (i >>> 16).toUInt8))
  let t1 ← nsNow
  emit ([("phase", Json.str "sha512"), ("bytes", toJson bytes), ("mb_per_s", toJson (bytes * calls * 1000 / max (t1 - t0) 1))]
    ++ rate calls (t1 - t0) ++ [("check", toJson acc.toNat)])

def benchNode (calls : Nat) : IO Unit := do
  let dom := Sha512.hash (msg 8 1)
  let l := Sha512.hash (msg 8 2)
  let r := Sha512.hash (msg 8 3)
  let t0 ← nsNow
  let mut acc : UInt64 := t0.toUInt64
  for i in [0:calls] do
    acc := fold acc (FV3.Sha512.node dom (i % 17) i l r)
  let t1 ← nsNow
  emit ([("phase", Json.str "node")] ++ rate calls (t1 - t0) ++ [("check", toJson acc.toNat)])

def benchLeaf (calls : Nat) : IO Unit := do
  let dom := Sha512.hash (msg 8 1)
  let v := Sha512.hash (msg 8 4)
  let t0 ← nsNow
  let mut acc : UInt64 := t0.toUInt64
  for i in [0:calls] do
    acc := fold acc (FV3.Sha512.leafAs Spot.HistoryTable.SCHEMA dom i v)
  let t1 ← nsNow
  emit ([("phase", Json.str "leaf")] ++ rate calls (t1 - t0) ++ [("check", toJson acc.toNat)])

def benchTreeLeaf (calls : Nat) : IO Unit := do
  let t0 ← nsNow
  let mut acc : UInt64 := t0.toUInt64
  for i in [0:calls] do
    acc := fold acc (Hm96.Default512.treeLeaf (msg 128 i))
  let t1 ← nsNow
  emit ([("phase", Json.str "tree_leaf")] ++ rate calls (t1 - t0) ++ [("check", toJson acc.toNat)])

def benchRoot : IO Unit := do
  let dom := Sha512.hash (msg 8 1)
  let leaves := (Array.range CAP).map fun i => Sha512.hash (msg 16 i)
  let t0 ← nsNow
  let dom := if t0 == 0 then ByteArray.empty else dom
  let r := FV3.root dom leaves
  let t1 ← nsNow
  emit ([("phase", Json.str "root"), ("leaves", toJson CAP)] ++ rate (CAP - 1) (t1 - t0)
    ++ [("check", toJson (fold 0 r).toNat)])

def benchTable (len tables : Nat) : IO Unit := do
  let rows := (Array.range tables).map fun t => (Array.range len).map fun i => msg 128 (t * CAP + i)
  let placeholder := msg 128 0 0
  let t0 ← nsNow
  let mut acc : UInt64 := t0.toUInt64
  for t in [0:tables] do
    acc := fold acc (Spot.HistoryTable.rootAt CAP (Sha512.hash (msg 8 (t + 7))) rows[t]! placeholder)
  let t1 ← nsNow
  let calls := tables * (len + CAP + (CAP - 1) + 1)
  emit ([("phase", Json.str "table"), ("len", toJson len), ("tables", toJson tables),
    ("ms_per_table", toJson ((t1 - t0) / max tables 1 / 1000000))] ++ rate calls (t1 - t0)
    ++ [("check", toJson acc.toNat)])

def flag (args : List String) (name : String) : Option String :=
  match args.dropWhile (· != name) with
  | _ :: v :: _ => some v
  | _ => none

def main (args : List String) : IO UInt32 := do
  let tables := ((flag args "--tables").bind String.toNat?).getD 2
  let lengths := ((flag args "--lengths").map fun s => (s.splitOn ",").filterMap String.toNat?).getD [4096, 40960, CAP]
  for (bytes, calls) in [(277, 1000000), (237, 1000000), (208, 1000000), (128, 1000000), (64, 1000000),
      (1 <<< 20, 64)] do
    benchHash bytes calls
  benchNode 1000000
  benchLeaf 1000000
  benchTreeLeaf 1000000
  benchRoot
  for len in lengths do
    benchTable (min len CAP) tables
  return 0
