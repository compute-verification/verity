import Flock.HmRow
import VStar.In

/-!
# The trace assigner: the witness the deployed C-Flock prover assigns a part's circuit at its port inputs

`assign c pin mask` is the hidden vector `flock-circuit prove` commits for circuit `c` at the port inputs `pin`
(`VStar.In.PartIn`, what the stager writes to the part's `inst.bin`), block-major: bit `col` of block `b` at
`b · 2^kLog + col`, `2^m` bits in all (`m = kLog + nbl`, `Stmt::new`'s). It mirrors `live/src/circuit.rs` step for step:

* **The rows' slots** (`rowIns`, `Stmt::row_inputs`): per block, VU and port, the row's message after its prefix (its
  bytes, then SHA-512's padding) chained from the prefix's midstate; each compression's chaining value and block in its
  `sha512x3` slot, and the last chaining value, the last block and the row's salt in the port's `hm96` slot. A padding
  instance's rows are the padding instance's (`HmOut.dummyRow`) under zero salts.
* **The units' slots** (`unitIns`, `Stmt::unit_inputs`): each unit's leaf words from its instance's rows, 16 bits a leaf
  (and its leaf cuts); a padding instance's are zero.
* **Every slot's values** (`evalNet`, `IrUnitNet::eval64`): its input rows, the constant, then every computed row in
  order. The rows' wires hold by construction, so with no other wire (`supported`) nothing converges
  (`Stmt::converge_with` returns the inputs as they are).
* **The witness** (`bitAt`, `Stmt::witness_with`): the mask range's bits from `mask`, every slot's values at its
  slot, zero elsewhere.

`mask b t` is bit `t` of block `b`'s mask range: zero under M0 (`NoMask::mask_fill`), uniform under `--zk`.
`assign` refuses (`none`) what V\*'s circuits don't use: a typed template, a lookup, a segmented row, and a wire into a
slot other than a row's (a tail, which `converge` would iterate). `digests` is what the difftest compares
(`flock-trace`, `vstar-trace`): each block's and each range's SHA-512 of its bits packed eight to a byte, low bit first.
-/

namespace VStar.Trace

open Flock

/-- The instances: one ref per instance in every port (`inst.bin`'s `instances`). -/
def count (pin : In.PartIn) : Nat := (pin[0]?.map (·.2.refs.size)).getD 0

/-- Port `nm`'s row at instance `i`: row `refs[i]` of its table (`Instances::instance_rows`' gather). -/
def rowAt (pin : In.PartIn) (nm : String) (i : Nat) : Option In.Row := do
  let p ← (pin.find? (·.1 == nm)).map (·.2)
  let r ← p.refs[i]?
  p.rows[r]?

def zeros (n : Nat) : ByteArray := ⟨Array.replicate n 0⟩

/-- A row's first `words` u16 words as bytes, each little-endian (zeros past the row). -/
def wordBytes (ws : Array Nat) (words : Nat) : ByteArray :=
  ⟨(Array.range words).flatMap fun i => #[(ws.getD i 0 % 256).toUInt8, (ws.getD i 0 / 256 % 256).toUInt8]⟩

/-- A salt as `SALT_BYTES` bytes (zeros past it). -/
def saltOf (s : ByteArray) : ByteArray := s.extract 0 HmRow.SALT_BYTES ++ zeros (HmRow.SALT_BYTES - s.size)

/-- Instance `i`'s port-`p` row and salt: its table's; past the instances, the padding instance's row under a zero salt. -/
def portIn (c : Circuit) (pin : In.PartIn) (p i : Nat) : ByteArray × ByteArray :=
  let port := c.ports.getD p default
  if i < count pin then
    match rowAt pin port.name i with
    | some r => (wordBytes r.words port.words, saltOf r.salt)
    | none => (zeros (2 * port.words), zeros HmRow.SALT_BYTES)
  else (HmOut.dummyRow c p, zeros HmRow.SALT_BYTES)

/-- Instance `i`'s rows, ports side by side (`Instances::instance_rows`). -/
def instRow (c : Circuit) (pin : In.PartIn) (i : Nat) : ByteArray :=
  (List.range c.ports.size).foldl (fun acc p => acc ++ (portIn c pin p i).1) ByteArray.empty

/-- `put`: u64 word `i` of `ws` at columns `col + 64 i`, its bit `j` at `+ j`. -/
def putWords (w : Array Bool) (col : Nat) (ws : Array UInt64) : Array Bool :=
  (List.range (64 * ws.size)).foldl
    (fun acc t => acc.set! (col + t) ((ws.getD (t / 64) 0 >>> (t % 64).toUInt64) &&& 1 == 1)) w

/-- Bytes at columns `col + 8 j + t`, byte `j`'s bit `t` (`put` of their little-endian u64 words). -/
def putBytes (w : Array Bool) (col : Nat) (b : ByteArray) : Array Bool :=
  (List.range (8 * b.size)).foldl (fun acc t => acc.set! (col + t) ((b.get! (t / 8)).toNat.testBit (t % 8))) w

/-- Block `b`'s row slots' inputs (`Stmt::row_inputs`): its `c.g · nSha` `sha512x3` slots, then its `c.g · perVu[hm]` `hm96`
slots, each VU's in order. -/
def rowIns (c : Circuit) (pin : In.PartIn) (b : Nat) : Array (Array Bool) × Array (Array Bool) := Id.run do
  let sha := (c.nets.getD c.shaNet default).2
  let hm := (c.nets.getD c.hmNet default).2
  let sin := portCols sha.inGroups
  let hin := portCols hm.inGroups
  let (nSha, where_) := HmRow.rowSlots c.ports
  let perHm := c.perVu.getD c.hmNet 0
  let mut si := Array.replicate (c.g * nSha) (Array.replicate (sha.inWords * WORD) false)
  let mut hi := Array.replicate (c.g * perHm) (Array.replicate (hm.inWords * WORD) false)
  for g in [0:c.g] do
    for p in [0:c.ports.size] do
      let port := c.ports.getD p default
      let (row, salt) := portIn c pin p (b * c.g + g)
      let msg := row.extract 0 port.bytes ++ zeros (port.bytes - row.size) ++ HmRow.tailOf port
      let mut cv := HmRow.midstateOf port
      for k in [0:port.chunks] do
        let (slot, sub) := (where_.getD p #[]).getD k (0, 0)
        let blk := HmRow.wordsBE (msg.extract (128 * k) (128 * k + 128)) 16
        si := si.modify (g * nSha + slot) fun w =>
          putWords (putWords w (sin.getD (2 * sub) default).1 cv) (sin.getD (2 * sub + 1) default).1 blk
        cv := Sha512.compress cv msg (128 * k)
      let pad := HmRow.wordsBE (msg.extract (128 * port.chunks) msg.size) 16
      hi := hi.modify (g * perHm + p) fun w =>
        putBytes (putWords (putWords w (hin.getD 0 default).1 cv) (hin.getD 1 default).1 pad) (hin.getD 2 default).1 salt
  return (si, hi)

/-- A u16 ORed in at column `col` (`w |= v << col % 128` on the column's 64-bit half: a bit past the half is dropped). -/
def orU16 (w : Array Bool) (col v : Nat) : Array Bool :=
  (List.range 16).foldl (fun acc t => if col % 64 + t < 64 && v.testBit t then acc.set! (col + t) true else acc) w

def u16At (b : ByteArray) (at_ : Nat) : Nat := (b.get! at_).toNat + 256 * (b.get! (at_ + 1)).toNat

/-- Unit slot `k` of block `b`'s inputs (`Stmt::unit_inputs`): VU `k / upv`'s instance's row words at the unit's leaves,
16 bits each from its first input group's first column (a zero leaf, `-1`, left zero), and at its leaf cuts; zero for a
padding instance. -/
def unitIns (c : Circuit) (pin : In.PartIn) (b k : Nat) : Array Bool :=
  let unit := c.unit
  let w0 := Array.replicate (unit.inWords * WORD) false
  let (g, u) := (k / c.upv, k % c.upv)
  let i := b * c.g + g
  if i < count pin then
    let rows := instRow c pin i
    let g0 := unit.inGroups.getD 0 default
    let li := c.leavesIn.getD u #[]
    let w1 := ((List.range li.size).foldl (fun (acc : Array Bool × Nat) j =>
        let leaf := li.getD j (-1)
        ((if 0 ≤ leaf then orU16 acc.1 acc.2 (u16At rows (2 * leaf.toNat)) else acc.1), acc.2 + g0.bits.getD j 0))
      (w0, g0.col * WORD)).1
    let pc := portCols unit.inGroups
    c.leafCuts.foldl (fun acc x => if x.1 == u then orU16 acc (pc.getD x.2.1 default).1 (u16At rows (2 * x.2.2)) else acc) w1
  else w0

/-- A slot's values from its inputs (`IrUnitNet::eval64`): each input row (a self row) its input bit, the constant 1, every
computed row `(A z)(B z)` in order; `2^unitLog` bits, zero past the useful rows. -/
def evalNet (net : Net) (ins : Array Bool) : Array Bool := Id.run do
  let first := net.inWords * WORD
  let mut z := Array.replicate (2 ^ net.unitLog) false
  for i in [0:first] do
    if (net.a.row i).size == 1 then z := z.set! i (ins.getD i false)
  z := z.set! net.constPos true
  for i in [first:net.useful] do
    if i != net.constPos then
      let x := (net.a.row i).foldl (fun s k => s ^^ z.getD k false) false
      let y := (net.b.row i).foldl (fun s k => s ^^ z.getD k false) false
      z := z.set! i (x && y)
  return z

/-- The blocks' count's log, `max 3 (log2ceil blocks_real)` (`Stmt::new`). -/
def nbl (c : Circuit) (pin : In.PartIn) : Nat := max 3 (log2ceil (max ((count pin + c.g - 1) / c.g) 1))

/-- The witness's log size, `kLog + nbl`. -/
def m (c : Circuit) (pin : In.PartIn) : Nat := c.kLog + nbl c pin

/-- Every slot's values, per circuit and slot `x = block · count + k` of its range (`Stmt::converge`, `Stmt::slot_zab`):
the rows' slots from `rowIns`, the units' from `unitIns`, any other circuit's from zero inputs. -/
def slots (c : Circuit) (pin : In.PartIn) : Array (Array (Array Bool)) :=
  let nb := 2 ^ nbl c pin
  let rows := (Array.range nb).map (rowIns c pin)
  (Array.range c.nets.size).map fun n =>
    let net := (c.nets.getD n default).2
    let cnt := (c.rangeOf n).count
    if n == c.shaNet then (rows.flatMap (·.1)).map (evalNet net)
    else if n == c.hmNet then (rows.flatMap (·.2)).map (evalNet net)
    else if n == c.unitNet then (Array.range (nb * cnt)).map fun x => evalNet net (unitIns c pin (x / cnt) (x % cnt))
    else (Array.range (nb * cnt)).map fun _ => evalNet net (Array.replicate (net.inWords * WORD) false)

/-- Witness bit `q` (`Stmt::witness_with`): in block `q / 2^kLog` at column `q % 2^kLog`, the value of the slot covering it
(a circuit's range, slot `k` at `slot k`, one slot's `2^unitLog` bits; the last circuit's where two cover it), else the mask
range's bit, else zero. `checkLayout` makes the ranges disjoint and a unit range's slots its circuit's size. -/
def bitAt (c : Circuit) (z : Array (Array (Array Bool))) (mask : Nat → Nat → Bool) (q : Nat) : Bool :=
  let blk := q / 2 ^ c.kLog
  let col := q % 2 ^ c.kLog
  let fromNet := (List.range c.nets.size).reverse.findSome? fun n =>
    let r := c.rangeOf n
    let k := (col - r.start) / 2 ^ r.slotLog
    let t := col - r.slot k
    let zs := z.getD n #[]
    if r.start ≤ col && k < r.count && t < 2 ^ (c.nets.getD n default).2.unitLog && blk * r.count + k < zs.size then
      some ((zs.getD (blk * r.count + k) #[]).getD t false)
    else none
  match fromNet with
  | some v => v
  | none =>
    match c.ranges.toList.reverse.find? (fun r => r.kind == Kind.mask && r.start ≤ col && col < r.start + 2 ^ r.slotLog) with
    | some r => mask blk (col - r.start)
    | none => false

/-- What the assigner mirrors: no typed template, no lookup, no segmented row, and no wire but the rows' (into a
`sha512x3` or `hm96` slot). -/
def supported (c : Circuit) : Bool :=
  c.typed.isNone && c.nets.all (·.2.lookup.isNone) && c.ports.all (·.segs.isNone) &&
    c.wires.all fun w => w.dst.1 == c.shaNet || w.dst.1 == c.hmNet

/-- **The trace**: the hidden vector the deployed prover assigns circuit `c` at port inputs `pin` under mask `mask`,
`2^m` bits block-major; `none` where the circuit is outside what it mirrors (`supported`). -/
def assign (c : Circuit) (pin : In.PartIn) (mask : Nat → Nat → Bool) : Option (List Bool) :=
  if supported c then
    let z := slots c pin
    some ((List.range (2 ^ m c pin)).map (bitAt c z mask))
  else none

/-- Bits `[lo, lo + n)` of `f`, eight to a byte, the low bit first. -/
def pack (f : Nat → Bool) (lo n : Nat) : ByteArray := Id.run do
  let mut out := ByteArray.empty
  for j in [0:n / 8] do
    let mut v : Nat := 0
    for t in [0:8] do
      if f (lo + 8 * j + t) then v := v + 2 ^ t
    out := out.push v.toUInt8
  return out

/-- What the difftest compares: per block, the SHA-512 of its `2^kLog` bits, and of each range's slots' bits (in
`c.ranges`' order), under mask `mask`. -/
def digests (c : Circuit) (pin : In.PartIn) (mask : Nat → Nat → Bool) : Option (Array (ByteArray × Array ByteArray)) :=
  if supported c then
    let z := slots c pin
    let f := bitAt c z mask
    let bs := 2 ^ c.kLog
    some ((Array.range (2 ^ nbl c pin)).map fun b =>
      (Sha512.hash (pack f (b * bs) bs),
       c.ranges.map fun r => Sha512.hash (pack f (b * bs + r.start) (r.count * 2 ^ r.slotLog))))
  else none

end VStar.Trace
