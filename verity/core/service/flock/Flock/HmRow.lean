/-
PROTOCOL.md §16.10: `verity/flock-circuit` from PR #83 `eb90718f`. Serving rows are `hm96-sha512/row/v1` leaves proved in
the circuit: per VU, each input port's row is hashed by SHA-512 compressions in `sha512x3` slots (three per slot, chained from
the prefix's midstate) and closed by an `hm96` slot (the padding compression, then `b ‖ c` from the row's private salt). Both
slot circuits are sections of the circuit file, which must be the nets the verifier generates (`Flock.HmNets`). Every identity and binding digest is SHA-512: the circuit pin, the public
digest, the statement digest (Flock's own 32-byte slot is its first half), Σ and the record's digests.

A port's rows are `sha512/row/v1` rows of whole SHA-512 blocks or, when META names each port's `row_schemas`, core's bit
rows `sha512/row/v2` (`hm96-sha512/row/v2` leaves) of any u64 length `l` in bits: their `ceil(l / 16)` u16 leaves hold the
row's bits low first, and the digest hashes `P(l)` and the row's `ceil(l / 8)` bytes. From `l` alone the verifier derives
the prefix, its midstate, the message's blocks and every message bit at or past `l` (the last byte's fill bits, zero, and
SHA-512's padding, in a block with row bytes or in its own), which Δ pins as constants; a unit's leaf bit past `l` reads
its slot's forced-zero row.

The verifier evaluates no part of the computation: it recomputes the rows' constants, the wiring and Δ from the pinned
circuit, the frame-v3-sha512 roots from the public `b ‖ c` and outputs, and the pinned dummy `b ‖ c` from the zero row.
-/
import Flock.Statement
import Flock.TemplateQuery
import Flock.Typed
import Flock.Extract
import Flock.Qcall
import Flock.HmNets
import Flock.RecOpen
import Flock.RecResiduals
import Flock.RecSparse
import Flock.RecOpening

namespace Flock

open Lean (Json)

namespace HmRow

def SHA_NET : String := "sha512x3"
def HM_NET : String := "hm96"
def SCHEME : String := "frame-v3-sha512/hm96-sha512"
def ROW_SCHEMA : String := "hm96-sha512/row/v1"
def ROW_SCHEMA_V2 : String := "hm96-sha512/row/v2"
def ROW_SCHEMA_SEG : String := "hm96-sha512/row-seg/v1"
/-- A `sha512/row-seg/v1` segment, in bits (1 KiB; the last may be shorter). -/
def ROW_SEG_BITS : Nat := 8192
def PER_SLOT : Nat := 3
/-- A bit row's compressions at most: one 1024-bit block per compression, within the largest block a proof covers (2^26). -/
def MAX_COMPS : Nat := 2 ^ 16
/-- A block's `k_log` at most: the largest a fast100 schedule covers (`setupH` refuses past it, after the layout). -/
def MAX_K_LOG : Nat := 27
/-- META's bindings (what the statement binds about the circuit): a class leaves them out. -/
def BINDING_KEYS : List String := ["partition", "program_sha512", "program_digests"]
def PARTITION_V1 : String := "verity/partition/v1"
def PARTITION_PROVISIONAL : String := "flock-circuit/unit-cover/v0"

def isSha512Hex (s : String) : Bool := s.length == 128 && s.all (fun c => c.isDigit || ('a' ≤ c && c ≤ 'f'))

/-- The class of the instances a circuit file proves: SHA-512 of the file with META's bindings removed (META re-serialized
canonically), which a partition can name before it is bound. -/
def classOf (bytes : ByteArray) (head : String) (mj : Json) : V ByteArray := do
  let some n1 := findByte bytes 10 | throw "circuit: no META line"
  let some n2 := findByte bytes 10 (n1 + 1) | throw "circuit: no section after META"
  let m := match mj with
    | .obj kvs => Json.mkObj (kvs.toArray.filter (fun kv => !BINDING_KEYS.contains kv.1)).toList
    | j => j
  return Sha512.hash (head.toUTF8 ++ "\nMETA ".toUTF8 ++ (← canon m).toUTF8 ++ "\n".toUTF8 ++ bytes.extract (n2 + 1) bytes.size)
def SALT_BYTES : Nat := 192
def BC_BYTES : Nat := 128

/-- `sha512/row/v1`'s first block for a row of `words` u16 words (role 1, 16-bit words). -/
def rowPrefix (words : Nat) : ByteArray :=
  Hm96.pad ("verity/sha512-row/v1".toUTF8.push 0 ++ ByteArray.mk #[1, 16] ++ be32 words.toUInt32) 128

/-- The padding block after `total` bytes (a multiple of 128): `0x80`, zeros, the 128-bit bit length. -/
def padBlock (total : Nat) : ByteArray := Id.run do
  let mut b := (ByteArray.mk #[0x80]) ++ ByteArray.mk (Array.replicate 111 0)
  let bits := 8 * total
  for i in [0:16] do b := b.push ((bits >>> (8 * (15 - i))) % 256).toUInt8
  return b

def wordsBE (b : ByteArray) (n : Nat) : Array UInt64 :=
  (Array.range n).map fun i => (List.range 8).foldl (fun x j => (x <<< 8) ||| (b.get! (8 * i + j)).toUInt64) 0

def midstate (words : Nat) : Array UInt64 := Sha512.compress Sha512.iv (rowPrefix words) 0
def padWords (words : Nat) : Array UInt64 := wordsBE (padBlock (128 + 2 * words)) 16

/-- `sha512/row/v2`'s first block `P(l)` for a row of `l` bits: `"verity/sha512-row/v2\0"`, then `l` as a u64 big-endian,
zero-padded to 128 bytes (`verity.commitments.rowleaf.sha512_row_v2_prefix`). -/
def rowPrefixV2 (l : Nat) : ByteArray :=
  Hm96.pad ("verity/sha512-row/v2".toUTF8.push 0 ++ ByteArray.mk ((Array.range 8).map fun i => (l / 256 ^ (7 - i) % 256).toUInt8)) 128

def be64 (x : Nat) : ByteArray := ByteArray.mk ((Array.range 8).map fun i => (x / 256 ^ (7 - i) % 256).toUInt8)

/-- `sha512/row-seg/v1`'s outer first block `Q(l)`: `"verity/sha512-row-seg/v1\0"`, `l` and `s = ceil(l / 8192)` as u64s and
1024 as a u32, big-endian, zero-padded to 128 bytes (`verity.commitments.rowleaf.sha512_row_seg_prefix`). -/
def rowPrefixSeg (l : Nat) : ByteArray :=
  Hm96.pad ("verity/sha512-row-seg/v1".toUTF8.push 0 ++ be64 l ++ be64 ((l + ROW_SEG_BITS - 1) / ROW_SEG_BITS) ++ be32 1024) 128

/-- Segment `j` of `s`'s first block `S(j, s, m)`, `m` its bits: `"verity/sha512-row-seg/v1/segment\0"`, `j`, `s`, `m` as u64s
big-endian, zero-padded to 128 bytes (`verity.commitments.rowleaf.sha512_row_seg_segment_prefix`). -/
def segmentPrefix (j s m : Nat) : ByteArray :=
  Hm96.pad ("verity/sha512-row-seg/v1/segment".toUTF8.push 0 ++ be64 j ++ be64 s ++ be64 m) 128

/-- A `sha512/row-seg/v1` row of `l` bits: each segment's bits and compressions (its message's blocks after its constant
first block), then the outer message's (the `64 s` digest bytes, then SHA-512's padding). -/
def segShape (l : Nat) : Array (Nat × Nat) × Nat :=
  let s := (l + ROW_SEG_BITS - 1) / ROW_SEG_BITS
  ((Array.range s).map fun j => let m := min (l - ROW_SEG_BITS * j) ROW_SEG_BITS; (m, ((m + 7) / 8 + 17 + 127) / 128),
   (64 * s + 17 + 127) / 128)

/-- Bits as a number, the first least significant. -/
def bitsVal : List Bool → Nat
  | [] => 0
  | b :: bs => (if b then 1 else 0) + 2 * bitsVal bs

/-- A bit row's bytes (`sha512/row/v2`): bit `i` is bit `i % 8` of byte `i / 8`, the last byte zero-filled. -/
def packBits (bs : List Bool) : List UInt8 :=
  (List.range ((bs.length + 7) / 8)).map fun j => (bitsVal ((bs.drop (8 * j)).take 8)).toUInt8

/-- SHA-512's padding after a message of `total` bytes: `0x80`, zeros to 112 mod 128, the 128-bit bit length (a multiple
of 128: `padBlock`). -/
def padTail (total : Nat) : ByteArray := Id.run do
  let mut b := (ByteArray.mk #[0x80]) ++ ByteArray.mk (Array.replicate ((239 - total % 128) % 128) 0)
  let bits := 8 * total
  for i in [0:16] do b := b.push ((bits >>> (8 * (15 - i))) % 256).toUInt8
  return b

/-- A row's first block: `sha512/row/v1`'s prefix for its words, a bit row's `P(bits)`, a segmented row's outer `Q(bits)`. -/
def prefixOf (p : Port) : ByteArray :=
  if p.segs.isSome then rowPrefixSeg p.bits else if p.v2 then rowPrefixV2 p.bits else rowPrefix p.words

/-- The chaining value after a row's first block. -/
def midstateOf (p : Port) : Array UInt64 := Sha512.compress Sha512.iv (prefixOf p) 0

/-- A row's padding after its bytes (`padTail`; a `sha512/row/v1` row's is a whole block; a segmented row's after its outer
message's `64 s` digest bytes). -/
def tailOf (p : Port) : ByteArray := padTail (128 + match p.segs with | some s => 64 * s.size | none => p.bytes)

/-- A segmented row's segment `j` (of bits `m`): the chaining value after its first block, and its padding. -/
def segMidstate (p : Port) (j : Nat) : Array UInt64 :=
  let s := p.segs.getD #[]
  Sha512.compress Sha512.iv (segmentPrefix j s.size (s.getD j default).1) 0
def segTail (p : Port) (j : Nat) : ByteArray := padTail (128 + (((p.segs.getD #[]).getD j default).1 + 7) / 8)

/-- `sha512/row-seg/v1`'s digest of a row of `l` bits, its bytes `row` (`verity.commitments.rowleaf.sha512_row_seg_digest`). -/
def segDigest (l : Nat) (row : ByteArray) : ByteArray :=
  let segs := (segShape l).1
  Sha512.hash ((List.range segs.size).foldl (fun acc j =>
    let m := segs[j]!.1
    acc ++ Sha512.hash (segmentPrefix j segs.size m ++ row.extract (1024 * j) (1024 * j + (m + 7) / 8))) (rowPrefixSeg l))

/-- Compression `k`'s chain start, if it is one: segment `j`'s first compression (`some j`), the outer message's first
(`some s`; a serial row's first, `some 0`). Every other compression reads the previous one's output. -/
def chainStart (p : Port) (k : Nat) : Option Nat :=
  match p.segs with
  | none => if k == 0 then some 0 else none
  | some segs => Id.run do
    let mut first := 0
    for j in [0:segs.size] do
      if k == first then return some j
      first := first + segs[j]!.2
      if k < first then return none
    return if k == first then some segs.size else none

/-- Row byte `byte`'s compression and its offset in the block. -/
def byteAt (p : Port) (byte : Nat) : Nat × Nat :=
  match p.segs with
  | none => (byte / 128, byte % 128)
  | some segs =>
    let j := byte / 1024
    ((segs.extract 0 j).foldl (· + ·.2) 0 + byte % 1024 / 128, byte % 128)

/-- Column `t` of a message block's 1024 (word `t / 64`, its bit `t % 64`; the block's bytes big-endian per word) as the
block's byte and the byte's bit. -/
def blockBit (t : Nat) : Nat × Nat := (8 * (t / 64) + 7 - t % 64 / 8, t % 8)

/-- Bit `t` (a block column) of block `k` of row `p`'s message after the prefix: `none` for a row bit (`8 j + b < bits`), else
the constant there (a fill bit of the last row byte, zero, or SHA-512's padding `tl`). -/
def constBit (p : Port) (tl : ByteArray) (k t : Nat) : Option Bool :=
  let j := 128 * k + (blockBit t).1
  let b := (blockBit t).2
  if 8 * j + b < p.bits then none
  else some (decide (p.bytes ≤ j) && ((tl.get! (j - p.bytes)).toNat >>> b) % 2 == 1)

/-- Message bit `t` (a block column) of a port's compression `k` (`k = chunks`: its hm96 slot's block). -/
inductive MsgBit where
  /-- the row's bit -/
  | row (i : Nat)
  | const (v : Bool)
  /-- a segment's digest bit: column `col` of the output of the port's compression `k` (the segment's last) -/
  | digest (k col : Nat)
  deriving BEq, Repr

def msgBit (p : Port) (k t : Nat) : MsgBit :=
  let (o, b) := blockBit t
  match p.segs with
  | none => match constBit p (tailOf p) k t with
    | some v => .const v
    | none => .row (8 * (128 * k + o) + b)
  | some segs => Id.run do
    let mut first := 0
    for j in [0:segs.size] do
      let (m, n) := segs[j]!
      if k < first + n then
        let at_ := 128 * (k - first) + o
        let bytes := (m + 7) / 8
        if 8 * at_ + b < m then return .row (ROW_SEG_BITS * j + 8 * at_ + b)
        return .const (decide (bytes ≤ at_) && (((segTail p j).get! (at_ - bytes)).toNat >>> b) % 2 == 1)
      first := first + n
    let at_ := 128 * (k - first) + o
    let d := 64 * segs.size
    if at_ < d then
      let q := at_ % 64
      return .digest ((segs.extract 0 (at_ / 64 + 1)).foldl (· + ·.2) 0 - 1) (64 * (q / 8) + 56 - 8 * (q % 8) + b)
    return .const ((((tailOf p).get! (at_ - d)).toNat >>> b) % 2 == 1)

def hex16 (w : UInt64) : String :=
  let s := hexOf ((List.range 8).foldl (fun b j => b.push ((w >>> (8 * (7 - j)).toUInt64) &&& 0xff).toUInt8) ByteArray.empty)
  s

def bytesBE (ws : Array UInt64) : ByteArray :=
  ws.foldl (fun b w => (List.range 8).foldl (fun b j => b.push ((w >>> (8 * (7 - j)).toUInt64) &&& 0xff).toUInt8) b) ByteArray.empty

/-- Per VU: the `sha512x3` slots, and each port's compressions as `(slot, sub)` (three per slot, ports in order). -/
def rowSlots (ports : Array Port) : Nat × Array (Array (Nat × Nat)) := Id.run do
  let mut at_ := 0
  let mut where_ := #[]
  for p in ports do
    where_ := where_.push ((Array.range p.chunks).map fun k => ((at_ + k) / PER_SLOT, (at_ + k) % PER_SLOT))
    at_ := at_ + p.chunks
  return ((at_ + PER_SLOT - 1) / PER_SLOT, where_)

/-- Each row's chain (compression `k`'s `cv` from compression `k − 1`'s `out`) and its hm96 slot's `cv` (from the last). -/
def rowWires (ports : Array Port) (sha hm : Nat) : Array Wire := Id.run do
  let (_, where_) := rowSlots ports
  let mut out := #[]
  for p in [0:where_.size] do
    let w := where_[p]!
    -- a segmented row's chains restart at each chain start, whose midstate is a Δ constant
    for k in [1:w.size] do
      if (chainStart ports[p]! k).isNone then
        out := out.push ⟨(sha, w[k]!.1, 2 * w[k]!.2), (sha, w[k - 1]!.1, w[k - 1]!.2)⟩
    -- a row of no compression gives its hm96 slot the midstate, a Δ constant
    if 0 < w.size && (chainStart ports[p]! w.size).isNone then
      let last := w[w.size - 1]!
      out := out.push ⟨(hm, p, 0), (sha, last.1, last.2)⟩
  return out

/-- The zero row's `b ‖ c` under a zero salt: `b = x` (a zero salt's mask is zero), `c = SHA-512(salt_prefix ‖ 0^192)`;
`x` is the port's row digest of the zero row (`sha512/row/v1` of its words, or a bit row of its length). -/
def zeroRowBc (p : Port) : ByteArray :=
  (if p.segs.isSome then segDigest p.bits (ByteArray.mk (Array.replicate p.bytes 0))
   else Sha512.hash (prefixOf p ++ ByteArray.mk (Array.replicate p.bytes 0))) ++
    Sha512.hash (Hm96.Default512.saltPrefix ++ ByteArray.mk (Array.replicate SALT_BYTES 0))

private def ctx := "circuit META"

private def intLists (j : Json) (k : String) : V (Array (Array Int)) := do
  (← J.arr j k ctx).mapM fun u => do (← J.asArr u s!"{ctx}.{k}").mapM fun x => J.asInt x s!"{ctx}.{k}"

/-- META's `k_log`, refused past `MAX_K_LOG` where `parse` reads it: `check` computes `2 ^ k_log` and, once `slot_log ≤
k_log`, `2 ^ slot_log`, so neither is computed for a block no schedule covers. -/
def kLogOf (mj : Json) : V Nat := do
  let k ← J.nat mj "k_log" ctx
  if k > MAX_K_LOG then throw s!"META: k_log {k} is over {MAX_K_LOG}: no fast100 schedule"
  return k

/-- **`kLogOf` is META's `k_log` up to `MAX_K_LOG`:** it reads what `J.nat` reads and refuses only a larger one, which
`setupH` refuses too. -/
theorem kLogOf_ok {mj : Json} {k : Nat} : kLogOf mj = .ok k ↔ J.nat mj "k_log" ctx = .ok k ∧ k ≤ MAX_K_LOG := by
  unfold kLogOf
  cases h : J.nat mj "k_log" ctx with
  | error e => simp [bind, Except.bind]
  | ok k' =>
    simp only [bind, Except.bind]
    by_cases hk : k' > MAX_K_LOG
    · simp only [hk, ↓reduceIte, throw, throwThe, MonadExceptOf.throw, Except.error.injEq, reduceCtorEq, false_iff,
        not_and, Except.ok.injEq]
      omega
    · simp only [hk, ↓reduceIte, pure, Except.pure, Except.ok.injEq, true_and]
      constructor
      · rintro rfl; omega
      · rintro ⟨rfl, -⟩; rfl

/-- **`parse`'s lookup index test is `2 ^ in_bits = words` alone:** comparing `in_bits` with the table's `log2` first,
so that no power past the table's is computed, refuses nothing more. -/
theorem lookup_index_test (n size : Nat) : (n != Nat.log2 size || 2 ^ n != size) = (2 ^ n != size) := by
  by_cases h : 2 ^ n = size
  · subst h; simp [Nat.log2_two_pow]
  · simp [h]

/-- #273's `check_typed`: the instance's root one slot per VU, its ports the rows' words and the output words (16 bits each,
one output group); each placed layout's circuit one slot of its range per part; a forced-zero row in the root's slot; the
pinned dummy values and leaf scheme. -/
def checkTyped (c : Circuit) (t : TypedBlock) (tags : Tags) (leafScheme : Json) : V Unit := do
  -- the rows' circuits are found by name, and the placed layouts are named by digest: no part may be one of them
  if c.shaNet ≤ t.rangeLogs.size || c.hmNet ≤ t.rangeLogs.size then throw "the rows' circuits are among the template's"
  let root := c.unit
  let nOut := c.outWords
  if c.perVu.getD c.unitNet 0 != 1 then throw "the instance's root is not one slot per VU"
  if t.ins != c.ports.map (fun p => (p.words, 16)) || t.outs != c.outPorts.map (fun p => (p.2, 16)) then
    throw "the instance type's ports are not the rows' words and the output words"
  if root.outGroups.size != 1 || (root.outGroups.getD 0 default).bits.foldl (· + ·) 0 != 16 * nOut || !isPow2 (16 * nOut) then
    throw "the instance's outputs are not one output group of the output words"
  for (n, s) in t.rangeLogs do
    if c.perVu.getD n 0 != (t.parts.filter (·.1 == n)).size || (c.nets.getD n default).2.unitLog != s then
      throw s!"layout {(c.nets.getD n default).1.take 16}..: not one slot of its range per part"
  if root.useful ≥ 2 ^ root.unitLog then throw "the instance's root fills its slot: no forced-zero row for a zero bit"
  if c.dummyDigests.size != c.ports.size || c.dummyOutputs.size != nOut then
    throw "the pinned dummy values are not one per port and output leaf"
  if let some pinned := tags.pinnedLeafScheme then
    if (← canon leafScheme) != (← canon pinned) then throw "META: leaf_scheme is not the pinned leaf-commitment scheme"

/-- Upstream's `Composite::check`: the layout's shape, the row slots' ports and counts, every non-row input port wired
exactly once, the leaf maps, the pinned dummy values. -/
def check (c : Circuit) (tags : Tags) (leafScheme : Json) : V Unit := do
  let mut spans : Array (Nat × Nat) := #[]
  for r in c.ranges do
    if r.slotLog < 7 || r.slotLog > c.kLog then throw "a range's slots are smaller than a word or larger than the block"
    let (span, align) := if r.packed then (r.count * 2 ^ r.slotLog, 2 ^ r.slotLog)
      else (nextPow2 r.count * 2 ^ r.slotLog, nextPow2 r.count * 2 ^ r.slotLog)
    if r.packed && (r.kind == Kind.net c.hmNet || r.kind == Kind.net c.outNet || r.kind == Kind.mask) then
      throw "a packed range that a region reads"
    if r.start % align != 0 || r.start + span > 2 ^ c.kLog then throw "a range is not aligned to its span or overruns the block"
    match r.kind with
    | .net i => if (c.nets.getD i default).2.unitLog != r.slotLog then throw "a unit range's slot size is not its circuit's"
    | .mask => if r.count != 1 then throw "not one mask slot"
    | .comp => throw "a compression-leaf range in an hm96-sha512 row statement"
    spans := spans.push (r.start, r.start + span)
  -- every pair, so that two ranges' disjointness is a direct fact of the loop
  for i in [0:spans.size] do
    for j in [i + 1:spans.size] do
      if spans[i]!.1 < spans[j]!.2 && spans[j]!.1 < spans[i]!.2 then throw "two ranges overlap"
  if c.unit.lookup.isSome then throw "the unit circuit is a lookup slot"
  let (nSha, _) := rowSlots c.ports
  if c.perVu.getD c.shaNet 0 != nSha || c.perVu.getD c.hmNet 0 != nextPow2 c.ports.size then
    throw "the rows' slots per VU are not the ports' compressions (three per slot) and one hm96 slot per port"
  let sha := (c.nets.getD c.shaNet default).2
  let hm := (c.nets.getD c.hmNet default).2
  let si := portCols sha.inGroups
  if si.size != 2 * PER_SLOT || (List.range si.size).any (fun i => si[i]!.2 != (if i % 2 == 0 then 512 else 1024))
      || (portCols sha.outGroups).map (·.2) != #[512, 512, 512]
      || (portCols hm.inGroups).map (·.2) != #[512, 1024, 8 * SALT_BYTES]
      || (portCols hm.outGroups).map (·.2) != #[8 * BC_BYTES] then
    throw "the sha512x3 and hm96 circuits' ports are not (cv, msg) x 3 -> out x 3 and (cv, pad, y) -> bc"
  if let some t := c.typed then
    for r in c.ranges do
      if let .net n := r.kind then
        if r.count != c.g * c.perVu.getD n 0 then throw s!"a range of {(c.nets.getD n default).1} is not {c.perVu.getD n 0} slots per VU"
    for n in [0:c.nets.size] do
      if (c.ranges.filter (·.kind == Kind.net n)).size != 1 then throw s!"circuit {(c.nets.getD n default).1} has not exactly one range"
    return ← checkTyped c t tags leafScheme
  let unit := c.unit
  let ur := c.ranges.filter (·.kind == Kind.net c.unitNet)
  if ur.size != 1 || ur[0]!.count != c.g * c.upv || (c.outNet == c.unitNet && !isPow2 ur[0]!.count) then
    throw "not one unit range of vus_per_block x units_per_vu (a power of two) units"
  -- a tail that returns the outputs (PR #83 from 15260468): its last stage's only output group is the output words
  let tailOut := c.outNet != c.unitNet
  let g0in := unit.inGroups.getD 0 default
  let g0out := unit.outGroups.getD 0 default
  if unit.inGroups.isEmpty || unit.inGroups.size > 2 || unit.outGroups.isEmpty || unit.outGroups.size > 2
      || g0in.bits.any (· != 16) || (!tailOut && g0out.bits.any (· != 16)) then
    throw "the unit's ports are not a group of 16-bit leaf inputs (then cut inputs) and one of 16-bit returned outputs"
  if tailOut then
    let nOut := c.outWords
    let onet := c.nets.getD c.outNet default
    let oc := portCols onet.2.outGroups
    if !onet.1.startsWith "stage" || c.outNetPorts != Array.range oc.size || oc.size != nOut || oc.any (·.2 != 16)
        || onet.2.outGroups.size != 1 || !isPow2 (16 * nOut) || c.leavesOut.any (!·.isEmpty) then
      throw "the returned outputs are not the last stage's only output group: one 16-bit port per output word"
  for r in c.ranges do
    if let .net n := r.kind then
      if r.count != c.g * c.perVu.getD n 0 then throw s!"a range of {(c.nets.getD n default).1} is not {c.perVu.getD n 0} slots per VU"
  for n in [0:c.nets.size] do
    if (c.ranges.filter (·.kind == Kind.net n)).size != 1 then throw s!"circuit {(c.nets.getD n default).1} has not exactly one range"
  let mut wired : Array ((Nat × Nat × Nat) × Nat) := #[]
  for w in c.wires do
    let (dn, di, dp) := w.dst
    let (sn, si, sp) := w.src
    let dc := portCols (c.nets.getD dn default).2.inGroups
    let sc := portCols (c.nets.getD sn default).2.outGroups
    if dn ≥ c.nets.size || sn ≥ c.nets.size || di ≥ c.perVu.getD dn 0 || si ≥ c.perVu.getD sn 0 || dp ≥ dc.size
        || sp ≥ sc.size || dc[dp]!.2 != sc[sp]!.2 then
      throw "a wire outside its slots' ports or of unequal widths"
    if dn == c.unitNet && dp < g0in.bits.size then throw "a wire into a unit's leaf port"
    wired := match wired.findIdx? (·.1 == w.dst) with
      | some i => wired.modify i fun (k, v) => (k, v + 1)
      | none => wired.push (w.dst, 1)
  -- a leaf cut wires a unit's cut input from a row word
  let nInW := c.inWords
  let pcu := portCols unit.inGroups
  for (u, j, leaf) in c.leafCuts do
    if u ≥ c.upv || j < g0in.bits.size || j ≥ pcu.size || pcu[j]!.2 < 16 || leaf ≥ nInW then
      throw "a leaf cut outside the unit's cut inputs or the rows"
    let key := (c.unitNet, u, j)
    wired := match wired.findIdx? (·.1 == key) with
      | some i => wired.modify i fun (k, v) => (k, v + 1)
      | none => wired.push (key, 1)
  for n in [0:c.nets.size] do
    if n == c.shaNet || n == c.hmNet then continue
    let net := (c.nets.getD n default).2
    let first := if n == c.unitNet then g0in.bits.size else 0
    for i in [0:c.perVu.getD n 0] do
      for p in [first:(portCols net.inGroups).size] do
        if ((wired.find? (·.1 == (n, i, p))).map (·.2)).getD 0 != 1 then
          throw s!"input port {p} of {(c.nets.getD n default).1} slot {i} is not wired exactly once"
  let nIn := c.inWords
  let nOut := c.outWords
  if c.leavesIn.size != c.upv || c.leavesOut.size != c.upv then throw "the leaf maps are not one per unit"
  -- a tail that returns the outputs returns every one of them (the units return none)
  let mut seen : Array Bool := Array.replicate nOut tailOut
  for u in [0:c.upv] do
    let li := c.leavesIn[u]!
    let lo := c.leavesOut[u]!
    if li.size != g0in.bits.size || li.any (fun x => x < -1 || x ≥ (nIn : Int)) || (!tailOut && lo.size != g0out.bits.size) then
      throw s!"unit {u}'s leaf maps are not its circuit's ports"
    for o in lo do
      if o ≥ nOut || seen[o]! then throw s!"output leaf {o} is outside the outputs or returned twice"
      seen := seen.set! o true
  if seen.any (!·) then throw "an output leaf no unit returns"
  if unit.useful ≥ 2 ^ unit.unitLog then throw "the unit fills its slot: no forced-zero row for a zero leaf"
  -- a tail that returns the outputs has no power-of-two count, which alone refuses zero units otherwise
  if c.upv == 0 then throw "units_per_vu is 0: a VU of no unit"
  if c.g == 0 then throw "vus_per_block is 0: a block of no VU"
  -- the Digest region's VU bits are `Nat.log2 g` here and `trailing_zeros` in the Rust verifier, which pin every VU,
  -- and agree, only at a power of two
  if !isPow2 c.g then throw "vus_per_block is not a power of two: the Digest region would pin some VUs and not others"
  if c.dummyDigests.size != c.ports.size || c.dummyOutputs.size != nOut then
    throw "the pinned dummy values are not one per port and output leaf"
  if let some pinned := tags.pinnedLeafScheme then
    if (← canon leafScheme) != (← canon pinned) then throw "META: leaf_scheme is not the pinned leaf-commitment scheme"

/-- Part `i`'s slot among its layout's circuit's slots in the VU: how many parts before it place the same layout. -/
def slotOf (ps : Array (Nat × Nat)) (i : Nat) : Nat :=
  ((ps.toList.take i).filter (·.1 == (ps.getD i default).1)).length

/-- A template's block with its parts placed (#273's `TypedBlock`), by position: `parse` puts the root's circuit first and
then the placed layouts' in order, so placed layout `k`'s circuit is `1 + k`. Part `i` is its layout's circuit, its slot
(`slotOf`, in item order) and its base. -/
def blockOf (t : Typed.Template) : TypedBlock :=
  { own := t.own, ins := t.ins, outs := t.outs, inCols := t.inCols,
    parts := (Array.range t.parts.size).map fun i => (1 + (t.parts.getD i default).1, slotOf t.parts i, (t.parts.getD i default).2),
    rangeLogs := (Array.range t.layouts.size).map fun k => (1 + k, (t.layouts.getD k default).2.2),
    bind := t.bind, cross := t.cross }

/-- **A template's block, as `blockOf` builds it** (for audit-lean's T3): the root's size, ports, input columns, bound rows
and cross entries are the template's; part `i` is circuit `1 + k` of its layout's position `k`, its slot `slotOf`, and its
base; placed layout `k`'s circuit is `1 + k`, with its range's log size. -/
theorem blockOf_spec (t : Typed.Template) :
    let tb := blockOf t
    tb.own = t.own ∧ tb.ins = t.ins ∧ tb.outs = t.outs ∧ tb.inCols = t.inCols ∧ tb.bind = t.bind ∧ tb.cross = t.cross ∧
      tb.parts.size = t.parts.size ∧
      (∀ i (hi : i < t.parts.size) (hb : i < tb.parts.size), tb.parts[i] = (1 + t.parts[i].1, slotOf t.parts i, t.parts[i].2)) ∧
      tb.rangeLogs.size = t.layouts.size ∧
      ∀ k (hk : k < t.layouts.size) (hb : k < tb.rangeLogs.size), tb.rangeLogs[k] = (1 + k, t.layouts[k].2.2) := by
  refine ⟨rfl, rfl, rfl, rfl, rfl, rfl, by simp [blockOf], fun i hi hb => ?_, by simp [blockOf], fun k hk hb => ?_⟩
  · simp [blockOf, Array.getD, hi]
  · simp [blockOf, Array.getD, hk]

/-- Part `i`'s slot is below the number of parts on its layout: its own place counts. -/
theorem slotOf_lt {ps : Array (Nat × Nat)} {i : Nat} (hi : i < ps.size) :
    slotOf ps i < (ps.toList.filter (·.1 == ps[i].1)).length := by
  have hl : i < ps.toList.length := by simpa using hi
  have hsplit : ps.toList = ps.toList.take i ++ ps[i] :: ps.toList.drop (i + 1) := by
    conv => lhs; rw [← List.take_append_drop i ps.toList]
    rw [List.drop_eq_getElem_cons hl, Array.getElem_toList]
  have hg : ps.getD i default = ps[i] := by simp [Array.getD, hi]
  unfold slotOf
  rw [hg]
  conv => rhs; rw [hsplit]
  simp only [List.filter_append, List.filter_cons, beq_self_eq_true, ↓reduceIte, List.length_append, List.length_cons]
  omega

/-- Two parts on one layout have distinct slots: a later part's counts the earlier one. -/
theorem slotOf_lt_of_lt {ps : Array (Nat × Nat)} {i j : Nat} (hij : i < j) (hj : j < ps.size) (he : ps[i].1 = ps[j].1) :
    slotOf ps i < slotOf ps j := by
  have hi : i < ps.size := Nat.lt_trans hij hj
  have hgi : ps.getD i default = ps[i] := by simp [Array.getD, hi]
  have hgj : ps.getD j default = ps[j] := by simp [Array.getD, hj]
  unfold slotOf
  rw [hgi, hgj, he]
  obtain ⟨d, hd⟩ : ∃ d, j = i + 1 + d := ⟨j - (i + 1), by omega⟩
  have hti : ps.toList.take (i + 1) = ps.toList.take i ++ [ps[i]] := by
    rw [List.take_add_one, List.getElem?_eq_getElem (by simpa using hi), Array.getElem_toList]; rfl
  have htj : ps.toList.take j = ps.toList.take i ++ ps[i] :: (ps.toList.drop (i + 1)).take d := by
    rw [hd, List.take_add, hti, List.append_assoc]; rfl
  rw [htj]
  simp only [List.filter_append, List.filter_cons, he, beq_self_eq_true, ↓reduceIte, List.length_append, List.length_cons]
  omega

/-- A typed template's META keys that its block derives instead (#273): stating one is refused. -/
def TEMPLATE_DERIVED : List String :=
  ["unit_log", "units_per_vu", "leaves_in", "leaves_out", "wires", "leaf_cuts", "out_net", "out_net_ports", "lookups"]

/-- META's row ports: one `sha512/row/v1` schema for every port, each row whole SHA-512 blocks, or (`row_schemas`) each
port's bit row of a u64 length in bits, its leaves and compressions derived from the length (a typed statement's rows are
`sha512/row/v1`'s); every row's constants recomputed from its schema and length, never taken from META. -/
def rowPorts (mj : Json) (typed : Bool) : V (Array Port) := do
  let v2 := J.has mj "row_schemas"
  let one := (mj.getObjValAs? String "row_schema").toOption
  if (← J.str mj "scheme" ctx) != SCHEME || v2 == J.has mj "row_schema" || (!v2 && one != some ROW_SCHEMA) then
    throw s!"META: rows are not {SCHEME} ({ROW_SCHEMA}, or {ROW_SCHEMA_V2} per port) leaves"
  if v2 && typed then throw s!"META: a typed statement's rows are {ROW_SCHEMA} leaves"
  let listedPorts ← J.arr mj "ports" ctx
  let lengths : Array (Option (Nat × Bool)) ← if v2 then do
      let s ← J.arr mj "row_schemas" ctx
      if s.size != listedPorts.size then throw "META: row_schemas is not one [schema, bits] per port"
      s.mapM fun x => do
        let a ← J.asArr x "row_schemas"
        let schema := (a[0]!.getStr?).toOption
        if a.size != 2 || (schema != some ROW_SCHEMA_V2 && schema != some ROW_SCHEMA_SEG) then
          throw s!"META: row_schemas: each port's is [{ROW_SCHEMA_V2} or {ROW_SCHEMA_SEG}, bits]"
        let l ← J.asNat a[1]! "a bit row's length"
        if l ≥ 2 ^ 64 then throw "META: a bit row's length is not a u64"
        pure (some (l, schema == some ROW_SCHEMA_SEG))
    else pure (Array.replicate listedPorts.size none)
  let ports ← (Array.range listedPorts.size).mapM fun i => do
    let a ← J.asArr listedPorts[i]! s!"{ctx}: a port"
    if a.size != 3 then throw s!"{ctx}: a port is not [name, words, comps]"
    let name ← J.asStr a[0]! "port name"
    let words ← J.asNat a[1]! "port words"
    let comps ← J.asNat a[2]! "port comps"
    match lengths[i]! with
    | none =>
      if words == 0 || (2 * words) % 128 != 0 || comps != 2 * words / 128 then throw s!"port {name}: not whole SHA-512 blocks"
      pure ({ name, words, chunks := comps, nb := 0 } : Port)
    | some (l, seg) =>
      -- every count derived from the length: its leaves, and its messages' blocks after their prefixes but the last
      let serial := ((l + 7) / 8 + 17 + 127) / 128 - 1
      -- a segmented row's are at most a serial row's and 18 more per 2^13 bits: bounded before it is shaped
      let segs := if seg && serial ≤ MAX_COMPS then some (segShape l) else none
      let want := ((l + 15) / 16, match segs with
        | some (s, o) => s.foldl (· + ·.2) 0 + o - 1
        | none => if seg then MAX_COMPS + 1 else serial)
      if want.2 > MAX_COMPS || (words, comps) != want then
        throw s!"port {name}: a row of {l} bits is {want.1} u16 leaves and {want.2} compressions, not META's {words} and {comps}"
      pure ({ name, words, chunks := comps, nb := 0, bits := l, v2 := true, segs := segs.map (·.1) } : Port)
  let listed ← J.arr mj "rows" ctx
  let bad := s!"META: a row's constants are not {if v2 then "sha512/row/v2 or sha512/row-seg/v1" else "sha512/row/v1"}'s"
  if listed.size != ports.size then throw bad
  for i in [0:ports.size] do
    let p := ports[i]!
    let m := listed[i]!
    let ms ← (← J.arr m "midstate" ctx).mapM (J.asStr · "midstate")
    if (← J.str m "prefix" ctx) != hexOf (prefixOf p) || ms != (midstateOf p).map hex16 || (← J.str m "pad" ctx) != hexOf (tailOf p) then
      throw bad
  if v2 && ports.all (·.chunks == 0) then
    throw s!"META: no row has a compression before its last block (the {SHA_NET} range, the pin's, holds no slot)"
  return ports

/-- Parse the circuit file (upstream's `Composite::parse`), check its layout, and refuse `sha512x3` and `hm96` nets other
than the verifier's (`HmNets.check`), and a `rec-open/…`, `rec-residuals/…`, `rec-sparse/…` or `rec-opening/…` unit
other than the verifier's (`RecOpen.check`, `RecResiduals.check`, `RecSparse.check`, `RecOpening.check`). A typed
template's block (`tmpl`, item 1e) brings its root and placed layouts' circuits; its wiring is derived. -/
def parse (tags : Tags) (bytes : ByteArray) (tables : List (String × ByteArray) := [])
    (tmpl : Option Typed.Template := none) (pre : Array (String × Net) := #[]) : V Circuit := do
  let sha := Sha512.hash bytes
  let text ← match String.fromUTF8? bytes with
    | some s => pure s
    | none => throw "circuit: not UTF-8"
  let lines0 := (text.splitOn "\n").toArray
  let lines := if lines0.back? == some "" then lines0.pop else lines0
  let head := (lines.getD 0 "").trimAsciiEnd.toString
  if !head.startsWith (tags.fileFormat ++ " ") then throw s!"circuit: not a {tags.fileFormat} file"
  let name := (head.drop (tags.fileFormat.length + 1)).toString
  let metaLine := (lines.getD 1 "").trimAsciiEnd.toString
  if !metaLine.startsWith "META " then throw "circuit: no META line"
  let mj ← match Json.parse (metaLine.drop 5).toString with
    | .ok j => pure j
    | .error e => throw s!"META: {e}"
  if (← J.str mj "format" ctx) != tags.fileFormat || (← J.str mj "statement" ctx) != tags.statement then
    throw s!"META: not {tags.statement}"
  let classSha ← if tags.bindings then do
      if !isSha512Hex ((mj.getObjValAs? String "program_sha512").toOption.getD "") then
        throw "META: program_sha512 (the Program the partition is over) is not a SHA-512 digest"
      let part := (mj.getObjVal? "partition").toOption.getD Json.null
      let rule := (part.getObjValAs? String "rule").toOption.getD ""
      if !(rule == PARTITION_V1 || rule == PARTITION_PROVISIONAL) || !isSha512Hex ((part.getObjValAs? String "digest").toOption.getD "") then
        throw s!"META: the partition is not a {PARTITION_V1} (or the provisional {PARTITION_PROVISIONAL}) SHA-512 digest"
      classOf bytes head mj
    else pure ByteArray.empty
  let mut nets : Array (String × Net) := match tmpl with
    | some t => #[("root", t.root)] ++ t.layouts.map fun (d, n, _) => (d, n)
    | none => pre
  let mut at_ := 2
  while at_ < lines.size do
    let h := tokens lines[at_]!
    if h.size != 3 || h[0]! != tags.blockKeyword then throw s!"circuit: expected a {tags.blockKeyword} line"
    let some n := h[2]!.toNat? | throw s!"circuit: {tags.blockKeyword} line count"
    if at_ + 1 + n > lines.size then throw s!"circuit: {h[1]!} is truncated"
    let net ← match Net.parse (lines.extract (at_ + 1) (at_ + 1 + n)) with
      | .ok v => pure v
      | .error e => throw s!"{h[1]!}: {e}"
    nets := nets.push (h[1]!, net)
    at_ := at_ + 1 + n
  for k in Circuit.FORBIDDEN do
    if J.has mj k then throw s!"META: {k} (a verifier-evaluated part of the computation); the verifier only checks proofs"
  if tmpl.isSome then
    for k in TEMPLATE_DERIVED do
      if J.has mj k then throw s!"META: {k}: a typed template's block wiring is derived from its types, never stated"
  let lks ← match mj.getObjVal? "lookups" with
    | .ok (Json.obj kvs) => pure (kvs.toArray.qsort (fun a b => a.1 < b.1))
    | .ok Json.null => pure #[]
    | .ok _ => throw "META: lookups is not an object"
    | .error _ => pure #[]
  for (lname, spec) in lks do
    let some pinned := MUFU_TABLES.find? (·.name == lname) | throw "META: a lookup of a table the statement does not pin"
    let size := pinned.words
    if (← J.str spec "sha256" ctx) != pinned.sha256 || (← J.str spec "table" ctx) != lname || (← J.nat spec "out_bits" ctx) != 31 then
      throw s!"META: lookup {lname} is not the pinned table"
    let n ← J.nat spec "in_bits" ctx
    let lo ← J.nat spec "lo_bits" ctx
    if n != Nat.log2 size || 2 ^ n != size || !(2 ≤ lo && lo < n) then throw s!"META: lookup {lname}'s index is not its table's"
    let some (_, words) := tables.find? (·.1 == lname) | throw s!"the circuit has a lookup of {lname}: its table is not given"
    if words.size != 4 * size || hexOf (Sha512.hash words) != pinned.sha512 then throw s!"MUFU table {lname}: not the pinned table"
    nets := nets.push (s!"lookup:{lname}", ← Lookup.build s!"lookup:{lname}" words n lo 31)
  let ports ← rowPorts mj tags.typed
  if (← J.str mj "row_key_sha512" ctx) != hexOf (Sha512.hash Hm96.Default512.key) then
    throw "META: the rows' hm96 key is not hm96-sha512/v1's default key"
  let g ← J.nat mj "vus_per_block" ctx
  -- a typed template's unit is its instance's root, one per VU
  let upv ← if tmpl.isSome then pure 1 else J.nat mj "units_per_vu" ctx
  let some unitNet := nets.findIdx? (·.1 == if tmpl.isSome then "root" else "unit") | throw "circuit: no unit circuit"
  let netOf := fun (nm : String) => match nets.findIdx? (·.1 == nm) with
    | some i => (pure i : V Nat)
    | none => throw s!"META: unknown circuit {nm}"
  let ranges ← (← J.arr mj "ranges" ctx).mapM fun r => do
    let kind ← match ← J.str r "type" ctx with
      | "mask" => pure Kind.mask
      | "unit" => pure (Kind.net (← netOf (← J.str r "net" ctx)))
      | _ => throw "a range of an unknown type"
    let packed := match r.getObjVal? "packed" with
      | .ok (Json.bool b) => b
      | _ => false
    pure ({ kind, slotLog := ← J.nat r "slot_log" ctx, pos0 := ← J.nat r "pos0" ctx, count := ← J.nat r "count" ctx,
            role := .none, packed } : Range)
  let leavesIn ← if tmpl.isSome then pure #[#[]] else intLists mj "leaves_in"
  let leavesOut ← if tmpl.isSome then pure #[#[]] else (← intLists mj "leaves_out").mapM fun u => u.mapM fun x =>
    if x < 0 then (throw "META: a negative output leaf" : V Nat) else pure x.toNat
  let dummy ← J.get mj "dummy" ctx
  let dummyDigests ← (← J.arr dummy "bc" ctx).mapM fun d => do
    let b ← unhex (← J.asStr d "dummy bc")
    if b.size != BC_BYTES then throw "b || c length"
    pure b
  let dummyOutputs ← (← J.arr dummy "outputs" ctx).mapM fun x => do
    let v ← J.asNat x "dummy output"
    if v ≥ 65536 then throw "META: a dummy output is not a u16"
    pure v
  let outPorts ← (← J.arr mj "out_ports" ctx).mapM fun p => do
    let a ← J.asArr p "out port"
    if a.size != 2 then throw "META: an out port is not [name, words]"
    pure (← J.asStr a[0]! "out port name", ← J.asNat a[1]! "out port words")
  let outRowPorts ← match mj.getObjVal? "out_row_ports" with
    | .error _ => pure (Array.replicate outPorts.size 1)
    | .ok v => do
      let bad := "META: out_row_ports is not each output row's run of output ports, in order, covering every output port once"
      let r ← (← J.asArr v "out_row_ports").mapM fun x => do
        let k ← J.asNat x "out_row_ports"
        if k < 1 || k > outPorts.size then throw bad
        pure k
      if r.foldl (· + ·) 0 != outPorts.size then throw bad
      pure r
  let mut perVu := (Array.replicate nets.size 0).set! unitNet upv
  match mj.getObjVal? "per_vu" with
  | .ok (Json.obj kvs) =>
    for (k, v) in kvs.toArray do
      perVu := perVu.set! (← netOf k) (← J.asNat v "per_vu")
  | _ => pure ()
  let shaNet ← netOf SHA_NET
  let hmNet ← netOf HM_NET
  let outNet ← if tmpl.isSome then pure unitNet else netOf ((mj.getObjValAs? String "out_net").toOption.getD "unit")
  let outNetPorts ← match mj.getObjVal? "out_net_ports" with
    | .ok (.arr a) => a.mapM (J.asNat · "out_net_ports")
    | _ => pure #[]
  let leafCuts ← match mj.getObjVal? "leaf_cuts" with
    | .ok (.arr a) => a.mapM fun w => do
      let x ← J.asArr w "a leaf cut"
      if x.size != 4 then throw "META: a leaf cut is not [circuit, unit, port, leaf]"
      if (x[0]!.getStr?).toOption != some "unit" then throw "META: a leaf cut into anything but a unit"
      pure (← J.asNat x[1]! "a leaf cut", ← J.asNat x[2]! "a leaf cut", ← J.asNat x[3]! "a leaf cut")
    | _ => pure #[]
  let wires ← match mj.getObjVal? "wires" with
    | .ok w => (← J.asArr w "wires").mapM fun x => do
      let a ← J.asArr x "wire"
      if a.size != 6 then throw "META: a wire is not six fields"
      pure (⟨(← netOf (← J.asStr a[0]! "wire"), ← J.asNat a[1]! "wire", ← J.asNat a[2]! "wire"),
              (← netOf (← J.asStr a[3]! "wire"), ← J.asNat a[4]! "wire", ← J.asNat a[5]! "wire")⟩ : Wire)
    | .error _ => pure #[]
  -- the rows' wires come first, exactly as the ports imply; a typed template's are just those
  let derived := rowWires ports shaNet hmNet
  let wires := if tmpl.isSome then derived else wires
  let rowNet := fun (n : Nat) => n == shaNet || n == hmNet
  if wires.size < derived.size || (List.range derived.size).any (fun i => wires[i]!.dst != derived[i]!.dst || wires[i]!.src != derived[i]!.src)
      || (wires.extract derived.size wires.size).any (fun w => rowNet w.dst.1 || rowNet w.src.1) then
    throw "META: the rows' wires are not the chains and hm96 inputs the ports imply"
  let leafScheme ← J.get mj "leaf_scheme" ctx
  let leafId ← J.str leafScheme "id" ctx
  if leafId != tags.merkleLeaf then throw s!"leaf-commitment scheme {leafId} is not one this verifier implements"
  let c : Circuit := { name, sha, nets, ranges, ports, outPorts, outRowPorts, g, kLog := ← kLogOf mj, upv, unitNet,
                       leavesIn, leavesOut, key := ByteArray.empty, dummyDigests, dummyOutputs, leafScheme := leafId, perVu, wires,
                       shaNet, hmNet, classSha, metaJ := mj, outNet, outNetPorts, leafCuts,
                       typed := ← tmpl.mapM fun t => pure (blockOf t) }
  check c tags leafScheme
  HmNets.check c
  RecOpen.check c
  RecResiduals.check c
  RecSparse.check c
  RecOpening.check c
  return c

/-- A typed statement (item 1e), its rows derived once through `deriveChecked` (`Typed.read`), every net built from them.
A flat class is the typed file parsed with its unit's net first (`parse`'s `pre`), where the old statement's `CIRCUIT
unit` section was. A template is parsed with its block (`parse`'s `tmpl`): the root and placed layouts' circuits, the
derived wiring, and Δ in #273's order. The circuit carries its class (`Circuit.cls`). -/
def parseTyped (tags : Tags) (bytes : ByteArray) (tables : List (String × ByteArray) := []) : V Circuit := do
  let (k, r) ← Typed.read tags bytes tables
  let c ← match r with
    | .inr t => parse tags bytes tables (tmpl := some t)
    | .inl unit => parse tags bytes tables (pre := #[("unit", unit)])
  return { c with cls := some k }

/-- The pinned constant column: the first slot of the first range that is not the mask, a column of the block. -/
def pin (c : Circuit) : V Nat := do
  let p ← match c.ranges.find? (·.kind != Kind.mask) with
    | some r => match r.kind with
      | .net i =>
        if r.count == 0 then throw "circuit: the pin's range has no slot"
        else pure (r.start + (c.nets.getD i default).2.constPos)
      | _ => throw "no pin"
    | none => throw "circuit: no range with a constant"
  if p ≥ 2 ^ c.kLog then throw "the pinned constant column is outside the block"
  return p

/-- Δ (A and B, upstream's `delta`): every slot's constant from the pin; the rows' constants (each row's first `cv` the
prefix's midstate, in its first compression or, for a row of none, its hm96 slot; every message bit at or past the row's
length, in its hm96 slot's `pad` and in the compressions before it: bit 1 from the pin, bit 0 from the unit range's first
slot's forced-zero row); every unit's leaf bits copied from the message bits that hash them (a zero leaf, or a leaf bit
past the row's length, from its own forced-zero row); every wire, the rows' chains first. -/
def delta (c : Circuit) (pin : Nat) : Array (Nat × Nat) × Array (Nat × Nat) := Id.run do
  let mut d : Array (Nat × Nat) := #[]
  for r in c.ranges do
    if let .net i := r.kind then
      for s in [0:r.count] do
        let k := r.slot s + (c.nets.getD i default).2.constPos
        if k != pin then d := d.push (k, k) |>.push (k, pin)
  let (_, where_) := rowSlots c.ports
  let sin := portCols (c.nets.getD c.shaNet default).2.inGroups
  let hin := portCols (c.nets.getD c.hmNet default).2.inGroups
  let unit := c.unit
  let ur := c.unitRange
  let zero := ur.slot 0 + unit.useful
  let sout := portCols (c.nets.getD c.shaNet default).2.outGroups
  for g in [0:c.g] do
    for p in [0:where_.size] do
      let w := where_[p]!
      let port := c.ports[p]!
      -- a segmented row: each chain start's midstate (a segment's, the outer message's), every constant message bit, and the
      -- outer message's digest bits copies of the segments' last outputs
      if let some segs := port.segs then
        for k in [0:w.size + 1] do
          let (cv, msg) := if k < w.size then
              (c.slotCol c.shaNet g w[k]!.1 + sin[2 * w[k]!.2]!.1, c.slotCol c.shaNet g w[k]!.1 + sin[2 * w[k]!.2 + 1]!.1)
            else (c.slotCol c.hmNet g p + hin[0]!.1, c.slotCol c.hmNet g p + hin[1]!.1)
          if let some j := chainStart port k then
            let ms := if j < segs.size then segMidstate port j else midstateOf port
            for t in [0:512] do
              let src := if (ms[t / 64]! >>> (t % 64).toUInt64) &&& 1 == 1 then pin else zero
              d := d.push (cv + t, cv + t) |>.push (cv + t, src)
          for t in [0:1024] do
            match msgBit port k t with
            | .const v => d := d.push (msg + t, msg + t) |>.push (msg + t, if v then pin else zero)
            | .digest kk col =>
              let src := c.slotCol c.shaNet g w[kk]!.1 + sout[w[kk]!.2]!.1 + col
              d := d.push (msg + t, msg + t) |>.push (msg + t, src)
            | .row _ => pure ()
        continue
      let ms := midstateOf port
      let tl := tailOf port
      let cv := if 0 < w.size then c.slotCol c.shaNet g w[0]!.1 + sin[2 * w[0]!.2]!.1 else c.slotCol c.hmNet g p + hin[0]!.1
      for t in [0:512] do
        let src := if (ms[t / 64]! >>> (t % 64).toUInt64) &&& 1 == 1 then pin else zero
        d := d.push (cv + t, cv + t) |>.push (cv + t, src)
      -- the last block's constant bits (a sha512/row/v1 row's: all of it), then those of the blocks before it from the
      -- first that holds a bit at or past the row's length
      let pad := c.slotCol c.hmNet g p + hin[1]!.1
      for t in [0:1024] do
        if let some v := constBit port tl w.size t then
          d := d.push (pad + t, pad + t) |>.push (pad + t, if v then pin else zero)
      for k in [port.bits / 1024:w.size] do
        let msg := c.slotCol c.shaNet g w[k]!.1 + sin[2 * w[k]!.2 + 1]!.1
        for t in [0:1024] do
          if let some v := constBit port tl k t then
            d := d.push (msg + t, msg + t) |>.push (msg + t, if v then pin else zero)
  let msgCol := fun (g p byte t : Nat) =>
    let (k, o) := byteAt c.ports[p]! byte
    if k < where_[p]!.size then
      let (slot, sub) := where_[p]![k]!
      c.slotCol c.shaNet g slot + sin[2 * sub + 1]!.1 + 64 * (o / 8) + 56 - 8 * (o % 8) + t
    else c.slotCol c.hmNet g p + hin[1]!.1 + 64 * (o / 8) + 56 - 8 * (o % 8) + t
  let leafCols := (portCols #[unit.inGroups.getD 0 default]).map (·.1)
  let starts := (c.ports.foldl (fun (acc : Array Nat × Nat) p => (acc.1.push acc.2, acc.2 + p.words)) (#[], 0)).1
  for k in [0:ur.count] do
    let (g, u) := (k / c.upv, k % c.upv)
    let base := ur.slot k
    let li := c.leavesIn.getD u #[]
    for j in [0:li.size] do
      let lf := li[j]!
      for t in [0:16] do
        -- a zero leaf, or a bit row's leaf bit past its length: the slot's forced-zero row
        let src := if lf < 0 then base + unit.useful else
          let x := lf.toNat
          let p := ((List.range starts.size).reverse.find? (fun q => starts[q]! ≤ x)).getD 0
          if 16 * (x - starts[p]!) + t < c.ports[p]!.bits then msgCol g p (2 * (x - starts[p]!) + t / 8) (t % 8)
          else base + unit.useful
        let i := base + leafCols.getD j 0 + t
        d := d.push (i, i) |>.push (i, src)
  let pc := portCols unit.inGroups
  for g in [0:c.g] do
    for (u, j, leaf) in c.leafCuts do
      let p := ((List.range starts.size).reverse.find? (fun q => starts[q]! ≤ leaf)).getD 0
      let dst := c.slotCol c.unitNet g u + pc[j]!.1
      for t in [0:pc[j]!.2] do
        let src := if t < 16 && 16 * (leaf - starts[p]!) + t < c.ports[p]!.bits then
          msgCol g p (2 * (leaf - starts[p]!) + t / 8) (t % 8) else zero
        d := d.push (dst + t, dst + t) |>.push (dst + t, src)
  for g in [0:c.g] do
    for w in c.wires do
      let dc := (portCols (c.nets.getD w.dst.1 default).2.inGroups).getD w.dst.2.2 default
      let sc := (portCols (c.nets.getD w.src.1 default).2.outGroups).getD w.src.2.2 default
      let dst := c.slotCol w.dst.1 g w.dst.2.1 + dc.1
      let src := c.slotCol w.src.1 g w.src.2.1 + sc.1
      for t in [0:dc.2] do d := d.push (dst + t, dst + t) |>.push (dst + t, src + t)
  -- a typed template (#273), per VU: each input of the instance a copy of its message bit (input w: bit w % 16 of row word
  -- w / 16); each part's bound input rows their forms over the unit's columns, in A and B; the root's rows' entries at
  -- its parts' columns, A's in A and B's in B
  let some tb := c.typed | return (d, d)
  let col := fun (g u : Nat) =>
    if u < tb.own then c.slotCol c.unitNet g 0 + u
    else match tb.parts.toList.reverse.find? (·.2.2 ≤ u) with
      | some (n, q, b) => c.slotCol n g q + (u - b)
      | none => 0
  let mut da := d
  let mut db := d
  for g in [0:c.g] do
    for w in [0:tb.inCols.size] do
      let (leaf, bit) := (w / 16, w % 16)
      let p := ((List.range starts.size).reverse.find? (fun q => starts[q]! ≤ leaf)).getD 0
      let i := col g tb.inCols[w]!
      let src := msgCol g p (2 * (leaf - starts[p]!) + bit / 8) (bit % 8)
      da := da.push (i, i) |>.push (i, src)
      db := db.push (i, i) |>.push (i, src)
    for (r, src) in tb.bind do
      let i := col g r
      da := da.push (i, i) ++ src.map (fun x => (i, col g x))
      db := db.push (i, i) ++ src.map (fun x => (i, col g x))
    for (r, xa, xb) in tb.cross do
      let i := col g r
      da := da ++ xa.map (fun x => (i, col g x))
      db := db ++ xb.map (fun x => (i, col g x))
  return (da, db)

/-- The header's `units`: the instances' global unit indices (ascending), and per block its units. -/
def unitsEntry (c : Circuit) (indices : Array Nat) : Json :=
  let base := [("vus_per_block", Lean.toJson c.g), ("units_per_vu", Lean.toJson c.upv),
    ("indices", Json.arr (indices.map fun x => Lean.toJson x)),
    ("blocks", Json.arr ((Array.range ((indices.size + c.g - 1) / c.g)).map fun b =>
      Json.arr ((indices.extract (b * c.g) (b * c.g + c.g)).map fun x => Lean.toJson x)))]
  -- from `e51e2b86`: every instance's class, the verifier's own computation of it
  Json.mkObj (if c.classSha.isEmpty then base
    else base ++ [("classes", Json.arr (Array.replicate indices.size (Json.str (hexOf c.classSha))))])

/-- The queries the verifier can evaluate itself (`Flock.Qword`), given the program's Call graphs. -/
def EVALUABLE : List (String × Nat) := [("Q_word", 1), ("Q_word", 2)]

/-- `Q_call` v1, which the verifier evaluates itself (`Flock.Qcall`). -/
def Q_CALL : String × Nat := ("Q_call", 1)

/-- A `verity/partition/v1` object's form (#111's canonical form, open query names): exactly `format`, `program`, `query`; a
SHA-512 program; a query of exactly `name` (a non-empty string), `version` (a natural number) and `params` (an object). For a
query the verifier evaluates, its parameters too (`Q_word` v1 and v2: exactly `X`, `W`, positive integers; `Q_call` v1: exactly
`X`, an integer in 1..2^16). The query's name and version. -/
def checkPartitionObject (obj : Json) : V (String × Nat) := do
  let keys := fun (j : Json) => match j with
    | .obj kvs => (kvs.toArray.map (·.1)).qsort (· < ·)
    | _ => #[]
  if keys obj != #["format", "program", "query"] then throw "partition: the object's keys are not format, program, query"
  if (obj.getObjValAs? String "format").toOption != some PARTITION_V1 then throw s!"partition: not a {PARTITION_V1} object"
  if !isSha512Hex ((obj.getObjValAs? String "program").toOption.getD "") then throw "partition: program is not a SHA-512 digest"
  let q := (obj.getObjVal? "query").toOption.getD Json.null
  if keys q != #["name", "params", "version"] then throw "partition: the query's keys are not name, version, params"
  let name := (q.getObjValAs? String "name").toOption.getD ""
  if name.isEmpty then throw "partition: the query's name is not a non-empty string"
  let version ← match (q.getObjVal? "version").toOption with
    | some (.num n) => if n.exponent == 0 && n.mantissa ≥ 0 then pure n.mantissa.toNat else throw "partition: the query's version is not a natural number"
    | _ => throw "partition: the query's version is not a natural number"
  let ps := (q.getObjVal? "params").toOption.getD Json.null
  match ps with
  | .obj _ => pure ()
  | _ => throw "partition: the query's params are not an object"
  if EVALUABLE.contains (name, version) then
    if keys ps != #["W", "X"] then throw s!"partition: Q_word v{version}'s parameters are not X, W"
    for k in ["X", "W"] do
      match (ps.getObjVal? k).toOption with
      | some (.num n) => if !(n.exponent == 0 && n.mantissa > 0) then throw s!"partition: {k} is not a positive integer"
      | _ => throw s!"partition: {k} is not a positive integer"
  if (name, version) == Q_CALL then
    if keys ps != #["X"] then throw "partition: Q_call v1's parameters are not X"
    match (ps.getObjVal? "X").toOption with
    | some (.num n) =>
      if !(n.exponent == 0 && n.mantissa > 0 && n.mantissa ≤ Qcall.PARAM_MAX) then
        throw s!"partition: X is not an integer in 1..{Qcall.PARAM_MAX}"
    | _ => throw s!"partition: X is not an integer in 1..{Qcall.PARAM_MAX}"
  if let some ts := TemplateQuery.templatesOf name version ps then discard ts
  return (name, version)

/-- `SHA-512("verity/partition/v1\0" ‖ canon(object))`. -/
def partitionDigest (obj : Json) : V ByteArray := do
  return Sha512.hash (PARTITION_V1.toUTF8.push 0 ++ (← canon obj).toUTF8)

/-- What the statement check established about the partition: the rule, the query, the digest, and whether the verifier
derived the units itself or took them as stated (the audit reports the latter as a gap). -/
structure PartitionFinding where
  rule : String
  query : String := ""
  digest : String := ""
  derived : Bool := false
  note : String
  deriving Inhabited

def PartitionFinding.toJson (f : PartitionFinding) : Json :=
  Json.mkObj [("rule", Json.str f.rule), ("query", Json.str f.query), ("digest", Json.str f.digest),
    ("units", Json.str (if f.derived then "derived" else "as stated")), ("note", Json.str f.note)]

/-- The circuit's template: META `program_digests`' one key (the Definition it lowers). -/
def circuitTemplate (c : Circuit) : Option String :=
  match (c.metaJ.getObjVal? "program_digests").toOption with
  | some (.obj kvs) => if kvs.toArray.size == 1 then some kvs.toArray[0]!.1 else none
  | _ => none

/-- The template a template query must list: the circuit's (`circuitTemplate`), or for a unit class's circuit (META
`template` `unit-class`, its one template `unit-class/…`) META `definition`, the Definition its units are instances of. That
the class circuit lowers that Definition is the class's own claim, which the verifier does not check; the units it names
are then held to instances of it. -/
def queryTemplate (c : Circuit) : Option (String × Bool) := do
  let t ← circuitTemplate c
  match (c.metaJ.getObjValAs? String "template").toOption, (c.metaJ.getObjValAs? String "definition").toOption with
  | some "unit-class", some d => if t.startsWith "unit-class/" then some (d, true) else some (t, false)
  | _, _ => some (t, false)

/-- A template query's units, derived from the verifier's own program: the program is the object's, the query applies to it,
the circuit's template (`queryTemplate`) is one it lists, and every unit the statement claims (`indices`) is an instance of
that template. -/
def deriveTemplateUnits (c : Circuit) (obj : Json) (program : ByteArray) (indices : Array Nat) : V String := do
  let q := (obj.getObjVal? "query").toOption.getD Json.null
  let name := (q.getObjValAs? String "name").toOption.getD ""
  let version := (q.getObjValAs? Nat "version").toOption.getD 0
  let some ts := TemplateQuery.templatesOf name version ((q.getObjVal? "params").toOption.getD Json.null)
    | throw s!"partition: {name} v{version} is not a template query"
  let ts ← ts
  let d ← TemplateQuery.Desc.parse program
  if (obj.getObjValAs? String "program").toOption != some (hexOf d.sha512) then
    throw "partition: the verifier's program is not the one the partition object names"
  let rs ← match TemplateQuery.evaluate d ts with
    | .ok rs => pure rs
    | .error e => throw s!"partition: {e}"
  let some (t, cls) := queryTemplate c | throw "partition: META program_digests does not name the circuit's one template"
  if !ts.contains t then throw s!"partition: the circuit's template {t} is not one the query lists"
  let pop := TemplateQuery.population rs
  for i in indices do
    match TemplateQuery.locate rs i with
    | none => throw s!"partition: unit {i} is outside the partition's {pop} units"
    | some (_, _, ti) => if ti != t then throw s!"partition: unit {i} is an instance of {ti}, not the circuit's template {t}"
  let of := if cls then s!" (META definition of the unit class {(circuitTemplate c).getD ""}; that the class circuit lowers it is not checked)" else ""
  return s!"units derived: {name} v{version} over the verifier's program ({pop} units); the statement's {indices.size} units are instances of {t}{of}"

/-- `Q_word` v1's or v2's units, derived from the verifier's own program: the program is the object's, the query evaluates on
it (`Flock.Extract`, `Flock.Qword`; v2's units are v1's), and every unit the statement claims (`indices`) is one whole
activation of the circuit's template; and every Call's cut holds the partition invariant and the width rule on its whole graph
(`Extract.checkCalls`, as `partition_object.verify` checks them), under v2 with a value computed in two units reported, not
refused. -/
def deriveQwordUnits (c : Circuit) (obj : Json) (program : ByteArray) (indices : Array Nat) : V String := do
  let q := (obj.getObjVal? "query").toOption.getD Json.null
  let version := (q.getObjValAs? Nat "version").toOption.getD 0
  let ps := (q.getObjVal? "params").toOption.getD Json.null
  let X := (ps.getObjValAs? Nat "X").toOption.getD 0
  let W := (ps.getObjValAs? Nat "W").toOption.getD 0
  let P ← Program.decode program
  if (obj.getObjValAs? String "program").toOption != some (hexOf P.sha512) then
    throw "partition: the verifier's program is not the one the partition object names"
  let calls ← match Extract.evaluate P X with
    | .ok cs => pure cs
    | .error e => throw s!"partition: {e}"
  let some t := circuitTemplate c | throw "partition: META program_digests does not name the circuit's one template"
  let checked ← Extract.checkCalls P calls X W (report := version == 2)
  if let some (k, r) := (Extract.cutFailures calls checked)[0]? then
    throw s!"partition: cut: Call {k}'s cut fails {r.codes}{if r.wide.isEmpty then "" else s!" (too wide, (unit, bits, values): {(r.wide.extract 0 4).toList})"}"
  let across := if version == 2 then
      s!"; values computed in two units, reported, not refused: {checked.foldl (fun n (_, r) => n + r.across) 0} pair(s) over its Definitions"
    else ""
  match Extract.claimedUnits P calls t indices with
  | .ok note => return s!"units derived: Q_word v{version} {"{"}X: {X}, W: {W}{"}"} over the verifier's program ({note}); every Call's cut holds the partition invariant and the width rule{across}"
  | .error e => throw s!"partition: {e}"

/-- `Q_call` v1's units, derived from the verifier's own program: the program is the object's, the query evaluates on it
(`Flock.Qcall`), every unit the statement claims (`indices`) is one whole activation of the circuit's template, and every Call's
cut holds the partition invariant and at most X bits out of every unit, a value computed in two units reported, not refused
(as `partition_object.verify` checks them). -/
def deriveQcallUnits (c : Circuit) (obj : Json) (program : ByteArray) (indices : Array Nat) : V String := do
  let q := (obj.getObjVal? "query").toOption.getD Json.null
  let X := ((q.getObjVal? "params").toOption.getD Json.null |>.getObjValAs? Nat "X").toOption.getD 0
  let P ← Program.decode program
  if (obj.getObjValAs? String "program").toOption != some (hexOf P.sha512) then
    throw "partition: the verifier's program is not the one the partition object names"
  let calls ← match Qcall.evaluate P X with
    | .ok cs => pure cs
    | .error e => throw s!"partition: {e}"
  let some t := circuitTemplate c | throw "partition: META program_digests does not name the circuit's one template"
  let mut across := 0
  let mut seen : Array Nat := #[]
  for x in calls do
    let r := Qcall.check x.c X
    if !r.codes.isEmpty then
      throw s!"partition: cut: Call {x.k}'s cut fails {r.codes}{if r.wide.isEmpty then "" else s!" (too wide, (unit, bits): {(r.wide.extract 0 4).toList})"}"
    if !seen.contains x.d then
      seen := seen.push x.d
      across := across + r.across.size
  match Qcall.claimedUnits P calls t indices with
  | .ok note => return s!"units derived: Q_call v1 {"{"}X: {X}{"}"} over the verifier's program ({note}); every Call's cut holds the partition invariant and the width rule; values computed in two units, reported, not refused: {across} pair(s) over its Definitions"
  | .error e => throw s!"partition: {e}"

/-- The partition finding of a query the template queries don't cover, given the verifier's program: `Q_word` v1's or v2's
units derived (`deriveQwordUnits`), `Q_call` v1's (`deriveQcallUnits`), any other query's finding as it stands. -/
def qwordFinding (c : Circuit) (pub : Public) (obj : Json) (program : ByteArray) (f : PartitionFinding) :
    V (Option PartitionFinding) := do
  let q := (obj.getObjVal? "query").toOption.getD Json.null
  if EVALUABLE.contains ((q.getObjValAs? String "name").toOption.getD "", (q.getObjValAs? Nat "version").toOption.getD 0) then
    let indices ← (← J.arr (← J.get pub.header "units" "public file") "indices" "public file").mapM (J.asNat · "a unit index")
    pure (some { f with derived := true, note := ← deriveQwordUnits c obj program indices })
  else if ((q.getObjValAs? String "name").toOption.getD "", (q.getObjValAs? Nat "version").toOption.getD 0) == Q_CALL then
    let indices ← (← J.arr (← J.get pub.header "units" "public file") "indices" "public file").mapM (J.asNat · "a unit index")
    pure (some { f with derived := true, note := ← deriveQcallUnits c obj program indices })
  else pure (some f)

/-- The circuit's partition binding against the verifier's own copy of the partition object: a `verity/partition/v1` binding
must be the tagged digest of that object (any query name), over META's `program_sha512`; the provisional unit cover takes
none. Units are derived only for a query the verifier evaluates, from the program's Call graphs, which it does not decode yet:
today every finding says the units were taken as stated, and why. -/
def checkPartition (c : Circuit) (obj : Option Json) : V PartitionFinding := do
  let part := (c.metaJ.getObjVal? "partition").toOption.getD Json.null
  let rule := (part.getObjValAs? String "rule").toOption.getD ""
  let digest := (part.getObjValAs? String "digest").toOption.getD ""
  match rule, obj with
  | "verity/partition/v1", none => throw "partition: the circuit binds a verity/partition/v1 digest; the verifier's partition object is required"
  | "verity/partition/v1", some o =>
    let (name, version) ← checkPartitionObject o
    if hexOf (← partitionDigest o) != digest then
      throw "partition: the circuit's partition digest is not the verifier's partition object's"
    if (o.getObjValAs? String "program").toOption != (c.metaJ.getObjValAs? String "program_sha512").toOption then
      throw "partition: the partition object is over another program than META's program_sha512"
    let note := if (TemplateQuery.templatesOf name version Json.null).isSome then
        s!"units taken as stated, not derived: the verifier's copy of the program was not given (--program) to evaluate {name} v{version}"
      else if EVALUABLE.contains (name, version) || (name, version) == Q_CALL then
        s!"units taken as stated, not derived: the verifier's copy of the program was not given (--program) to evaluate {name} v{version}"
      else s!"units taken as stated, not derived: the verifier does not evaluate {name} v{version}"
    return { rule, query := s!"{name} v{version}", digest, derived := false, note }
  | _, some _ => throw "partition: the circuit binds the provisional unit cover, not a verity/partition/v1 object"
  | _, none => return { rule, digest, derived := false, note := "units taken as stated, not derived: the provisional unit cover names no query" }

def publicDigest (header : Json) (pubs : ByteArray) : V ByteArray := do
  return Sha512.hash ((← canon header).toUTF8 ++ pubs)

/-- Header `shared_rows` (from PR #83 `967b8d06`): exactly the circuit's input ports, each a table of at least one row (a count
that is not a non-negative integer counts as none, as upstream reads it); none for a per-instance file. -/
def sharedRows (tags : Tags) (c : Circuit) (header : Json) : V (Option (Array Nat)) := do
  let .ok s := header.getObjVal? "shared_rows" | return none
  if !tags.sharedRows then throw s!"public file: shared_rows: a {tags.name} file has none (shared-row files are verity/flock-circuit@967b8d06's)"
  let .obj kvs := s | throw "header: shared_rows is not an object"
  if kvs.toArray.size != c.ports.size || c.ports.any (fun p => !(s.getObjVal? p.name).isOk) then
    throw "header: shared_rows does not name exactly the circuit's input ports"
  let counts := c.ports.map fun p => match s.getObjVal? p.name with
    | .ok (.num x) => if x.exponent == 0 && x.mantissa ≥ 0 then x.mantissa.toNat else 0
    | _ => 0
  if counts.contains 0 then throw "header: a shared_rows table has no row"
  return some counts

/-- The verifier's `flock-circuit-inputs` file (no rows, no salts): every `b ‖ c`, then every output word. A shared-row file's
`b ‖ c` are its tables' rows (port-major), followed by every instance's refs (u32 LE, instance-major, ports in order). -/
def loadPublic (tags : Tags) (c : Circuit) (file : ByteArray) : V Public := do
  let some nl := findByte file 10 | throw "public file: no header line"
  let text ← match String.fromUTF8? (file.extract 0 nl) with
    | some s => pure s
    | none => throw "public file: header is not UTF-8"
  let header ← match Json.parse text with
    | .ok j => pure j
    | .error e => throw s!"public file: header: {e}"
  let pctx := "public file"
  if (← J.str header "format" pctx) != tags.publicFormat || (← J.str header "statement" pctx) != tags.statement then
    throw s!"public file: not a {tags.publicFormat} file"
  if (← J.str header tags.publicCircuitKey pctx) != hexOf c.sha then throw "public file: staged for another circuit"
  let n ← J.nat header "instances" pctx
  if (← J.bool header "rows" pctx) then throw "public file: it holds rows (the verifier's public file has none)"
  let np := c.ports.size
  let nOut := c.outWords
  let counts ← sharedRows tags c header
  let nBcs := (counts.map (·.foldl (· + ·) 0)).getD (n * np)
  let refsLen := if counts.isSome then 4 * n * np else 0
  let body := file.extract (nl + 1) file.size
  if body.size != BC_BYTES * nBcs + refsLen + 2 * n * nOut then throw "public file: length"
  let bcs := (Array.range nBcs).map fun i => body.extract (BC_BYTES * i) (BC_BYTES * i + BC_BYTES)
  let o := BC_BYTES * nBcs + refsLen
  let outputs := (Array.range (n * nOut)).map fun i => (body.get! (o + 2 * i)).toNat + 256 * (body.get! (o + 2 * i + 1)).toNat
  let tables ← match counts with
    | none => pure none
    | some counts => do
      let refs := (Array.range (n * np)).map fun k => (u32OfLE body (BC_BYTES * nBcs + 4 * k)).toNat
      if (Array.range refs.size).any (fun k => refs[k]! ≥ counts[k % np]!) then
        throw "shared rows: an instance's ref is past its table"
      let starts := (counts.foldl (fun (acc : Array Nat × Nat) r => (acc.1.push acc.2, acc.2 + r)) (#[], 0)).1
      pure (some ({ counts, starts, refs } : Tables))
  let units ← J.get header "units" pctx
  let indices ← (← J.arr units "indices" pctx).mapM (J.asNat · "a unit index")
  if !c.classSha.isEmpty then
    let classes ← (← J.arr units "classes" pctx).mapM (J.asStr · "a class")
    if classes.size != n || classes.any (· != hexOf c.classSha) then
      throw s!"header: an instance's class is not this circuit's class {(hexOf c.classSha).take 16}…"
  if indices.size != n || (List.range indices.size).any (fun i => i > 0 && indices[i - 1]! ≥ indices[i]!)
      || (← canon units) != (← canon (unitsEntry c indices)) then
    throw "header: the instances' unit indices are not ascending, one per instance, blocks of vus_per_block"
  if J.has header "unit_draw" then throw "header: a staged file is a registered population; a draw is made in the session"
  return { header, n, digests := bcs, outputs, sha := ← publicDigest header body, tables }

/-- The drawn units' statement over a registered population (upstream's `Instances::drawn`): instance `i` is the
population's `units[i]`; the header is the population's with the draw, `instances = K`, `population = N` and the drawn
units' indices; the public digest covers that header and the population's public bytes. A shared-row population's tables
stay the statement's, with the drawn units' refs into them. -/
def drawn (c : Circuit) (pub : Public) (draw : Json) : V Public := do
  let d ← Draw.UnitDraw.ofJson draw
  let proved := d.proved
  if d.population != pub.n || pub.population.isSome then throw "U: the draw's population is not this registered population"
  let np := c.ports.size
  let nOut := c.outWords
  let pctx := "public file"
  let popIdx ← (← J.arr (← J.get pub.header "units" pctx) "indices" pctx).mapM (J.asNat · "a unit index")
  let indices := proved.map fun u => popIdx.getD u 0
  let header := ((pub.header.setObjVal! "unit_draw" draw).setObjVal! "instances" (Lean.toJson proved.size)
    |>.setObjVal! "population" (Lean.toJson pub.n)).setObjVal! "units" (unitsEntry c indices)
  let mut pubs := pub.digests.foldl (· ++ ·) ByteArray.empty
  if let some t := pub.tables then
    for r in t.refs do pubs := pubs ++ le32 r.toUInt32
  for w in pub.outputs do pubs := pubs.push (w % 256).toUInt8 |>.push (w / 256).toUInt8
  return { header, n := proved.size,
           digests := if pub.tables.isSome then pub.digests
             else proved.foldl (fun acc u => acc ++ pub.digests.extract (u * np) (u * np + np)) #[],
           outputs := proved.foldl (fun acc u => acc ++ pub.outputs.extract (u * nOut) (u * nOut + nOut)) #[],
           sha := ← publicDigest header pubs, population := some (pub.n, pub.digests, pub.outputs),
           tables := pub.tables.map fun t => { t with refs := proved.foldl (fun acc u => acc ++ t.refs.extract (u * np) (u * np + np)) #[] } }

/-- The strata of the verifier's own program and partition object, and the population: the query applied to the program,
its units grouped by template descriptor id (`Draw.strataOf`). Only `Q_template_instance` v0 and `Q_template_instances` v0
define strata; `law` names the law that asks, in the refusal. -/
private def templateStrata (law : String) (obj : Json) (program : ByteArray) :
    V (Array (String × Array (Nat × Nat)) × Nat) := do
  let (name, version) ← checkPartitionObject obj
  let q := (obj.getObjVal? "query").toOption.getD Json.null
  let some ts := TemplateQuery.templatesOf name version ((q.getObjVal? "params").toOption.getD Json.null)
    | throw s!"{law}: {name} v{version} defines no strata (Q_template_instance v0 and Q_template_instances v0 do)"
  let ts ← ts
  let d ← TemplateQuery.Desc.parse program
  if (obj.getObjValAs? String "program").toOption != some (hexOf d.sha512) then
    throw "partition: the verifier's program is not the one the partition object names"
  let rs ← match TemplateQuery.evaluate d ts with
    | .ok rs => pure rs
    | .error e => throw s!"partition: {e}"
  return (Draw.strataOf (rs.map fun r => (r.template, r.first, r.count)), TemplateQuery.population rs)

/-- `stratified:K` expanded over the verifier's own program and partition object (`draw.derive`), and the population. -/
def stratifiedLaw (obj : Json) (program : ByteArray) (k : Nat) : V (Draw.Law × Nat) := do
  let (strata, population) ← templateStrata "stratified" obj program
  match Draw.derive k strata with
  | .ok law => return (law, population)
  | .error e => throw s!"stratified: {e}"

/-- `work:K` expanded over the verifier's own program, partition object and work table of works and floors
(`Draw.deriveWork`), and the population. -/
def workLaw (obj : Json) (program : ByteArray) (k : Nat) (table : Array (String × Nat × Nat)) : V (Draw.Law × Nat) := do
  let (strata, population) ← templateStrata "work" obj program
  match Draw.deriveWork k strata table with
  | .ok law => return (law, population)
  | .error e => throw s!"work: {e}"

/-- The frame-v3-sha512 roots recomputed from the population's `b ‖ c` (a shared-row file: each input port's over its table)
and output words against the header's; the pinned dummy `b ‖ c` against the zero row's (upstream's `check_public`). A
statement with public outputs has no output row to link, so META `links` (§16.13) is refused. -/
def checkPublic (c : Circuit) (pub : Public) : V Unit := do
  if J.has c.metaJ "links" then throw "META links: a statement with public outputs has no output row to link (links are §16.13's)"
  let fv ← pub.frameV3
  if (← J.str fv "frame" "public file") != "frame-v3-sha512"
      || (← J.str fv "row_key_sha512" "public file") != hexOf (Sha512.hash Hm96.Default512.key) then
    throw "the instance file's rows are not frame-v3-sha512 hm96 leaves under the default key"
  let np := c.ports.size
  let (n, bcs, outputs) := pub.population.getD (pub.n, pub.digests, pub.outputs)
  if let .ok d := pub.header.getObjVal? "unit_draw" then
    let u ← Draw.UnitDraw.ofJson d
    if (← J.nat pub.header "instances" "public file") != u.proved.size || pub.n != u.proved.size then
      throw "U3: the statement's instances are not the drawn units and their closure"
  let H := Sha512.hash
  for p in [0:np] do
    let port := c.ports[p]!
    let dom ← pub.fv3Bytes "domain_ids" port.name
    let schema := if port.segs.isSome then ROW_SCHEMA_SEG else if port.v2 then ROW_SCHEMA_V2 else ROW_SCHEMA
    if (← J.str (← J.get fv "schemas" "public file") port.name "public file") != schema then
      throw s!"input port {port.name}: schema is not {schema}"
    let (count, row) : Nat × (Nat → Nat) := match pub.tables with
      | some t => (t.counts[p]!, fun r => t.starts[p]! + r)
      | none => (n, fun i => i * np + p)
    let leaves := (Array.range count).map fun r => FV3.leaf dom r schema (Hm96.Default512.treeLeaf (bcs.getD (row r) ByteArray.empty)) H
    if FV3.root dom leaves H != (← pub.fv3Bytes "roots" port.name) then
      throw s!"input port {port.name}: the rows' b || c do not hash to the header's frame-v3-sha512 root"
    if zeroRowBc port != c.dummyDigests.getD p ByteArray.empty then
      throw s!"the pinned dummy b || c of port {port.name} is not the zero row's"
  let nOut := c.outWords
  let mut at_ := 0
  for (oname, w) in c.outPorts do
    let dom ← pub.fv3Bytes "domain_ids" oname
    let words := (Array.range n).foldl (fun acc i => acc ++ (Array.range w).map fun k => outputs.getD (i * nOut + at_ + k) 0) #[]
    let leaves := (Array.range words.size).map fun r =>
      FV3.leaf dom r "u16" (ByteArray.mk #[(words[r]! / 256).toUInt8, (words[r]! % 256).toUInt8]) H
    if FV3.root dom leaves H != (← pub.fv3Bytes "roots" oname) then
      throw s!"output port {oname}: the output words do not hash to the header's frame-v3-sha512 root"
    at_ := at_ + w

/-- The regions: each port's `b ‖ c` of every VU (its hm96 slot's output; a padding VU's the pinned dummy), then every
unit slot's returned-output words. -/
def regions (c : Circuit) (pub : Public) : V (Array Region) := do
  let kl := c.kLog
  let bits := fun (lo n : Nat) => (Array.range n).map (lo + ·)
  let hr := c.rangeOf c.hmNet
  let hm := (c.nets.getD c.hmNet default).2
  let lp := Nat.log2 (c.perVu.getD c.hmNet 1)
  let lg := Nat.log2 c.g
  let bc0 := ((portCols hm.outGroups).getD 0 default).1
  let np := c.ports.size
  let mut out := #[]
  for p in [0:np] do
    out := out.push (← mkRegion kl s!"digest {p}" (bits 0 10 ++ bits (hr.slotLog + lp) lg) (hr.start + p * 2 ^ hr.slotLog + bc0)
      fun b => (Array.range c.g).foldl (fun acc g => acc ++ match c.instOf pub.n b g with
        | some i => pub.digest i p np
        | none => c.dummyDigests.getD p ByteArray.empty) ByteArray.empty)
  if c.outNet != c.unitNet || c.typed.isSome then
    -- the last stage's (or a typed instance's root's) output group is the output words (one slot per VU)
    let onet := (c.nets.getD c.outNet default).2
    let r := c.rangeOf c.outNet
    let nOut := c.outWords
    let tz := fun (x : Nat) => (List.range 64).find? (fun i => x % 2 ^ (i + 1) != 0) |>.getD 0
    return out.push (← mkRegion kl "out" (bits 0 (tz (16 * nOut)) ++ bits r.slotLog (tz r.count))
      (r.start + (onet.outGroups.getD 0 default).col * WORD) fun b => (Array.range c.g).foldl (fun acc g =>
        (Array.range nOut).foldl (fun acc o =>
          let v := match c.instOf pub.n b g with
            | some i => pub.outputs.getD (i * nOut + o) 0
            | none => c.dummyOutputs.getD o 0
          acc.push (v % 256).toUInt8 |>.push (v / 256).toUInt8) acc) ByteArray.empty)
  return out.push (← c.outRegion pub)

/-- The columns that share a 128-bit word with a partial region's columns without being one of them, as (region, column).
A region is partial when some in-word bit (0 to 6) is not free; its covered words are its fixed bits above 6 with each
combination of its free bits above 6. -/
def sharedColumns (regions : Array Region) : Array (Nat × Nat) := Id.run do
  let mut out := #[]
  for ri in [0:regions.size] do
    let r := regions[ri]!
    if !(List.range 7).all r.free.contains then
      let high := r.free.filter (· ≥ 7)
      for a in [0:2 ^ high.size] do
        let w := (List.range high.size).foldl (fun x i => x ||| (a >>> i) % 2 * 2 ^ high[i]!) (r.fixed / WORD * WORD)
        for v in [0:WORD] do
          if (w + v) &&& r.mask != r.fixed then out := out.push (ri, w + v)
  return out

/-- A sorted list of columns sums to zero over GF(2): every column appears an even number of times. -/
def evenRuns (s : Array Nat) : Bool := s.size % 2 == 0 && (List.range (s.size / 2)).all fun i => s[2 * i]! == s[2 * i + 1]!

/-- **The region-word check** (`docs/region-word-check.md`; the red team's C3 on the public-circuit ZK proof, H_reg): a
region claim's ring-switch message carries every bit of each word the region covers in the clear, so a column sharing a
word with a region's columns must be one of them, or be forced to zero: its rows in the combined `A` and `B` both empty.
Row `q` of the combined `A` is the row of the slot type placed at `q` in block columns (a mask slot's is `[q]`), plus every
`Δ_A` entry `(q, col)`, summed over GF(2) as `addDelta` sums them (an entry twice cancels); likewise `B`. -/
def checkRegionWords (c : Circuit) (da db : Array (Nat × Nat)) (regions : Array Region) : V Unit := do
  let shared := sharedColumns regions
  let want : Std.HashSet Nat := shared.foldl (fun s e => s.insert e.2) {}
  let byRow := fun (d : Array (Nat × Nat)) => d.foldl (fun (m : Std.HashMap Nat (Array Nat)) e =>
    if want.contains e.1 then m.insert e.1 ((m.getD e.1 #[]).push e.2) else m) {}
  let (ra, rb) := (byRow da, byRow db)
  let empty := fun (q : Nat) (delta : Std.HashMap Nat (Array Nat)) (useA : Bool) => Id.run do
    let mut cols := delta.getD q #[]
    if let some r := c.ranges.find? (fun r => r.start ≤ q && q < r.start + r.count * 2 ^ r.slotLog) then
      let slot0 := q / 2 ^ r.slotLog * 2 ^ r.slotLog
      match r.kind with
      | .net i =>
        let net := (c.nets.getD i default).2
        cols := cols ++ ((if useA then net.a else net.b).row (q - slot0)).map (slot0 + ·)
      | _ => cols := cols.push q
    return evenRuns (cols.qsort (· < ·))
  for (ri, q) in shared do
    if !(empty q ra true && empty q rb false) then
      throw s!"region {(regions.getD ri default).name}: column {q} shares a word with the region's bits and is not forced to zero"

end HmRow

/-- `setup` for an `hm96-sha512/row/v1` statement: the circuit, the population's public file and, for a drawn session, the
drawn units' statement over it. -/
def Stmt.setupH (tags : Tags) (circuitFile publicFile : ByteArray) (coins : Coins) (tables : List (String × ByteArray) := [])
    (draw : Option Json := none) (partition : Option Json := none) (program : Option ByteArray := none) : V Stmt := do
  let c ← if tags.typed then HmRow.parseTyped tags circuitFile tables else HmRow.parse tags circuitFile tables
  let finding0 ← if tags.bindings then (some <$> HmRow.checkPartition c partition)
    else if partition.isSome then throw s!"{tags.name}: the statement binds no partition" else pure none
  let some merkle := MerkleScheme.find? c.leafScheme | throw s!"leaf-commitment scheme {c.leafScheme}: not one this verifier implements"
  let pop ← HmRow.loadPublic tags c publicFile
  let pub ← match draw with
    | some d => HmRow.drawn c pop d
    | none => pure pop
  -- a stratified draw's strata are the verifier's own, from its program and query, when it holds both
  if let some d := draw then
    if let (Except.ok (Draw.Law.stratified k strata), some obj, some prog) :=
        ((Draw.UnitDraw.ofJson d).map (·.law), partition, program) then
      let (law, _) ← HmRow.stratifiedLaw obj prog k
      if law != Draw.Law.stratified k strata then throw "U2: the draw's strata are not the verifier's own (its program and query)"
  HmRow.checkPublic c pub
  -- a template query's units, derived from the verifier's own program: the statement's units must be instances of its template
  let finding ← match finding0, partition, program with
    | some f, some obj, some prog =>
      let q := (obj.getObjVal? "query").toOption.getD Json.null
      if (TemplateQuery.templatesOf ((q.getObjValAs? String "name").toOption.getD "") ((q.getObjValAs? Nat "version").toOption.getD 0) Json.null).isSome then
        let indices ← (← J.arr (← J.get pub.header "units" "public file") "indices" "public file").mapM (J.asNat · "a unit index")
        pure (some { f with derived := true, note := ← HmRow.deriveTemplateUnits c obj prog indices })
      else HmRow.qwordFinding c pub obj prog f
    | f, _, _ => pure f
  let n := pub.n
  let blocksReal := (n + c.g - 1) / c.g
  let nbl := max 3 (log2ceil (max blocksReal 1))
  let m := c.kLog + nbl
  if c.kLog > 27 || m > 35 then throw s!"m = {m}: no fast100 schedule (or k_log > 27)"
  let pin ← HmRow.pin c
  let (da, db) := HmRow.delta c pin
  let regions ← HmRow.regions c pub
  HmRow.checkRegionWords c da db regions
  checkInRange c.kLog regions.size (PT_LOCAL + nbl)
  let identity := if J.has c.metaJ "row_schemas" then Tags.rowV2Identity tags.identity else tags.identity
  let identity := if c.ports.any (·.segs.isSome) then Tags.rowSegIdentity identity else identity
  let v2Run := (List.range c.outRowPorts.size).any fun j =>
    (c.ports.getD (c.ports.size - c.outRowPorts.size + j) default).segs.isNone && c.outRowPorts[j]! > 1
  let identity := if !J.has c.metaJ "out_row_ports" then identity
    else if v2Run then Tags.outRowsV2Identity identity else Tags.outRowsIdentity identity
  let mut h := tags.statement.toUTF8 ++ c.sha ++ (← canon identity).toUTF8
  for x in [n, blocksReal, nbl, m, c.kLog, pin, regions.size, c.g] do h := pushLe64 h x
  for d in [da, db] do
    h := pushLe64 h d.size
    for (i, cc) in d do h := pushLe32 (pushLe32 h i) cc
  let digest := Sha512.hash h
  let sigma := Sha512.hash (tags.sigmaTag.toUTF8 ++ c.sha ++ digest ++ pub.sha)
  let hello ← helloOf tags sigma (PT_LOCAL + nbl) coins
  return { tags, leaf := default, c, pub, n, blocksReal, nbl, m, mPts := PT_LOCAL + nbl, pin, regions, da, db, digest, sigma,
           hello, merkle, partition := (finding.map (·.toJson)).getD Json.null }

/-!
PROTOCOL.md §16.13: hidden outputs (`Tags.hiddenOutputs`, the default `verity/flock-circuit` and `verity/flock-circuit/types`).
The rows are the input rows (META `input_ports` of them), then one output row per output port: its name, its words
zero-padded to whole SHA-512 blocks, an `hm96-sha512/row/v1` row (in a statement of bit rows, a `v2` row of all their bits)
hashed in the circuit like an input row; or, per run of output ports in META `out_row_ports`, one `hm96-sha512/row-seg/v1`
row of exactly their words in port order, which is the row a unit reading it as one input port commits
(`frame_v3/PROTOCOL.md` §6). Δ adds each output row's message bits, copies of the output bits the circuit
computes; the regions are every row's `b ‖ c`, and the public file holds nothing else of the outputs. The rows' parse, checks and Δ are `HmRow`'s over every row.
-/
namespace HmOut

open HmRow

/-- The input rows: the rows before one output row per run of output ports (`outRowPorts`). -/
def nIn (c : Circuit) : Nat := c.ports.size - c.outRowPorts.size

def inWords (c : Circuit) : Nat := (c.ports.extract 0 (nIn c)).foldl (· + ·.words) 0

/-- Output row `j`'s run of output ports: its first port, their count, the output word it starts at and their words. -/
def outGroup (c : Circuit) (j : Nat) : Nat × Nat × Nat × Nat :=
  let first := (c.outRowPorts.extract 0 j).foldl (· + ·) 0
  let k := c.outRowPorts.getD j 0
  (first, k, (c.outPorts.extract 0 first).foldl (· + ·.2) 0, (c.outPorts.extract first (first + k)).foldl (· + ·.2) 0)

/-- What makes the rows inputs then outputs: META `input_ports` is the rows before one output row per output port, each
named after its port with its words zero-padded to whole SHA-512 blocks (a bit row of all their bits; a `sha512/row/v2`
row may instead be exactly its port's words), or per run of output ports (META `out_row_ports`) a `sha512/row-seg/v1` row
of exactly their words named after its first, every input row's and output port's name once (so every row's: an output
row is named after its run's first port); a unit's leaves and leaf cuts read input rows only. -/
def checkRows (c : Circuit) : V Unit := do
  let k ← J.nat c.metaJ "input_ports" "circuit META"
  let names := ((c.ports.extract 0 k).map (·.name) ++ c.outPorts.map (·.1)).qsort (· < ·)
  if k + c.outRowPorts.size != c.ports.size || (List.range (names.size - 1)).any (fun i => names[i]! == names[i + 1]!)
      || (List.range c.outRowPorts.size).any (fun j =>
        let (first, n, _, w) := outGroup c j
        let p := c.ports.getD (k + j) default
        let padded := n == 1 && p.words == (w + 63) / 64 * 64 && p.bits == 16 * p.words
        (c.outPorts[first]?.map (·.1)) != some p.name ||
          !(if p.segs.isSome then p.bits == 16 * w else if p.v2 then p.bits == 16 * w || padded else padded)) then
    throw "the rows are not the input rows, then one output row per output port (its name, its words zero-padded to whole SHA-512 blocks, all their bits) or per run of output ports (META out_row_ports), a bit row of exactly their words (sha512/row/v2 or sha512/row-seg/v1) named after the first, each name once"
  let w := inWords c
  if c.leavesIn.any (·.any (· ≥ (w : Int))) || c.leafCuts.any (·.2.2 ≥ w) then
    throw "a unit's leaf or leaf cut reads an output row: units read the input rows only"

/-- Parse the circuit file as `HmRow.parse` (a typed one as `HmRow.parseTyped`) reads it, its rows every row, and check
them (`checkRows`). `HmRow.checkTyped` holds a template's instance inputs to every row, so it reads the template with the
output rows' words after its inputs, and the circuit keeps the template's own block. -/
def parse (tags : Tags) (bytes : ByteArray) (tables : List (String × ByteArray) := []) : V Circuit := do
  let c ← if !tags.typed then HmRow.parse tags bytes tables else do
    let (k, r) ← Typed.read tags bytes tables
    match r with
    | .inr t =>
      let c ← HmRow.parse tags bytes tables (tmpl := some { t with ins := t.ins ++ t.outs.map fun (w, b) => ((w + 63) / 64 * 64, b) })
      pure { c with typed := some (blockOf t), cls := some k }
    | .inl unit => pure { (← HmRow.parse tags bytes tables (pre := #[("unit", unit)])) with cls := some k }
  checkRows c
  return c

/-- Row `p` of the padding instance (u16 LE): zero, or an output row's pinned dummy outputs, zero-padded. -/
def dummyRow (c : Circuit) (p : Nat) : ByteArray := Id.run do
  let words := (c.ports.getD p default).words
  let k := nIn c
  let (_, _, at_, w) := if p < k then (0, 0, 0, 0) else outGroup c (p - k)
  let mut row := ByteArray.empty
  for i in [0:words] do
    let v := if i < w then c.dummyOutputs.getD (at_ + i) 0 else 0
    row := row.push (v % 256).toUInt8 |>.push (v / 256).toUInt8
  return row

/-- Row `p`'s `b ‖ c` under a zero salt: `b = x`, the port's row digest of the row's first `p.bytes` bytes (zeros past
`row`; a zero salt's mask is zero), `c = SHA-512(salt_prefix ‖ 0^192)`. -/
def rowBcZeroSalt (p : Port) (row : ByteArray) : ByteArray :=
  let bytes := row.extract 0 p.bytes ++ ByteArray.mk (Array.replicate (p.bytes - row.size) 0)
  (if p.segs.isSome then segDigest p.bits bytes else Sha512.hash (prefixOf p ++ bytes)) ++
    Sha512.hash (Hm96.Default512.saltPrefix ++ ByteArray.mk (Array.replicate SALT_BYTES 0))

/-- `g o ↦` the first column of output word `o` of VU `g` (16 bits from it), where the circuit computes it: a typed
instance's root's output group, the last stage's output port, or the unit that returns it. -/
def outCol (c : Circuit) : Nat → Nat → Nat :=
  let unit := c.unit
  let ret := (List.range c.upv).foldl (fun r u =>
    ((c.leavesOut.getD u #[]).foldl (fun (r, q) o => (r.set! o (u, q), q + 1)) (r, 0)).1) (Array.replicate c.outWords (0, 0))
  let unitCols := portCols #[unit.outGroups.getD 0 default]
  let tailCols := portCols (c.nets.getD c.outNet default).2.outGroups
  fun g o =>
    if c.typed.isSome then c.slotCol c.unitNet g 0 + (unit.outGroups.getD 0 default).col * WORD + 16 * o
    else if c.outNet != c.unitNet then c.slotCol c.outNet g 0 + (tailCols.getD (c.outNetPorts.getD o 0) default).1
    else
      let (u, q) := ret.getD o (0, 0)
      c.slotCol c.unitNet g u + (unitCols.getD q default).1

/-- Δ: `HmRow.delta`'s, then per VU each output row's message bits: word `k` bit `t` (bit `t % 8` of row byte `2k + t / 8`) a
copy of the output bit, zero (the unit range's first slot's forced-zero row) past its port's words. -/
def delta (c : Circuit) (pin : Nat) : Array (Nat × Nat) × Array (Nat × Nat) := Id.run do
  let (da, db) := HmRow.delta c pin
  let (_, where_) := rowSlots c.ports
  let sin := portCols (c.nets.getD c.shaNet default).2.inGroups
  let hin := portCols (c.nets.getD c.hmNet default).2.inGroups
  let msgCol := fun (g p byte t : Nat) =>
    let (k, o) := byteAt (c.ports.getD p default) byte
    if k < where_[p]!.size then
      let (slot, sub) := where_[p]![k]!
      c.slotCol c.shaNet g slot + sin[2 * sub + 1]!.1 + 64 * (o / 8) + 56 - 8 * (o % 8) + t
    else c.slotCol c.hmNet g p + hin[1]!.1 + 64 * (o / 8) + 56 - 8 * (o % 8) + t
  let forcedZero := c.unitRange.slot 0 + c.unit.useful
  let col := outCol c
  let k0 := nIn c
  let mut d : Array (Nat × Nat) := #[]
  for g in [0:c.g] do
    for j in [0:c.outRowPorts.size] do
      let (_, _, at_, w) := outGroup c j
      for k in [0:(c.ports.getD (k0 + j) default).words] do
        for t in [0:16] do
          let i := msgCol g (k0 + j) (2 * k + t / 8) (t % 8)
          d := d.push (i, i) |>.push (i, if k < w then col g (at_ + k) + t else forcedZero)
  return (da ++ d, db ++ d)

/-- The regions: each row's `b ‖ c` of every VU (its hm96 slot's output; a padding VU's the pinned dummy). -/
def regions (c : Circuit) (pub : Public) : V (Array Region) := do
  let bits := fun (lo n : Nat) => (Array.range n).map (lo + ·)
  let hr := c.rangeOf c.hmNet
  let hm := (c.nets.getD c.hmNet default).2
  let lp := Nat.log2 (c.perVu.getD c.hmNet 1)
  let bc0 := ((portCols hm.outGroups).getD 0 default).1
  let np := c.ports.size
  (Array.range np).mapM fun p =>
    mkRegion c.kLog s!"digest {p}" (bits 0 10 ++ bits (hr.slotLog + lp) (Nat.log2 c.g)) (hr.start + p * 2 ^ hr.slotLog + bc0)
      fun b => (Array.range c.g).foldl (fun acc g => acc ++ match c.instOf pub.n b g with
        | some i => pub.digest i p np
        | none => c.dummyDigests.getD p ByteArray.empty) ByteArray.empty

/-- The verifier's public file: `HmRow.loadPublic`'s with no output word. A shared-row file's output rows are each
instance's own: their tables one row per instance, every instance's ref its own row. -/
def loadPublic (tags : Tags) (c : Circuit) (file : ByteArray) : V Public := do
  let pub ← HmRow.loadPublic tags { c with outPorts := #[] } file
  if let some t := pub.tables then
    let (np, k) := (c.ports.size, nIn c)
    if (t.counts.extract k np).any (· != pub.n) then throw "header: an output row's table is not one row per instance"
    if (List.range t.refs.size).any (fun x => x % np ≥ k && t.refs[x]! != x / np) then
      throw "shared rows: an instance's output row is not its own"
  return pub

def drawn (c : Circuit) (pub : Public) (draw : Json) : V Public := HmRow.drawn { c with outPorts := #[] } pub draw

/-- The frame-v3-sha512 roots recomputed from the population's `b ‖ c` (a shared-row file: each row's over its table)
against the header's, one tree per row; each row's pinned dummy `b ‖ c` against the padding instance's row under a zero
salt. -/
def checkPublic (c : Circuit) (pub : Public) : V Unit := do
  let fv ← pub.frameV3
  if (← J.str fv "frame" "public file") != "frame-v3-sha512"
      || (← J.str fv "row_key_sha512" "public file") != hexOf (Sha512.hash Hm96.Default512.key) then
    throw "the instance file's rows are not frame-v3-sha512 hm96 leaves under the default key"
  let np := c.ports.size
  let (n, bcs, _) := pub.population.getD (pub.n, pub.digests, pub.outputs)
  if let .ok d := pub.header.getObjVal? "unit_draw" then
    let u ← Draw.UnitDraw.ofJson d
    if (← J.nat pub.header "instances" "public file") != u.proved.size || pub.n != u.proved.size then
      throw "U3: the statement's instances are not the drawn units and their closure"
  let H := Sha512.hash
  for p in [0:np] do
    let port := c.ports[p]!
    let dom ← pub.fv3Bytes "domain_ids" port.name
    let schema := if port.segs.isSome then ROW_SCHEMA_SEG else if port.v2 then ROW_SCHEMA_V2 else ROW_SCHEMA
    if (← J.str (← J.get fv "schemas" "public file") port.name "public file") != schema then
      throw s!"row {port.name}: schema is not {schema}"
    let (count, row) : Nat × (Nat → Nat) := match pub.tables with
      | some t => (t.counts[p]!, fun r => t.starts[p]! + r)
      | none => (n, fun i => i * np + p)
    let leaves := (Array.range count).map fun r => FV3.leaf dom r schema (Hm96.Default512.treeLeaf (bcs.getD (row r) ByteArray.empty)) H
    if FV3.root dom leaves H != (← pub.fv3Bytes "roots" port.name) then
      throw s!"row {port.name}: the rows' b || c do not hash to the header's frame-v3-sha512 root"
    if rowBcZeroSalt port (dummyRow c p) != c.dummyDigests.getD p ByteArray.empty then
      throw s!"the pinned dummy b || c of row {port.name} is not the padding instance's row under a zero salt"

end HmOut

/-!
### Public input ports (§16.10)

A statement with hidden outputs may name public input ports (META `public_inputs`): unit inputs that are no row but public
words the verifier recomputes from public data, each port naming its recomputation (its `seed` and `derivation`). The units
read them from `in` regions over whole words of every unit slot's leaf group (META `leaves_public`: each leaf's word among
the public ports' words, or -1), with no Δ copy; the public file ends with every instance's public words, and the verifier
takes them only when they equal its own recomputation (`--public-inputs`, `checkOwn`).
-/
namespace HmIn

open HmRow

def FORMAT : String := "verity/flock-public-inputs/v1"

/-- META's public input ports: the list as META has it, each port's name and words, and each unit's public leaf map. -/
structure Ports where
  json : Json := Json.null
  ports : Array (String × Nat) := #[]
  leaves : Array (Array Int) := #[]
  deriving Inhabited

def Ports.words (pi : Ports) : Nat := pi.ports.foldl (· + ·.2) 0

/-- The 128-bit words of a unit slot holding some unit's public leaf (leaf `j` in word `col + j / 8` of the leaf group),
ascending and distinct. -/
def words (c : Circuit) (pi : Ports) : Array Nat :=
  let col := (c.unit.inGroups.getD 0 default).col
  let ws := pi.leaves.foldl (fun acc l => (Array.range l.size).foldl (fun acc j => if l[j]! ≥ 0 then acc.push (col + j / 8) else acc) acc) #[]
  (ws.qsort (· < ·)).foldl (fun acc w => if acc.back? == some w then acc else acc.push w) #[]

/-- Ascending distinct words as aligned power-of-two blocks `(w, e)` (`w` a multiple of `2^e`): each maximal run of
consecutive words cut from its start, every block the largest aligned one that fits in what is left of the run. -/
def blocks (ws : Array Nat) : Array (Nat × Nat) := Id.run do
  let runs := ws.foldl (fun (acc : Array (Nat × Nat)) w => match acc.back? with
    | some (a, b) => if w == b then acc.pop.push (a, b + 1) else acc.push (w, w + 1)
    | none => #[(w, w + 1)]) #[]
  let mut out := #[]
  for (a0, b) in runs do
    let mut a := a0
    for _ in [0:b - a0] do
      if a < b then
        let e := ((List.range (Nat.log2 (b - a) + 1)).filter (a % 2 ^ · == 0)).getLast?.getD 0
        out := out.push (a, e)
        a := a + 2 ^ e
  return out

/-- The in regions as `(w, s)`, `2^s` words of every unit slot from word `w`: each block of `words`, `2^e` words, split into
`2^(e - s)` regions, `s = min(e, PT_LOCAL - 7 - log2ceil(count))` (`7 + s` free bits beside the unit range's slot bits). -/
def inRegions (c : Circuit) (pi : Ports) : Array (Nat × Nat) :=
  let top := PT_LOCAL - (7 + log2ceil c.unitRange.count)
  (blocks (words c pi)).foldl (fun acc (w, e) =>
    let s := min e top
    (Array.range (2 ^ (e - s))).foldl (fun acc r => acc.push (w + r * 2 ^ s, s)) acc) #[]

/-- META `public_inputs` (each port `{port, words, seed, derivation}`, a non-empty seed and derivation, nothing else) and
`leaves_public`, checked: a port named once and no row's name; no typed instance; one public leaf map per unit, each public
leaf a public word and a zero leaf of `leaves_in`; the unit range unpacked (the in regions read it); the in regions' words
inside the leaf group, each leaf in them a public or a zero leaf in every unit. -/
def parse (c : Circuit) : V Ports := do
  let mj := c.metaJ
  let .ok pj := mj.getObjVal? "public_inputs" | do
    if J.has mj "leaves_public" then throw "META: leaves_public without public_inputs"
    return {}
  let ps ← J.asArr pj "META public_inputs"
  if ps.isEmpty then throw "META: public_inputs names no port"
  let ports ← ps.mapM fun p => do
    let .obj kvs := p | throw "META: a public input is not an object"
    let keys := (kvs.toArray.map (·.1)).qsort (· < ·)
    let s := fun k => (p.getObjValAs? String k).toOption.getD ""
    let w := (p.getObjValAs? Nat "words").toOption.getD 0
    if keys != #["derivation", "port", "seed", "words"] || s "port" == "" || s "seed" == "" || s "derivation" == "" || w == 0 then
      throw "META: a public input names no port, words, or recomputation (a non-empty seed and derivation)"
    pure (s "port", w)
  if !J.has mj "leaves_public" then throw "META: public_inputs without leaves_public"
  let pi : Ports := { json := pj, ports, leaves := ← intLists mj "leaves_public" }
  if c.typed.isSome then throw "public inputs: a typed statement's instance reads only rows"
  let names := (ports.map (·.1) ++ c.ports.map (·.name)).qsort (· < ·)
  if (List.range (names.size - 1)).any (fun i => names[i]! == names[i + 1]!) then
    throw "public inputs: a port named twice, or both a row and a public input"
  let np : Int := pi.words
  if pi.leaves.size != c.upv || (List.range c.upv).any (fun u =>
      let (lp, li) := (pi.leaves[u]!, c.leavesIn.getD u #[])
      lp.size != li.size || (List.range lp.size).any fun j => lp[j]! < -1 || lp[j]! ≥ np || (lp[j]! ≥ 0 && li[j]! != -1)) then
    throw "public inputs: the public leaf maps are not one per unit, each public leaf a public word and a zero leaf of leaves_in"
  let ur := c.unitRange
  if ur.packed then throw "public inputs: a packed unit range (the in region reads it)"
  if 7 + log2ceil ur.count > PT_LOCAL then
    throw s!"public inputs: {ur.count} unit slots, too many for an in region of at most {PT_LOCAL} free bits"
  let ws := words c pi
  if ws.isEmpty then throw "public inputs: no unit reads a public word"
  let g0 := c.unit.inGroups.getD 0 default
  for (w, e) in blocks ws do
    if w < g0.col || w + 2 ^ e > g0.col + g0.words then
      throw s!"public inputs: the in region's words {w}..{w + 2 ^ e} leave the unit's leaf group"
  for u in [0:c.upv] do
    let (lp, li) := (pi.leaves[u]!, c.leavesIn.getD u #[])
    if let some j := (List.range li.size).find? (fun j => ws.binSearchContains (g0.col + j / 8) (· < ·) && lp[j]! < 0 && li[j]! ≥ 0) then
      throw s!"public inputs: leaf {j} reads a row word inside word {g0.col + j / 8}, which holds a public word"
  return pi

/-- Whether block column `q` is a bit of a public leaf: in unit slot `k`'s leaf group (unit `k % upv`), leaf `j`'s 16 bits
from `col 128 + 16 j`, a public word. -/
def publicCol (c : Circuit) (pi : Ports) (q : Nat) : Bool :=
  let ur := c.unitRange
  let g0 := c.unit.inGroups.getD 0 default
  let (k, at_) := ((q - ur.start) / 2 ^ ur.slotLog, (q - ur.start) % 2 ^ ur.slotLog)
  ur.start ≤ q && k < ur.count && g0.col * WORD ≤ at_ && ((pi.leaves.getD (k % c.upv) #[]).getD ((at_ - g0.col * WORD) / 16) (-1)) ≥ 0

/-- Δ: `HmOut.delta`'s without any entry of a public leaf's rows; a public leaf's bits are no copy but the in regions'. -/
def delta (c : Circuit) (pi : Ports) (pin : Nat) : Array (Nat × Nat) × Array (Nat × Nat) :=
  let (da, db) := HmOut.delta c pin
  if pi.ports.isEmpty then (da, db) else (da.filter (!publicCol c pi ·.1), db.filter (!publicCol c pi ·.1))

/-- The in regions (`inRegions`): region `r`, `(w, s)`, is `2^s` words from word `w` of every unit slot, its bytes per slot the
leaves' 16-bit words, u16 LE: a public leaf its instance's public word, zero for a zero leaf, a padding instance or a slot
past the units. -/
def regions (c : Circuit) (pi : Ports) (pub : Public) : V (Array Region) := do
  if pi.ports.isEmpty then return #[]
  let bits := fun (lo n : Nat) => (Array.range n).map (lo + ·)
  let ur := c.unitRange
  let (sb, np) := (log2ceil ur.count, pi.words)
  let col := (c.unit.inGroups.getD 0 default).col
  let ins := inRegions c pi
  (Array.range ins.size).mapM fun r =>
    let (w, s) := ins[r]!
    let j0 := 8 * (w - col)
    mkRegion c.kLog s!"in {r}" (bits 0 (7 + s) ++ bits ur.slotLog sb) (ur.start + w * WORD) fun b =>
      (Array.range (2 ^ sb)).foldl (fun acc k =>
        let (g, u) := (k / c.upv, k % c.upv)
        let i := if k < ur.count then c.instOf pub.n b g else none
        (Array.range (8 * 2 ^ s)).foldl (fun acc j =>
          let q := (pi.leaves.getD u #[]).getD (j0 + j) (-1)
          let v := match i with
            | some i => if q ≥ 0 then pub.outputs.getD (i * np + q.toNat) 0 else 0
            | none => 0
          acc.push (v % 256).toUInt8 |>.push (v / 256).toUInt8) acc) ByteArray.empty

/-- The verifier's public file: `HmOut.loadPublic`'s, then every instance's public words (`Public.outputs`), its header's
`public_inputs` META's (and none without). -/
def loadPublic (tags : Tags) (c : Circuit) (pi : Ports) (file : ByteArray) : V Public := do
  let pub ← HmRow.loadPublic tags { c with outPorts := if pi.ports.isEmpty then #[] else #[("public inputs", pi.words)] } file
  match pub.header.getObjVal? "public_inputs", pi.ports.isEmpty with
  | .error _, true => pure ()
  | .ok h, false => if (← canon h) != (← canon pi.json) then throw "public file: header public_inputs is not the circuit's"
  | _, _ => throw "public file: header public_inputs is not the circuit's"
  if let some t := pub.tables then
    let (np, k) := (c.ports.size, HmOut.nIn c)
    if (t.counts.extract k np).any (· != pub.n) then throw "header: an output row's table is not one row per instance"
    if (List.range t.refs.size).any (fun x => x % np ≥ k && t.refs[x]! != x / np) then
      throw "shared rows: an instance's output row is not its own"
  return pub

def drawn (c : Circuit) (pi : Ports) (pub : Public) (draw : Json) : V Public :=
  HmRow.drawn { c with outPorts := if pi.ports.isEmpty then #[] else #[("public inputs", pi.words)] } pub draw

/-- The verifier's own recomputation (`--public-inputs F`): a header line `{format, instances, ports}`, its ports META's
`public_inputs` (each naming its seed and derivation), then every instance of the population's public words as the public
file holds them (u16 LE, instance-major, the ports in order). The public file's words must be these; a circuit with no
public input takes no such file, and one with public inputs needs it. -/
def checkOwn (pi : Ports) (pop : Public) (own : Option ByteArray) : V Unit := do
  match own, pi.ports.isEmpty with
  | none, true => return
  | some _, true => throw "--public-inputs: the circuit names no public input port"
  | none, false => throw "public inputs: the circuit names public input ports; pass the verifier's own recomputation (--public-inputs F)"
  | some f, false =>
    let some nl := findByte f 10 | throw "--public-inputs: no header line"
    let header ← match String.fromUTF8? (f.extract 0 nl) with
      | some t => match Json.parse t with
        | .ok j => pure j
        | .error e => throw s!"--public-inputs: header: {e}"
      | none => throw "--public-inputs: header is not UTF-8"
    let ctx := "--public-inputs"
    if (← J.str header "format" ctx) != FORMAT then throw s!"--public-inputs: not a {FORMAT} file"
    let (n, pw) := ((pop.population.map (·.1)).getD pop.n, pi.words)
    if (← J.nat header "instances" ctx) != n then throw s!"--public-inputs: not the population's {n} instances"
    if (← canon (← J.get header "ports" ctx)) != (← canon pi.json) then
      throw "--public-inputs: its ports (names, words, seeds, derivations) are not the circuit's public_inputs"
    let body := f.extract (nl + 1) f.size
    if body.size != 2 * n * pw then throw "--public-inputs: length"
    let words := (pop.population.map (·.2.2)).getD pop.outputs
    if let some x := (List.range (n * pw)).find? (fun x => words.getD x 0 != (body.get! (2 * x)).toNat + 256 * (body.get! (2 * x + 1)).toNat) then
      let (i, w) := (x / pw, x % pw)
      let starts := (pi.ports.foldl (fun (acc : Array Nat × Nat) p => (acc.1.push acc.2, acc.2 + p.2)) (#[], 0)).1
      let k := ((List.range pi.ports.size).filter (starts[·]! ≤ w)).getLast?.getD 0
      let pj := (pi.json.getArrVal? k).toOption.getD Json.null
      let s := fun f => (pj.getObjValAs? String f).toOption.getD ""
      throw s!"public input {pi.ports[k]!.1}: word {w - starts[k]!} of instance {i} is not the verifier's own recomputation (seed {s "seed"}, derivation {s "derivation"})"

end HmIn

/-! PROTOCOL.md §16.13 Linking: under a template query, with the verifier's own program, every cut wire between two units of
the statement is a check on the public file. A reader whose parameter leaves are another unit's outputs commits them in a link
row (META `links`): an input row encoded as the one output row is, under the writer's salt, so that its `b ‖ c` is the
writer's output row's. This binds a public topology; it does not hide the topology: the wiring is read from the verifier's own
program, so in this model the topology is public, a known leak that later work removes. -/
namespace HmLinks

open HmRow

/-- META `links`: each `{port, leaves: [lo, hi]}`, an input row by name that holds template parameter leaves `[lo, hi)` as
the one output row holds outputs, a `hm96-sha512/row/v1` row of that row's words (neither a bit row nor segmented); a row
once, the ranges disjoint, at least one. None without the key. -/
def parse (c : Circuit) : V (Array (Nat × Nat × Nat)) := do
  let some lj := (c.metaJ.getObjVal? "links").toOption | return #[]
  let xs ← J.asArr lj "META links"
  let k := HmOut.nIn c
  let out := c.ports.getD k default
  if c.outRowPorts.size != 1 || out.v2 || out.segs.isSome then
    throw "META links: a statement with links has one output row, a hm96-sha512/row/v1 row"
  if xs.isEmpty then throw "META links: an empty list (a statement without links has no key)"
  let mut r := #[]
  for x in xs do
    let keys := match x with
      | .obj kvs => (kvs.toArray.map (·.1)).qsort (· < ·)
      | _ => #[]
    if keys != #["leaves", "port"] then throw "META links: a link is exactly {port, leaves}"
    let name ← J.str x "port" "META links"
    let lv ← (← J.arr x "leaves" "META links").mapM (J.asNat · "a link leaf")
    let some p := (List.range k).find? (fun q => (c.ports.getD q default).name == name)
      | throw s!"META links: {name} is not an input row"
    let pt := c.ports.getD p default
    if lv.size != 2 || lv[0]! ≥ lv[1]! then throw s!"META links: {name}'s leaves are not [lo, hi) with lo < hi"
    if pt.v2 || pt.segs.isSome || pt.words != out.words || lv[1]! - lv[0]! > out.words then
      throw s!"META links: {name} is not a hm96-sha512/row/v1 row of the output row's {out.words} words"
    if r.any (fun (q, a, b) => q == p || (a < lv[1]! && lv[0]! < b)) then
      throw s!"META links: {name} is linked twice, or its leaves overlap another link's"
    r := r.push (p, lv[0]!, lv[1]!)
  return r

/-- The template query's population over the program (`TemplateQuery.evaluate`'s order): per unit its root node, member and
gates `[lo, hi)`. -/
def units (P : Program.Prog) : Array (Nat × Nat × Nat × Nat) := Id.run do
  let body := (P.defs[P.root]!).body
  let mut us := #[]
  for k in [0:body.nodes.size] do
    let node := body.nodes[k]!
    let f := P.defs[node.fn]!
    if f.prim && Program.isInputId f.key then continue
    for m in [0:node.members] do
      let lo := body.offsets[k]! + m * f.gates
      us := us.push (k, m, lo, lo + f.gates)
  return us

/-- The unit whose gates hold `g`, by bisection over the units' ascending gate ranges; none for an input gate. -/
def owner (us : Array (Nat × Nat × Nat × Nat)) (g : Nat) : Option Nat := Id.run do
  if us.isEmpty then return none
  let mut lo := 0
  let mut hi := us.size
  while hi - lo > 1 do
    let mid := (lo + hi) / 2
    if us[mid]!.2.2.1 ≤ g then lo := mid else hi := mid
  let (_, _, a, b) := us[lo]!
  return if a ≤ g && g < b then some lo else none

/-- Unit `v`'s output gates, in order. -/
def outGates (P : Program.Prog) (us : Array (Nat × Nat × Nat × Nat)) (v : Nat) : V (Array Nat) := do
  let (k, m, _, _) := us[v]!
  let node := (P.defs[P.root]!).body.nodes[k]!
  let L := (P.defs[node.fn]!).outLeaves
  (Array.range L).mapM fun j => Program.resolve P P.rootScope ⟨false, k, if node.form == .batch then m * L + j else j⟩

/-- Linking, over every unit of the registered population `pop` (before any draw, so a drawn reader is held to its writer's
committed row whether or not the writer is drawn): for each unit and each of its parameter leaves that reads another unit's
output, a link of META `links` holds it; each link's leaves are either all circuit inputs (the row is an ordinary input row)
or exactly one unit's outputs in order, and then that writer is a unit of the file and the reader's link row's `b ‖ c` is
the writer's output row's. The cut wires checked. -/
def check (c : Circuit) (pop : Public) (program : ByteArray) : V Nat := do
  let links ← parse c
  let P ← Program.decode program
  let us := units P
  let np := c.ports.size
  let popIdx ← (← J.arr (← J.get pop.header "units" "public file") "indices" "public file").mapM (J.asNat · "a unit index")
  let inst : Std.HashMap Nat Nat := (Array.range popIdx.size).foldl (fun m i => m.insert popIdx[i]! i) {}
  let mut outs : Std.HashMap Nat (Array Nat) := {}
  let mut wires := 0
  for i in [0:popIdx.size] do
    let u := popIdx[i]!
    let some (k, m, _, _) := us[u]? | throw s!"links: unit {u} is outside the program's {us.size} units"
    let ops ← Program.operands P P.rootScope ((P.defs[P.root]!).body.nodes[k]!) m
    for t in [0:ops.size] do
      if let some v := owner us ops[t]! then
        if !links.any (fun (_, lo, hi) => lo ≤ t && t < hi) then
          throw s!"links: cut wire: unit {u}'s parameter leaf {t} reads unit {v}'s output (gate {ops[t]!}), and no link row (META links) carries it"
    for (p, lo, hi) in links do
      let name := (c.ports.getD p default).name
      if hi > ops.size then throw s!"links: {name}'s leaves [{lo}, {hi}) are past unit {u}'s {ops.size} parameter leaves"
      let rd := ops.extract lo hi
      match owner us rd[0]! with
      | none =>
        if rd.any (owner us · |>.isSome) then
          throw s!"links: unit {u}'s leaves [{lo}, {hi}) (row {name}) read both circuit inputs and a unit's outputs"
      | some v =>
        let og ← match outs[v]? with
          | some og => pure og
          | none => do
            let og ← outGates P us v
            outs := outs.insert v og
            pure og
        if og != rd then
          throw s!"links: unit {u}'s leaves [{lo}, {hi}) (row {name}) are not exactly unit {v}'s {og.size} outputs in order"
        let some w := inst[v]?
          | throw s!"links: unit {u}'s row {name} reads unit {v}, which is not a unit of this public file (a link across statements is not checked)"
        if pop.digest i p np != pop.digest w (np - 1) np then
          throw s!"links: cut wire [{lo}, {hi}) from unit {v} to unit {u}: the reader's row {name} is not the writer's committed output row (their b ‖ c differ)"
        wires := wires + 1
  return wires

end HmLinks

/-- `Stmt.setupH` for a statement with hidden outputs (`Tags.hiddenOutputs`): every check `setupH` makes, on `HmOut`'s rows,
public file, Δ and regions, and on its public input ports (`HmIn`) against the verifier's own `publicInputs`. -/
def Stmt.setupHidden (tags : Tags) (circuitFile publicFile : ByteArray) (coins : Coins) (tables : List (String × ByteArray) := [])
    (draw : Option Json := none) (partition : Option Json := none) (program : Option ByteArray := none)
    (publicInputs : Option ByteArray := none) : V Stmt := do
  let c ← HmOut.parse tags circuitFile tables
  let pi ← HmIn.parse c
  let finding0 ← if tags.bindings then (some <$> HmRow.checkPartition c partition)
    else if partition.isSome then throw s!"{tags.name}: the statement binds no partition" else pure none
  let some merkle := MerkleScheme.find? c.leafScheme | throw s!"leaf-commitment scheme {c.leafScheme}: not one this verifier implements"
  let pop ← HmIn.loadPublic tags c pi publicFile
  HmIn.checkOwn pi pop publicInputs
  let pub ← match draw with
    | some d => HmIn.drawn c pi pop d
    | none => pure pop
  if let some d := draw then
    if let (Except.ok (Draw.Law.stratified k strata), some obj, some prog) :=
        ((Draw.UnitDraw.ofJson d).map (·.law), partition, program) then
      let (law, _) ← HmRow.stratifiedLaw obj prog k
      if law != Draw.Law.stratified k strata then throw "U2: the draw's strata are not the verifier's own (its program and query)"
  HmOut.checkPublic c pub
  let linked := J.has c.metaJ "links"
  let finding ← match finding0, partition, program with
    | some f, some obj, some prog =>
      let q := (obj.getObjVal? "query").toOption.getD Json.null
      if (TemplateQuery.templatesOf ((q.getObjValAs? String "name").toOption.getD "") ((q.getObjValAs? Nat "version").toOption.getD 0) Json.null).isSome then
        let indices ← (← J.arr (← J.get pub.header "units" "public file") "indices" "public file").mapM (J.asNat · "a unit index")
        let note ← HmRow.deriveTemplateUnits c obj prog indices
        let wires ← HmLinks.check c pop prog
        let linking := if wires == 0 && !linked then "; no cut wire between the statement's units"
          else s!"; Linking checked: {wires} cut wire(s) over the {pop.n} units of the population, each reader's link row the writer's committed output row"
        pure (some { f with derived := true, note := note ++ linking })
      else if linked then throw "META links: Linking is checked under a template query (Q_template_instance v0, Q_template_instances v0) only"
      else HmRow.qwordFinding c pub obj prog f
    | f, _, _ =>
      if linked then throw "META links: Linking needs the verifier's own program and partition (--program, --partition)"
      pure f
  let n := pub.n
  let blocksReal := (n + c.g - 1) / c.g
  let nbl := max 3 (log2ceil (max blocksReal 1))
  let m := c.kLog + nbl
  if c.kLog > 27 || m > 35 then throw s!"m = {m}: no fast100 schedule (or k_log > 27)"
  let pin ← HmRow.pin c
  let (da, db) := HmIn.delta c pi pin
  let regions := (← HmOut.regions c pub) ++ (← HmIn.regions c pi pub)
  checkInRange c.kLog regions.size (PT_LOCAL + nbl)
  let identity := if J.has c.metaJ "row_schemas" then Tags.rowV2Identity tags.identity else tags.identity
  let identity := if c.ports.any (·.segs.isSome) then Tags.rowSegIdentity identity else identity
  let v2Run := (List.range c.outRowPorts.size).any fun j =>
    (c.ports.getD (c.ports.size - c.outRowPorts.size + j) default).segs.isNone && c.outRowPorts[j]! > 1
  let identity := if !J.has c.metaJ "out_row_ports" then identity
    else if v2Run then Tags.outRowsV2Identity identity else Tags.outRowsIdentity identity
  let mut h := tags.statement.toUTF8 ++ c.sha ++ (← canon identity).toUTF8
  for x in [n, blocksReal, nbl, m, c.kLog, pin, regions.size, c.g] do h := pushLe64 h x
  for d in [da, db] do
    h := pushLe64 h d.size
    for (i, cc) in d do h := pushLe32 (pushLe32 h i) cc
  let digest := Sha512.hash h
  let sigma := Sha512.hash (tags.sigmaTag.toUTF8 ++ c.sha ++ digest ++ pub.sha)
  let hello ← helloOf tags sigma (PT_LOCAL + nbl) coins
  return { tags, leaf := default, c, pub, n, blocksReal, nbl, m, mPts := PT_LOCAL + nbl, pin, regions, da, db, digest, sigma,
           hello, merkle, partition := (finding.map (·.toJson)).getD Json.null }

/-- A session's statement in `count` tables (§16.12, upstream's `session_tables`): one table is `setupH`'s statement. For
several, the session's units (the draw's, without one the population's) split into `count` equal consecutive parts, and
table `j` is `setupH`'s statement of part `j` as a subset draw of the population, under `Tags.tableIdentity`: every check
`setupH` makes, it makes on each table. Every part comes from `draw`, the session's, never from a table's header. The
session's Σ covers every table's; its `unitDraw` is `draw`. `work` is the verifier's own `K` and work table, which a work
draw's law is held to, `stratified` its own `K`, at which a stratified draw's law is its own derivation, `closure` its
own closure map, which a work draw's closure is held to, and `publicInputs` its own recomputation of the public input
words (`HmIn.checkOwn`). -/
def Stmt.setupTables (tags : Tags) (circuitFile publicFile : ByteArray) (coins : Coins) (count : Nat)
    (tables : List (String × ByteArray) := []) (draw : Option Json := none) (partition : Option Json := none)
    (program : Option ByteArray := none) (work : Option (Nat × Array (String × Nat × Nat)) := none)
    (stratified : Option Nat := none) (closure : Option (Std.HashMap Nat (Array Nat)) := none)
    (publicInputs : Option ByteArray := none) : V (SessionSpec × Array Stmt) := do
  -- a closure draw proves more units than it draws, so what it adds must be the verifier's own closure of the draw
  if let some d := draw then
    if let .ok u := Draw.UnitDraw.ofJson d then
      if let .work .. := u.law then
        match closure with
        | some cm =>
          if u.closure != Draw.closureOf cm u.units then
            throw "U3: the draw's closure is not the verifier's own (its closure map)"
        | none =>
          if !u.closure.isEmpty then throw "U3: a closure draw needs the verifier's own closure map (--closure)"
  -- a stratified draw's K and strata set every k_s, so its law must be the verifier's own derivation, never taken as stated
  if let some d := draw then
    if let .ok (Draw.Law.stratified k strata) := (Draw.UnitDraw.ofJson d).map (·.law) then
      let (some obj, some prog, some K) := (partition, program, stratified)
        | throw "U2: a stratified draw needs the verifier's own K, program and partition (--stratified, --program, --partition)"
      if k != K then throw s!"U2: the draw's K is {k}, not the verifier's {K}"
      let (law, _) ← HmRow.stratifiedLaw obj prog K
      if law != Draw.Law.stratified k strata then
        throw "U2: the draw's strata are not the verifier's own (its K, program and query)"
  -- a work draw's K and stated work set every k_s, so its law must be the verifier's own derivation, never taken as stated
  if let some d := draw then
    if let .ok (Draw.Law.work k strata) := (Draw.UnitDraw.ofJson d).map (·.law) then
      let (some obj, some prog, some (K, table)) := (partition, program, work)
        | throw "U2: a work draw needs the verifier's own K, work table, program and partition (--work, --work-table, --program, --partition)"
      if k != K then throw s!"U2: the draw's K is {k}, not the verifier's {K}"
      let (law, _) ← HmRow.workLaw obj prog K table
      if law != Draw.Law.work k strata then
        throw "U2: the draw's work law is not the verifier's own (its K, program, query and work table)"
  if publicInputs.isSome && !tags.hiddenOutputs then
    throw s!"--public-inputs: a {tags.name} statement has no public input ports (only a statement with hidden outputs has them)"
  let setup := fun (t : Tags) (d : Option Json) => do
    if tags.hiddenOutputs then Stmt.setupHidden t circuitFile publicFile coins tables d partition program publicInputs
    else
      let st ← Stmt.setupH t circuitFile publicFile coins tables d partition program
      if J.has st.c.metaJ "public_inputs" then
        throw s!"META: public_inputs: a {tags.name} statement has no public input ports (only a statement with hidden outputs has them)"
      pure st
  if count ≤ 1 then
    let st ← setup tags draw
    return (st.spec, #[st])
  let some tt := tags.manyTables | throw s!"{tags.name}: one table per session"
  if count > Tags.MAX_TABLES then throw s!"tables: at most {Tags.MAX_TABLES} per session, not {count}"
  let splits := fun (n : Nat) => if n == 0 || n % count != 0 then throw s!"tables: {n} units do not split into {count} equal tables"
    else (pure () : V Unit)
  -- the draw's law and count are refused before the circuit is read
  let u? ← draw.mapM Draw.UnitDraw.ofJson
  if let some u := u? then
    match u.law with
    | .subset _ => splits u.units.size
    | _ => throw "tables: several tables need a subset draw (a Bernoulli draw's unit count is random, so the tables could not split it evenly)"
  -- without a draw, the population's units (`setupH` checks a draw's population against the file)
  let (population, units) ← match u? with
    | some u => pure (u.population, u.units)
    | none => do
      let n ← if tags.hiddenOutputs then do
          let c ← HmOut.parse tags circuitFile tables
          pure (← HmIn.loadPublic tags c (← HmIn.parse c) publicFile).n
        else do pure (← HmRow.loadPublic tags (← HmRow.parse tags circuitFile tables) publicFile).n
      pure (n, Array.range n)
  splits units.size
  let k := units.size / count
  let sts ← (Array.range count).mapM fun j =>
    let part := Json.mkObj [("law", Json.str "subset"), ("population", Lean.toJson population), ("k", Lean.toJson k),
      ("units", Json.arr ((units.extract (j * k) (j * k + k)).map Lean.toJson))]
    setup { tags with identity := tags.tableIdentity count j } (some part)
  let mPts := sts[0]!.mPts
  if sts.any (·.mPts != mPts) then throw "tables: the tables' link points have different lengths"
  let sigma := Sha512.hash (sts.foldl (fun h st => h ++ st.sigma) (pushLe64 tt.sigmaTag.toUTF8 count))
  let names := tags.tableNames count
  let hello ← helloOf tags sigma mPts coins names
  return ({ tables := names, domain := tags.tableDomain count, sigma, hello, mPts, retainRounds := tags.retainRounds,
            unitDraw := draw, instances := units.size, sha512Digests := tags.hm96Rows, coinTree := CoinTree.ofHello hello }, sts)

end Flock
