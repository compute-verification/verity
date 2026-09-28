/-
PROTOCOL.md §16.10: `verity/flock-circuit` from PR #83 `eb90718f`. Serving rows are `hm96-sha512/row/v1` leaves proved in
the circuit: per VU, each input port's row is hashed by SHA-512 compressions in `sha512x3` slots (three per slot, chained from
the prefix's midstate) and closed by an `hm96` slot (the padding compression, then `b ‖ c` from the row's private salt). Both
slot circuits are sections of the circuit file. Every identity and binding digest is SHA-512: the circuit pin, the public
digest, the statement digest (Flock's own 32-byte slot is its first half), Σ and the record's digests.

The verifier evaluates no part of the computation: it recomputes the rows' constants, the wiring and Δ from the pinned
circuit, the frame-v3-sha512 roots from the public `b ‖ c` and outputs, and the pinned dummy `b ‖ c` from the zero row.
-/
import Flock.Statement
import Flock.TemplateQuery

namespace Flock

open Lean (Json)

namespace HmRow

def SHA_NET : String := "sha512x3"
def HM_NET : String := "hm96"
def SCHEME : String := "frame-v3-sha512/hm96-sha512"
def ROW_SCHEMA : String := "hm96-sha512/row/v1"
def PER_SLOT : Nat := 3
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
    for k in [1:w.size] do
      out := out.push ⟨(sha, w[k]!.1, 2 * w[k]!.2), (sha, w[k - 1]!.1, w[k - 1]!.2)⟩
    let last := w[w.size - 1]!
    out := out.push ⟨(hm, p, 0), (sha, last.1, last.2)⟩
  return out

/-- The zero row's `b ‖ c` under a zero salt: `b = x` (a zero salt's mask is zero), `c = SHA-512(salt_prefix ‖ 0^192)`. -/
def zeroRowBc (words : Nat) : ByteArray :=
  Sha512.hash (rowPrefix words ++ ByteArray.mk (Array.replicate (2 * words) 0)) ++
    Sha512.hash (Hm96.Default512.saltPrefix ++ ByteArray.mk (Array.replicate SALT_BYTES 0))

private def ctx := "circuit META"

private def intLists (j : Json) (k : String) : V (Array (Array Int)) := do
  (← J.arr j k ctx).mapM fun u => do (← J.asArr u s!"{ctx}.{k}").mapM fun x => J.asInt x s!"{ctx}.{k}"

/-- Upstream's `Composite::check`: the layout's shape, the row slots' ports and counts, every non-row input port wired
exactly once, the leaf maps, the pinned dummy values. -/
def check (c : Circuit) (tags : Tags) (leafScheme : Json) : V Unit := do
  let mut spans : Array (Nat × Nat) := #[]
  for r in c.ranges do
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
  let sorted := spans.qsort (fun a b => a.1 < b.1 || (a.1 == b.1 && a.2 < b.2))
  for i in [1:sorted.size] do
    if sorted[i - 1]!.2 > sorted[i]!.1 then throw "two ranges overlap"
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
  if c.dummyDigests.size != c.ports.size || c.dummyOutputs.size != nOut then
    throw "the pinned dummy values are not one per port and output leaf"
  if let some pinned := tags.pinnedLeafScheme then
    if (← canon leafScheme) != (← canon pinned) then throw "META: leaf_scheme is not the pinned leaf-commitment scheme"

/-- Parse the circuit file (upstream's `Composite::parse`) and check its layout. -/
def parse (tags : Tags) (bytes : ByteArray) (tables : List (String × ByteArray) := []) : V Circuit := do
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
  let mut nets : Array (String × Net) := #[]
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
    if 2 ^ n != size || !(2 ≤ lo && lo < n) then throw s!"META: lookup {lname}'s index is not its table's"
    let some (_, words) := tables.find? (·.1 == lname) | throw s!"the circuit has a lookup of {lname}: its table is not given"
    if words.size != 4 * size || hexOf (Sha512.hash words) != pinned.sha512 then throw s!"MUFU table {lname}: not the pinned table"
    nets := nets.push (s!"lookup:{lname}", ← Lookup.build s!"lookup:{lname}" words n lo 31)
  if (← J.str mj "scheme" ctx) != SCHEME || (← J.str mj "row_schema" ctx) != ROW_SCHEMA then
    throw s!"META: rows are not {SCHEME} ({ROW_SCHEMA}) leaves"
  let ports ← (← J.arr mj "ports" ctx).mapM fun p => do
    let a ← J.asArr p s!"{ctx}: a port"
    if a.size != 3 then throw s!"{ctx}: a port is not [name, words, comps]"
    pure ({ name := ← J.asStr a[0]! "port name", words := ← J.asNat a[1]! "port words", chunks := ← J.asNat a[2]! "port comps",
            nb := 0 } : Port)
  for p in ports do
    if p.words == 0 || (2 * p.words) % 128 != 0 || p.chunks != 2 * p.words / 128 then
      throw s!"port {p.name}: not whole SHA-512 blocks"
  -- each row's constants, recomputed from the scheme (never taken from META)
  let listed ← J.arr mj "rows" ctx
  if listed.size != ports.size then throw "META: a row's constants are not sha512/row/v1's"
  for i in [0:ports.size] do
    let w := ports[i]!.words
    let m := listed[i]!
    let ms ← (← J.arr m "midstate" ctx).mapM (J.asStr · "midstate")
    if (← J.str m "prefix" ctx) != hexOf (rowPrefix w) || ms != (midstate w).map hex16 || (← J.str m "pad" ctx) != hexOf (bytesBE (padWords w)) then
      throw "META: a row's constants are not sha512/row/v1's"
  if (← J.str mj "row_key_sha512" ctx) != hexOf (Sha512.hash Hm96.Default512.key) then
    throw "META: the rows' hm96 key is not hm96-sha512/v1's default key"
  let g ← J.nat mj "vus_per_block" ctx
  let upv ← J.nat mj "units_per_vu" ctx
  let some unitNet := nets.findIdx? (·.1 == "unit") | throw "circuit: no unit circuit"
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
  let leavesIn ← intLists mj "leaves_in"
  let leavesOut ← (← intLists mj "leaves_out").mapM fun u => u.mapM fun x =>
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
  let mut perVu := (Array.replicate nets.size 0).set! unitNet upv
  match mj.getObjVal? "per_vu" with
  | .ok (Json.obj kvs) =>
    for (k, v) in kvs.toArray do
      perVu := perVu.set! (← netOf k) (← J.asNat v "per_vu")
  | _ => pure ()
  let shaNet ← netOf SHA_NET
  let hmNet ← netOf HM_NET
  let outNet ← netOf ((mj.getObjValAs? String "out_net").toOption.getD "unit")
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
  -- the rows' wires come first, exactly as the ports imply
  let derived := rowWires ports shaNet hmNet
  let rowNet := fun (n : Nat) => n == shaNet || n == hmNet
  if wires.size < derived.size || (List.range derived.size).any (fun i => wires[i]!.dst != derived[i]!.dst || wires[i]!.src != derived[i]!.src)
      || (wires.extract derived.size wires.size).any (fun w => rowNet w.dst.1 || rowNet w.src.1) then
    throw "META: the rows' wires are not the chains and hm96 inputs the ports imply"
  let leafScheme ← J.get mj "leaf_scheme" ctx
  let leafId ← J.str leafScheme "id" ctx
  if leafId != tags.merkleLeaf then throw s!"leaf-commitment scheme {leafId} is not one this verifier implements"
  let c : Circuit := { name, sha, nets, ranges, ports, outPorts, g, kLog := ← J.nat mj "k_log" ctx, upv, unitNet,
                       leavesIn, leavesOut, key := ByteArray.empty, dummyDigests, dummyOutputs, leafScheme := leafId, perVu, wires,
                       shaNet, hmNet, classSha, metaJ := mj, outNet, outNetPorts, leafCuts }
  check c tags leafScheme
  return c

/-- The pinned constant column: the first slot of the first range that is not the mask. -/
def pin (c : Circuit) : V Nat :=
  match c.ranges.find? (·.kind != Kind.mask) with
  | some r => match r.kind with
    | .net i => pure (r.start + (c.nets.getD i default).2.constPos)
    | _ => throw "no pin"
  | none => throw "circuit: no range with a constant"

/-- Δ (A and B, upstream's `delta`): every slot's constant from the pin; the rows' constants (each row's first `cv` the
prefix's midstate, its hm96 slot's `pad` the padding block: bit 1 from the pin, bit 0 from the unit range's first slot's
forced-zero row); every unit's leaf bits copied from the message bits that hash them (a zero leaf from its own forced-zero
row); every wire, the rows' chains first. -/
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
  for g in [0:c.g] do
    for p in [0:where_.size] do
      let w := where_[p]!
      let ms := midstate c.ports[p]!.words
      let pw := padWords c.ports[p]!.words
      let cv := c.slotCol c.shaNet g w[0]!.1 + sin[2 * w[0]!.2]!.1
      for t in [0:512] do
        let src := if (ms[t / 64]! >>> (t % 64).toUInt64) &&& 1 == 1 then pin else zero
        d := d.push (cv + t, cv + t) |>.push (cv + t, src)
      let pad := c.slotCol c.hmNet g p + hin[1]!.1
      for t in [0:1024] do
        let src := if (pw[t / 64]! >>> (t % 64).toUInt64) &&& 1 == 1 then pin else zero
        d := d.push (pad + t, pad + t) |>.push (pad + t, src)
  let msgCol := fun (g p byte t : Nat) =>
    let (k, o) := (byte / 128, byte % 128)
    let (slot, sub) := where_[p]![k]!
    c.slotCol c.shaNet g slot + sin[2 * sub + 1]!.1 + 64 * (o / 8) + 56 - 8 * (o % 8) + t
  let leafCols := (portCols #[unit.inGroups.getD 0 default]).map (·.1)
  let starts := (c.ports.foldl (fun (acc : Array Nat × Nat) p => (acc.1.push acc.2, acc.2 + p.words)) (#[], 0)).1
  for k in [0:ur.count] do
    let (g, u) := (k / c.upv, k % c.upv)
    let base := ur.slot k
    let li := c.leavesIn.getD u #[]
    for j in [0:li.size] do
      let lf := li[j]!
      for t in [0:16] do
        let src := if lf < 0 then base + unit.useful else
          let x := lf.toNat
          let p := ((List.range starts.size).reverse.find? (fun q => starts[q]! ≤ x)).getD 0
          msgCol g p (2 * (x - starts[p]!) + t / 8) (t % 8)
        let i := base + leafCols.getD j 0 + t
        d := d.push (i, i) |>.push (i, src)
  let pc := portCols unit.inGroups
  for g in [0:c.g] do
    for (u, j, leaf) in c.leafCuts do
      let p := ((List.range starts.size).reverse.find? (fun q => starts[q]! ≤ leaf)).getD 0
      let dst := c.slotCol c.unitNet g u + pc[j]!.1
      for t in [0:pc[j]!.2] do
        let src := if t < 16 then msgCol g p (2 * (leaf - starts[p]!) + t / 8) (t % 8) else zero
        d := d.push (dst + t, dst + t) |>.push (dst + t, src)
  for g in [0:c.g] do
    for w in c.wires do
      let dc := (portCols (c.nets.getD w.dst.1 default).2.inGroups).getD w.dst.2.2 default
      let sc := (portCols (c.nets.getD w.src.1 default).2.outGroups).getD w.src.2.2 default
      let dst := c.slotCol w.dst.1 g w.dst.2.1 + dc.1
      let src := c.slotCol w.src.1 g w.src.2.1 + sc.1
      for t in [0:dc.2] do d := d.push (dst + t, dst + t) |>.push (dst + t, src + t)
  return (d, d)

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
def EVALUABLE : List (String × Nat) := [("Q_word", 1)]

/-- A `verity/partition/v1` object's form (#111's canonical form, open query names): exactly `format`, `program`, `query`; a
SHA-512 program; a query of exactly `name` (a non-empty string), `version` (a natural number) and `params` (an object). For a
query the verifier evaluates, its parameters too (`Q_word` v1: exactly `X`, `W`, positive integers). The query's name and
version. -/
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
  if (name, version) == ("Q_word", 1) then
    if keys ps != #["W", "X"] then throw "partition: Q_word v1's parameters are not X, W"
    for k in ["X", "W"] do
      match (ps.getObjVal? k).toOption with
      | some (.num n) => if !(n.exponent == 0 && n.mantissa > 0) then throw s!"partition: {k} is not a positive integer"
      | _ => throw s!"partition: {k} is not a positive integer"
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

/-- A template query's units, derived from the verifier's own program: the program is the object's, the query applies to it,
the circuit's template is one it lists, and every unit the statement claims (`indices`) is an instance of that template. -/
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
  let some t := circuitTemplate c | throw "partition: META program_digests does not name the circuit's one template"
  if !ts.contains t then throw s!"partition: the circuit's template {t} is not one the query lists"
  let pop := TemplateQuery.population rs
  for i in indices do
    match TemplateQuery.locate rs i with
    | none => throw s!"partition: unit {i} is outside the partition's {pop} units"
    | some (_, _, ti) => if ti != t then throw s!"partition: unit {i} is an instance of {ti}, not the circuit's template {t}"
  return s!"units derived: {name} v{version} over the verifier's program ({pop} units); the statement's {indices.size} units are instances of {t}"

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
      else if EVALUABLE.contains (name, version) then
        s!"units taken as stated, not derived: {name} v{version} is a query the verifier evaluates (Flock.Qword), but it does not yet decode the program into its Call graphs"
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
  if d.population != pub.n || pub.population.isSome then throw "U: the draw's population is not this registered population"
  let np := c.ports.size
  let nOut := c.outWords
  let pctx := "public file"
  let popIdx ← (← J.arr (← J.get pub.header "units" pctx) "indices" pctx).mapM (J.asNat · "a unit index")
  let indices := d.units.map fun u => popIdx.getD u 0
  let header := ((pub.header.setObjVal! "unit_draw" draw).setObjVal! "instances" (Lean.toJson d.units.size)
    |>.setObjVal! "population" (Lean.toJson pub.n)).setObjVal! "units" (unitsEntry c indices)
  let mut pubs := pub.digests.foldl (· ++ ·) ByteArray.empty
  if let some t := pub.tables then
    for r in t.refs do pubs := pubs ++ le32 r.toUInt32
  for w in pub.outputs do pubs := pubs.push (w % 256).toUInt8 |>.push (w / 256).toUInt8
  return { header, n := d.units.size,
           digests := if pub.tables.isSome then pub.digests
             else d.units.foldl (fun acc u => acc ++ pub.digests.extract (u * np) (u * np + np)) #[],
           outputs := d.units.foldl (fun acc u => acc ++ pub.outputs.extract (u * nOut) (u * nOut + nOut)) #[],
           sha := ← publicDigest header pubs, population := some (pub.n, pub.digests, pub.outputs),
           tables := pub.tables.map fun t => { t with refs := d.units.foldl (fun acc u => acc ++ t.refs.extract (u * np) (u * np + np)) #[] } }

/-- `stratified:K` expanded over the verifier's own program and partition object (`draw.derive`), and the population: the
query applied to the program, its units grouped by template descriptor id. Only `Q_template_instance` v0 and
`Q_template_instances` v0 define strata. -/
def stratifiedLaw (obj : Json) (program : ByteArray) (k : Nat) : V (Draw.Law × Nat) := do
  let (name, version) ← checkPartitionObject obj
  let q := (obj.getObjVal? "query").toOption.getD Json.null
  let some ts := TemplateQuery.templatesOf name version ((q.getObjVal? "params").toOption.getD Json.null)
    | throw s!"stratified: {name} v{version} defines no strata (Q_template_instance v0 and Q_template_instances v0 do)"
  let ts ← ts
  let d ← TemplateQuery.Desc.parse program
  if (obj.getObjValAs? String "program").toOption != some (hexOf d.sha512) then
    throw "partition: the verifier's program is not the one the partition object names"
  let rs ← match TemplateQuery.evaluate d ts with
    | .ok rs => pure rs
    | .error e => throw s!"partition: {e}"
  match Draw.derive k (Draw.strataOf (rs.map fun r => (r.template, r.first, r.count))) with
  | .ok law => return (law, TemplateQuery.population rs)
  | .error e => throw s!"stratified: {e}"

/-- The frame-v3-sha512 roots recomputed from the population's `b ‖ c` (a shared-row file: each input port's over its table)
and output words against the header's; the pinned dummy `b ‖ c` against the zero row's (upstream's `check_public`). -/
def checkPublic (c : Circuit) (pub : Public) : V Unit := do
  let fv ← pub.frameV3
  if (← J.str fv "frame" "public file") != "frame-v3-sha512"
      || (← J.str fv "row_key_sha512" "public file") != hexOf (Sha512.hash Hm96.Default512.key) then
    throw "the instance file's rows are not frame-v3-sha512 hm96 leaves under the default key"
  let np := c.ports.size
  let (n, bcs, outputs) := pub.population.getD (pub.n, pub.digests, pub.outputs)
  if let .ok d := pub.header.getObjVal? "unit_draw" then
    let u ← Draw.UnitDraw.ofJson d
    if (← J.nat pub.header "instances" "public file") != u.units.size || pub.n != u.units.size then
      throw "U3: the statement's instances are not the drawn units"
  let H := Sha512.hash
  for p in [0:np] do
    let port := c.ports[p]!
    let dom ← pub.fv3Bytes "domain_ids" port.name
    if (← J.str (← J.get fv "schemas" "public file") port.name "public file") != ROW_SCHEMA then
      throw s!"input port {port.name}: schema is not {ROW_SCHEMA}"
    let (count, row) : Nat × (Nat → Nat) := match pub.tables with
      | some t => (t.counts[p]!, fun r => t.starts[p]! + r)
      | none => (n, fun i => i * np + p)
    let leaves := (Array.range count).map fun r => FV3.leaf dom r ROW_SCHEMA (Hm96.Default512.treeLeaf (bcs.getD (row r) ByteArray.empty)) H
    if FV3.root dom leaves H != (← pub.fv3Bytes "roots" port.name) then
      throw s!"input port {port.name}: the rows' b || c do not hash to the header's frame-v3-sha512 root"
    if zeroRowBc port.words != c.dummyDigests.getD p ByteArray.empty then
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
  if c.outNet != c.unitNet then
    -- the last stage's output group is the output words (one slot per VU)
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
  let c ← HmRow.parse tags circuitFile tables
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
      else pure (some f)
    | f, _, _ => pure f
  let n := pub.n
  let blocksReal := (n + c.g - 1) / c.g
  let nbl := max 3 (log2ceil (max blocksReal 1))
  let m := c.kLog + nbl
  if c.kLog > 26 || m > 35 then throw s!"m = {m}: no fast100 schedule (or k_log > 26)"
  let pin ← HmRow.pin c
  let (da, db) := HmRow.delta c pin
  let regions ← HmRow.regions c pub
  HmRow.checkRegionWords c da db regions
  let mut h := tags.statement.toUTF8 ++ c.sha ++ (← canon tags.identity).toUTF8
  for x in [n, blocksReal, nbl, m, c.kLog, pin, regions.size, c.g] do h := pushLe64 h x
  for d in [da, db] do
    h := pushLe64 h d.size
    for (i, cc) in d do h := pushLe32 (pushLe32 h i) cc
  let digest := Sha512.hash h
  let sigma := Sha512.hash (tags.sigmaTag.toUTF8 ++ c.sha ++ digest ++ pub.sha)
  let hello ← helloOf tags sigma (PT_LOCAL + nbl) coins
  return { tags, leaf := default, c, pub, n, blocksReal, nbl, m, mPts := PT_LOCAL + nbl, pin, regions, da, db, digest, sigma,
           hello, merkle, partition := (finding.map (·.toJson)).getD Json.null }

end Flock
