/-
PROTOCOL.md §16.5: the circuit file: line 1 is `<format> <name>`, line 2 is `META <json>` (the layout), then
`<keyword> <name> <lines>` blocks, each an expanded circuit in `flock-ir-unit/v2` form (format and keyword are the
statement's `Tags`). The parser refuses every part the verifier would have to evaluate (a `CUT` line; the META keys `tail`,
`cut`, `cut_words`, `public_ports`, `public`, `native`, `outputs_by_tail`), and checks the layout's generic shape; the row
leaf checks its own ranges (`RowLeaf.checkLayout`). Lookup slots are built from their pinned tables (`Flock.Lookup`).
-/
import Lean.Data.Json
import Flock.Lookup
import Flock.TmplAt
import Flock.Tags
import Flock.Typed

namespace Flock

open Lean (Json)


structure Port where
  name : String
  words : Nat
  chunks : Nat
  nb : Nat
  /-- the row's length in bits: a `sha512/row/v1` row's words' -/
  bits : Nat := 16 * words
  /-- a bit row (`sha512/row/v2`, its leaf `hm96-sha512/row/v2`; or a segmented one, `segs`) -/
  v2 : Bool := false
  /-- a segmented bit row (`sha512/row-seg/v1`, its leaf `hm96-sha512/row-seg/v1`): each segment's bits and compressions -/
  segs : Option (Array (Nat × Nat)) := none
  deriving Inhabited, Repr

/-- The row bytes its digest hashes after the prefix: `2 * words`, a bit row's `ceil(bits / 8)`. -/
def Port.bytes (p : Port) : Nat := if p.v2 then (p.bits + 7) / 8 else 2 * p.words

inductive Kind where
  | comp
  | net (i : Nat)
  | mask
  deriving BEq, DecidableEq, Inhabited, Repr

inductive Role where
  /-- chunk runs of input port `p`: run slot `q` holds `(vu, chunk)` or nothing (a padding run) -/
  | runs (p : Nat) (slots : Array (Option (Nat × Nat)))
  /-- port `p`'s parent-tree compressions: per template instance `chunks` slots, heap order (root first), the last a
padding slot -/
  | parents (p : Nat)
  | none
  deriving Inhabited, Repr

structure Range where
  kind : Kind
  slotLog : Nat
  pos0 : Nat
  count : Nat
  role : Role
  /-- aligned to its slot only, spanning its count (the circuit prover's `sha512x3` range); else aligned to its power-of-two span -/
  packed : Bool := false
  deriving Inhabited, Repr

def Range.start (r : Range) : Nat := r.pos0 * 2 ^ r.slotLog
def Range.slot (r : Range) (s : Nat) : Nat := (r.pos0 + s) * 2 ^ r.slotLog

/-- A copy of every bit of output port `src.2.2` of slot `(src.1, src.2.1)` into input port `dst.2.2` of slot
`(dst.1, dst.2.1)`, both per template instance. -/
structure Wire where
  dst : Nat × Nat × Nat
  src : Nat × Nat × Nat
  deriving Inhabited, Repr

/-- A typed template's block (item 1e; the typed template's `TypedBlock`): the instance type's root is the unit, one slot per template instance, beside
its parts, each in a slot of its layout's range. Unit coordinates: the root's own region `[0, own)`, then each part from its
base. -/
structure TypedBlock where
  own : Nat
  /-- the instance type's ports, (count, width) -/
  ins : Array (Nat × Nat)
  outs : Array (Nat × Nat)
  /-- the root's input columns, in input-bit order -/
  inCols : Array Nat
  /-- per part, in item order: its layout's circuit, its slot among that circuit's slots in the template instance, its
base -/
  parts : Array (Nat × Nat × Nat)
  /-- per placed layout's circuit, its range's log size -/
  rangeLogs : Array (Nat × Nat)
  /-- the parts' bound input rows and their sources (Δ), in item order and bits ascending -/
  bind : Array (Nat × Array Nat)
  /-- the root rows' entries at its parts' columns: (row, A's, B's) -/
  cross : Array (Nat × Array Nat × Array Nat)
  deriving Inhabited, Repr

/-- The pinned circuit file. -/
structure Circuit where
  name : String
  sha : ByteArray
  nets : Array (String × Net)
  ranges : Array Range
  ports : Array Port
  outPorts : Array (String × Nat)
  /-- each output row's run of output ports, in port order (META `out_row_ports`; absent: one each) -/
  outRowPorts : Array Nat := #[]
  g : Nat
  kLog : Nat
  upv : Nat
  unitNet : Nat
  leavesIn : Array (Array Int)
  leavesOut : Array (Array Nat)
  key : ByteArray
  dummyDigests : Array ByteArray
  dummyOutputs : Array Nat
  leafScheme : String
  perVu : Array Nat
  wires : Array Wire
  /-- the `sha512x3` and `hm96` circuits of an `hm96-sha512/row/v1` statement (`Flock.HmRow`) -/
  shaNet : Nat := 0
  hmNet : Nat := 0
  /-- the class of every instance (the circuit prover from `e51e2b86`): SHA-512 of the file without META's bindings; empty before -/
  classSha : ByteArray := ByteArray.empty
  /-- META, for its bindings (`program_sha512`, `partition`) -/
  metaJ : Json := Json.null
  /-- the circuit whose output group holds the returned outputs: the unit, or (a tail that returns them) its last stage -/
  outNet : Nat := 0
  /-- `outNet`'s output ports that are the output words (a stage), in output-leaf order (META `out_net_ports`) -/
  outNetPorts : Array Nat := #[]
  /-- each output word's slot of `outNet`, where META `out_net_ports` lists `[slot, port]` pairs (a repeated last stage);
  `none` where it lists ports alone, each of slot 0 -/
  outNetSlots : Option (Array Nat) := none
  /-- the unit inputs that are row words (`(unit, input port, leaf)`): the low 16 bits are the word's message bits -/
  leafCuts : Array (Nat × Nat × Nat) := #[]
  /-- a typed template's block (item 1e): the unit is its instance's root, beside its parts -/
  typed : Option TypedBlock := none
  /-- a typed statement's unit class (item 1e): its derivation's inputs and `deriveChecked`'s run on them -/
  cls : Option Typed.Class := none
  /-- a templated unit: its template, layouts and committed net (`Tmpl.unitNet`, its rows built only when read),
  which `Stmt.partialRangesT` folds from the template at the net's range -/
  tmpl : Option TmplUnit := none
  deriving Inhabited

namespace Circuit

def unit (c : Circuit) : Net := (c.nets.getD c.unitNet default).2
def outWords (c : Circuit) : Nat := c.outPorts.foldl (· + ·.2) 0
def inWords (c : Circuit) : Nat := c.ports.foldl (· + ·.words) 0
def rowBytes (c : Circuit) : Nat := c.ports.foldl (· + 2 * ·.words) 0

def rangeOf (c : Circuit) (n : Nat) : Range := (c.ranges.find? (·.kind == Kind.net n)).getD default
def unitRange (c : Circuit) : Range := c.rangeOf c.unitNet
def slotCol (c : Circuit) (n g i : Nat) : Nat := (c.rangeOf n).slot (g * c.perVu.getD n 0 + i)

private def ctx := "circuit META"

private def natPair (x : Json) : V (Option (Nat × Nat)) := do
  match x with
  | Json.null => pure none
  | _ =>
    let a ← J.asArr x s!"{ctx}: a run slot"
    if a.size != 2 then throw s!"{ctx}: a run slot is not [vu, chunk]"
    pure (some (← J.asNat a[0]! s!"{ctx}: run slot", ← J.asNat a[1]! s!"{ctx}: run slot"))

private def intLists (j : Json) (k : String) : V (Array (Array Int)) := do
  (← J.arr j k ctx).mapM fun u => do (← J.asArr u s!"{ctx}.{k}").mapM fun x => J.asInt x s!"{ctx}.{k}"

/-- The layout's generic shape (upstream's `Composite::check` without the row leaf's part): aligned, disjoint ranges inside
the block; roles only on the row leaf's `comp` ranges; one unit range of `g × units_per_vu` units whose leaf maps are the
unit circuit's ports; every non-leaf input port wired exactly once. -/
def checkLayout (c : Circuit) : V Unit := do
  let mut spans : Array (Nat × Nat) := #[]
  for r in c.ranges do
    let span := nextPow2 r.count * 2 ^ r.slotLog
    if r.start % span != 0 || r.start + span > 2 ^ c.kLog then throw "a range is not aligned to its span or overruns the block"
    if r.slotLog < 7 then throw "a range's slots are smaller than a word"
    match r.kind with
    | .net i => if (c.nets.getD i default).2.unitLog != r.slotLog then throw "a unit range's slot size is not its circuit's"
    | _ => pure ()
    let isRole := match r.role with | .none => false | _ => true
    if (r.kind == Kind.comp) != isRole || (r.kind == Kind.mask && r.count != 1) then
      throw "a range's type and role disagree"
    spans := spans.push (r.start, r.start + span)
  -- every pair, so that two ranges' disjointness is a direct fact of the loop
  for i in [0:spans.size] do
    for j in [i + 1:spans.size] do
      if spans[i]!.1 < spans[j]!.2 && spans[j]!.1 < spans[i]!.2 then throw "two ranges overlap"
  let unit := c.unit
  let ur := c.ranges.filter (·.kind == Kind.net c.unitNet)
  if ur.size != 1 || ur[0]!.count != c.g * c.upv || !isPow2 ur[0]!.count then
    throw "not one unit range of vus_per_block x units_per_vu (a power of two) units"
  let g0in := unit.inGroups.getD 0 default
  let g0out := unit.outGroups.getD 0 default
  if unit.inGroups.isEmpty || unit.inGroups.size > 2 || unit.outGroups.isEmpty || unit.outGroups.size > 2
      || g0in.bits.any (· != 16) || g0out.bits.any (· != 16) then
    throw "the unit's ports are not a group of 16-bit leaf inputs (then cut inputs) and one of 16-bit returned outputs"
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
  for n in [0:c.nets.size] do
    let net := (c.nets.getD n default).2
    let first := if n == c.unitNet then g0in.bits.size else 0
    for i in [0:c.perVu.getD n 0] do
      for p in [first:(portCols net.inGroups).size] do
        if ((wired.find? (·.1 == (n, i, p))).map (·.2)).getD 0 != 1 then
          throw s!"input port {p} of {(c.nets.getD n default).1} slot {i} is not wired exactly once"
  let nIn := c.inWords
  let nOut := c.outWords
  if c.leavesIn.size != c.upv || c.leavesOut.size != c.upv then throw "the leaf maps are not one per unit"
  let mut seen : Array Bool := Array.replicate nOut false
  for u in [0:c.upv] do
    let li := c.leavesIn[u]!
    let lo := c.leavesOut[u]!
    if li.size != g0in.bits.size || li.any (fun x => x < -1 || x ≥ (nIn : Int)) || lo.size != g0out.bits.size then
      throw s!"unit {u}'s leaf maps are not its circuit's ports"
    for o in lo do
      if o ≥ nOut || seen[o]! then throw s!"output leaf {o} is outside the outputs or returned twice"
      seen := seen.set! o true
  if seen.any (!·) then throw "an output leaf no unit returns"
  if unit.useful ≥ 2 ^ unit.unitLog then throw "the unit fills its slot: no forced-zero row for a zero leaf"
  if c.dummyDigests.size != c.ports.size || c.dummyOutputs.size != nOut then
    throw "the pinned padding values are not one per port and output leaf"

def FORBIDDEN : List String := ["tail", "cut", "cut_words", "public_ports", "public", "native", "outputs_by_tail"]

/-- Parse the circuit file and check its generic layout. -/
def parse (tags : Tags) (bytes : ByteArray) (tables : List (String × ByteArray) := []) : V Circuit := do
  let sha := Sha256.hash bytes
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
  for k in FORBIDDEN do
    if J.has mj k then throw s!"META: {k} (a verifier-evaluated part of the computation); the verifier only checks proofs"
  -- lookup slots: built from their pinned tables, appended after the expanded circuits, in key order
  let lks ← match mj.getObjVal? "lookups" with
    | .ok (Json.obj kvs) => pure (kvs.toArray.qsort (fun a b => a.1 < b.1))
    | .ok Json.null => pure #[]
    | .ok _ => throw "META: lookups is not an object"
    | .error _ => pure #[]
  for (name, spec) in lks do
    let some pinned := MUFU_TABLES.find? (·.name == name) | throw "META: a lookup of a table the statement does not pin"
    let size := pinned.words
    if (← J.str spec "sha256" ctx) != pinned.sha256 || (← J.str spec "table" ctx) != name || (← J.nat spec "out_bits" ctx) != 31 then
      throw s!"META: lookup {name} is not the pinned table"
    let n ← J.nat spec "in_bits" ctx
    let lo ← J.nat spec "lo_bits" ctx
    if 2 ^ n != size || !(2 ≤ lo && lo < n) then throw s!"META: lookup {name}'s index is not its table's"
    let some (_, words) := tables.find? (·.1 == name) | throw s!"the circuit has a lookup of {name}: its table is not given"
    if words.size != 4 * size || hexOf (Sha512.hash words) != pinned.sha512 then throw s!"MUFU table {name}: not the pinned table"
    nets := nets.push (s!"lookup:{name}", ← Lookup.build s!"lookup:{name}" words n lo 31)
  let ports ← (← J.arr mj "ports" ctx).mapM fun p => do
    let a ← J.asArr p s!"{ctx}: a port"
    if a.size != 4 then throw s!"{ctx}: a port is not [name, words, chunks, nb]"
    pure ({ name := ← J.asStr a[0]! "port name", words := ← J.asNat a[1]! "port words", chunks := ← J.asNat a[2]! "port chunks",
            nb := ← J.asNat a[3]! "port nb" } : Port)
  let g ← J.nat mj "vus_per_block" ctx
  let upv ← J.nat mj "units_per_vu" ctx
  let some unitNet := nets.findIdx? (·.1 == "unit") | throw "circuit: no unit circuit"
  let netOf := fun (nm : String) => match nets.findIdx? (·.1 == nm) with
    | some i => (pure i : V Nat)
    | none => throw s!"META: unknown circuit {nm}"
  let ranges ← (← J.arr mj "ranges" ctx).mapM fun r => do
    let kind ← match ← J.str r "type" ctx with
      | "comp" => pure Kind.comp
      | "mask" => pure Kind.mask
      | "unit" => pure (Kind.net (← netOf (← J.str r "net" ctx)))
      | _ => throw "a range of an unknown type"
    let role ← match r.getObjVal? "role", r.getObjVal? "port" with
      | .ok (Json.str "runs"), .ok p => pure (Role.runs (← J.asNat p "range port") (← (← J.arr r "slots" ctx).mapM natPair))
      | .ok (Json.str "parents"), .ok p => pure (Role.parents (← J.asNat p "range port"))
      | .error _, .error _ => pure Role.none
      | _, _ => throw "a range of an unknown role"
    pure ({ kind, slotLog := ← J.nat r "slot_log" ctx, pos0 := ← J.nat r "pos0" ctx, count := ← J.nat r "count" ctx, role } : Range)
  let leavesIn ← intLists mj "leaves_in"
  let leavesOut ← (← intLists mj "leaves_out").mapM fun u => u.mapM fun x =>
    if x < 0 then (throw "META: a negative output leaf" : V Nat) else pure x.toNat
  let key ← unhex (← J.str mj "row_key" ctx)
  if key.size != 32 then throw "META: row key length"
  let dummy ← J.get mj "dummy" ctx
  let dummyDigests ← (← J.arr dummy "digests" ctx).mapM fun d => do
    let b ← unhex (← J.asStr d "dummy digest")
    if b.size != 32 then throw "META: dummy digest length"
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
  let wires ← match mj.getObjVal? "wires" with
    | .ok w => (← J.asArr w "wires").mapM fun x => do
      let a ← J.asArr x "wire"
      if a.size != 6 then throw "META: a wire is not six fields"
      pure (⟨(← netOf (← J.asStr a[0]! "wire"), ← J.asNat a[1]! "wire", ← J.asNat a[2]! "wire"),
              (← netOf (← J.asStr a[3]! "wire"), ← J.asNat a[4]! "wire", ← J.asNat a[5]! "wire")⟩ : Wire)
    | .error _ => pure #[]
  let leafScheme ← J.str (← J.get mj "leaf_scheme" ctx) "id" ctx
  if leafScheme != tags.merkleLeaf then throw s!"leaf-commitment scheme {leafScheme} is not one this verifier implements"
  if let some pinned := tags.pinnedLeafScheme then
    if (← canon (← J.get mj "leaf_scheme" ctx)) != (← canon pinned) then
      throw "META: leaf_scheme is not the pinned leaf-commitment scheme"
  let c : Circuit := { name, sha, nets, ranges, ports, outPorts, g, kLog := ← J.nat mj "k_log" ctx, upv, unitNet,
                       leavesIn, leavesOut, key, dummyDigests, dummyOutputs, leafScheme, perVu, wires }
  c.checkLayout
  return c

end Circuit

end Flock
