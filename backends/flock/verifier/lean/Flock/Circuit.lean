/-
PROTOCOL.md §16.5: the circuit file (`flock-netlist/v1`): line 1 names it, line 2 is `META <json>` (the layout), then
`NET <name> <lines>` blocks, each an expanded circuit in `flock-ir-unit/v2` form. The parser refuses every part the verifier
would have to evaluate (a `CUT` line; the META keys `tail`, `cut`, `cut_words`, `public_ports`, `public`, `native`,
`outputs_by_tail`), and checks the layout's shape. Lookup slots are refused: this verifier does not implement them yet.
-/
import Lean.Data.Json
import Flock.Comp

namespace Flock

open Lean (Json)

def WORD : Nat := 128

def log2ceil (n : Nat) : Nat := if n ≤ 1 then 0 else Nat.log2 (n - 1) + 1
def isPow2 (n : Nat) : Bool := n > 0 && 2 ^ Nat.log2 n == n
def nextPow2 (n : Nat) : Nat := 2 ^ log2ceil n

def tokens (s : String) : Array String := Id.run do
  let mut out := #[]
  let mut cur := ""
  for c in s.toList do
    if c.isWhitespace then
      if cur != "" then
        out := out.push cur
        cur := ""
    else cur := cur.push c
  if cur != "" then out := out.push cur
  return out

/-- A port group of an expanded circuit: its first 128-bit word column, its word count (an aligned power of two) and its
ports' widths. -/
structure Group where
  col : Nat
  words : Nat
  bits : Array Nat
  deriving Inhabited, Repr

/-- Every port of a group list: its first column in the slot and its width, groups in order. -/
def portCols (gs : Array Group) : Array (Nat × Nat) := Id.run do
  let mut out := #[]
  for g in gs do
    let mut at_ := g.col * WORD
    for w in g.bits do
      out := out.push (at_, w)
      at_ := at_ + w
  return out

/-- An expanded circuit in `flock-ir-unit/v2` form, `2^unitLog` rows (rows past `useful` are empty). -/
structure Net where
  name : String
  useful : Nat
  constPos : Nat
  inGroups : Array Group
  outGroups : Array Group
  inWords : Nat
  a : Sparse
  b : Sparse
  unitLog : Nat
  deriving Inhabited

namespace Net

private def num (h : Array String) (i : Nat) : V Nat :=
  match h[i]?.bind String.toNat? with
  | some n => pure n
  | none => throw s!"flock-ir-unit/v2: header field {i} is not a number"

private def groups (h : Array String) (at0 : Nat) : V (Array Group × Nat) := do
  let count ← num h at0
  let mut at_ := at0 + 1
  let mut out := #[]
  for _ in [0:count] do
    let col ← num h at_
    let words ← num h (at_ + 1)
    let np ← num h (at_ + 2)
    let mut bits := #[]
    for i in [0:np] do bits := bits.push (← num h (at_ + 3 + i))
    if !(isPow2 words && col % words == 0) then throw "flock-ir-unit/v2: a port group is not an aligned power of two of words"
    out := out.push ⟨col, words, bits⟩
    at_ := at_ + 3 + np
  return (out, at_)

/-- Parse one expanded circuit (the lines of a `NET` block). -/
def parse (lines : Array String) : V Net := do
  let h := tokens (lines.getD 0 "")
  if h[0]? != some "flock-ir-unit/v2" then throw "not a flock-ir-unit/v2 circuit"
  let name := h.getD 1 ""
  let useful ← num h 2
  let constPos ← num h 3
  let (inGroups, at1) ← groups h 4
  let (outGroups, at2) ← groups h at1
  if at2 != h.size then throw "flock-ir-unit/v2: trailing header fields"
  if outGroups.isEmpty then throw "flock-ir-unit/v2: no output group"
  if constPos + 1 != useful then throw "flock-ir-unit/v2: the constant is not the last useful column"
  let inWords := (inGroups.back?.map fun g => g.col + g.words).getD 0
  let unitLog := max 7 (log2ceil useful)
  let mut ra := #[]
  let mut rb := #[]
  for i in [0:useful] do
    let v ← (tokens (lines.getD (1 + i) "")).mapM fun t => match t.toNat? with
      | some n => pure n
      | none => throw s!"flock-ir-unit/v2: row {i} is not numbers"
    let na := v.getD 0 0
    let nb := v.getD (1 + na) 0
    if v.size == 0 || v.size != 2 + na + nb then throw s!"flock-ir-unit/v2: row {i} is malformed"
    ra := ra.push (v.extract 1 (1 + na))
    rb := rb.push (v.extract (2 + na) (2 + na + nb))
  for i in [0:inWords * WORD] do
    let ai := ra.getD i #[]
    let bi := rb.getD i #[]
    if !((ai == #[i] && bi == #[i]) || (ai.isEmpty && bi.isEmpty)) then throw s!"flock-ir-unit/v2: row {i} is not an input row"
  if ra.getD constPos #[] != #[constPos] || rb.getD constPos #[] != #[constPos] then
    throw "flock-ir-unit/v2: the constant row is not A = B = [const]"
  -- after the rows: an optional `LEAVES <json>` line (bound by the pin, not read), and nothing else
  let mut at_ := 1 + useful
  if (lines.getD at_ "").startsWith "LEAVES " then
    match Json.parse ((lines.getD at_ "").drop 7).toString with
    | .ok _ => at_ := at_ + 1
    | .error e => throw s!"flock-ir-unit/v2: the LEAVES line is not JSON: {e}"
  if (lines.getD at_ "").startsWith "CUT " then
    throw s!"{name}: a CUT line (a tail the verifier evaluates); the verifier evaluates no part of the computation"
  if at_ != lines.size then throw "flock-ir-unit/v2: text after the rows that is not its LEAVES line"
  let a ← Sparse.ofRows ra (2 ^ unitLog)
  let b ← Sparse.ofRows rb (2 ^ unitLog)
  return { name, useful, constPos, inGroups, outGroups, inWords, a, b, unitLog }

end Net

structure Port where
  name : String
  words : Nat
  chunks : Nat
  nb : Nat
  deriving Inhabited, Repr

inductive Kind where
  | comp
  | net (i : Nat)
  | mask
  deriving BEq, Inhabited, Repr

inductive Role where
  /-- chunk runs of input port `p`: run slot `q` holds `(vu, chunk)` or nothing (a padding run) -/
  | runs (p : Nat) (slots : Array (Option (Nat × Nat)))
  /-- port `p`'s parent-tree compressions: per VU `chunks` slots, heap order (root first), the last a padding slot -/
  | parents (p : Nat)
  | none
  deriving Inhabited, Repr

structure Range where
  kind : Kind
  slotLog : Nat
  pos0 : Nat
  count : Nat
  role : Role
  deriving Inhabited, Repr

def Range.start (r : Range) : Nat := r.pos0 * 2 ^ r.slotLog
def Range.slot (r : Range) (s : Nat) : Nat := (r.pos0 + s) * 2 ^ r.slotLog

/-- A copy of every bit of output port `src.2.2` of slot `(src.1, src.2.1)` into input port `dst.2.2` of slot
`(dst.1, dst.2.1)`, both per VU. -/
structure Wire where
  dst : Nat × Nat × Nat
  src : Nat × Nat × Nat
  deriving Inhabited, Repr

def LEAF_SCHEME : String := "flock-leaf/sha256-unsalted/v0"

/-- The pinned circuit file. -/
structure Circuit where
  name : String
  sha : ByteArray
  nets : Array (String × Net)
  ranges : Array Range
  ports : Array Port
  outPorts : Array (String × Nat)
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

/-- The layout's shape (upstream's `Composite::check`): aligned, disjoint ranges inside the block; the compression ranges'
roles cover every run and parent of every VU exactly once; one unit range of `g × units_per_vu` units whose leaf maps are
the unit circuit's ports; every non-leaf input port wired exactly once. -/
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
    if (r.kind == Kind.comp) != (r.slotLog == Comp.kLog && isRole) || (r.kind == Kind.mask && r.count != 1) then
      throw "a range's type, role and slot size disagree"
    spans := spans.push (r.start, r.start + span)
  let sorted := spans.qsort (fun a b => a.1 < b.1 || (a.1 == b.1 && a.2 < b.2))
  for i in [1:sorted.size] do
    if sorted[i - 1]!.2 > sorted[i]!.1 then throw "two ranges overlap"
  let mut runs : Array (Nat × Nat × Nat) := #[]
  let mut parents : Array Nat := #[]
  for r in c.ranges do
    match r.role with
    | .runs p slots =>
      let some port := c.ports[p]? | throw "runs of an unknown port"
      if slots.size * port.nb != r.count || !isPow2 r.count then throw "a runs range's slot list is not its count"
      for x in slots do
        if let some (g, ch) := x then
          if g ≥ c.g || ch ≥ port.chunks || runs.contains (g, p, ch) then throw "a run outside the block's VUs or listed twice"
          runs := runs.push (g, p, ch)
    | .parents p =>
      let some port := c.ports[p]? | throw "parents of an unknown port"
      if port.chunks < 2 || r.count != c.g * port.chunks || parents.contains p then
        throw "a parents range is not one per multi-chunk port"
      parents := parents.push p
    | .none => pure ()
  let want := c.ports.foldl (· + ·.chunks) 0 * c.g
  if runs.size != want || parents.size != (c.ports.filter (·.chunks > 1)).size then
    throw "the runs do not cover every chunk of every VU (or a multi-chunk port has no parent tree)"
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
  if c.leafScheme != LEAF_SCHEME then throw s!"leaf-commitment scheme {c.leafScheme} is not one this verifier implements"

def FORBIDDEN : List String := ["tail", "cut", "cut_words", "public_ports", "public", "native", "outputs_by_tail"]

/-- Parse and check the circuit file. -/
def parse (bytes : ByteArray) : V Circuit := do
  let sha := Sha256.hash bytes
  let text ← match String.fromUTF8? bytes with
    | some s => pure s
    | none => throw "circuit: not UTF-8"
  let lines0 := (text.splitOn "\n").toArray
  let lines := if lines0.back? == some "" then lines0.pop else lines0
  let head := (lines.getD 0 "").trimRight
  if !head.startsWith "flock-netlist/v1 " then throw "circuit: not a flock-netlist/v1 file"
  let name := (head.drop 17).toString
  let metaLine := (lines.getD 1 "").trimRight
  if !metaLine.startsWith "META " then throw "circuit: no META line"
  let mj ← match Json.parse (metaLine.drop 5).toString with
    | .ok j => pure j
    | .error e => throw s!"META: {e}"
  if (← J.str mj "format" ctx) != "flock-netlist/v1" || (← J.str mj "statement" ctx) != "verity/flock-netlist/v1" then
    throw "META: not verity/flock-netlist/v1"
  let mut nets : Array (String × Net) := #[]
  let mut at_ := 2
  while at_ < lines.size do
    let h := tokens lines[at_]!
    if h.size != 3 || h[0]! != "NET" then throw "circuit: expected a NET line"
    let some n := h[2]!.toNat? | throw "circuit: NET line count"
    if at_ + 1 + n > lines.size then throw s!"circuit: NET {h[1]!} is truncated"
    let net ← match Net.parse (lines.extract (at_ + 1) (at_ + 1 + n)) with
      | .ok v => pure v
      | .error e => throw s!"NET {h[1]!}: {e}"
    nets := nets.push (h[1]!, net)
    at_ := at_ + 1 + n
  for k in FORBIDDEN do
    if J.has mj k then throw s!"META: {k} (a verifier-evaluated part of the computation); the verifier only checks proofs"
  match mj.getObjVal? "lookups" with
  | .ok (Json.obj kvs) => if kvs.size > 0 then throw "META: lookup slots are not implemented by this verifier"
  | .ok Json.null => pure ()
  | .ok _ => throw "META: lookups is not an object"
  | .error _ => pure ()
  let ports ← (← J.arr mj "ports" ctx).mapM fun p => do
    let a ← J.asArr p s!"{ctx}: a port"
    if a.size != 4 then throw s!"{ctx}: a port is not [name, words, chunks, nb]"
    pure (⟨← J.asStr a[0]! "port name", ← J.asNat a[1]! "port words", ← J.asNat a[2]! "port chunks", ← J.asNat a[3]! "port nb"⟩ : Port)
  for p in ports do
    let b := 2 * p.words
    let ok := isPow2 p.nb && p.chunks ≥ 1 && b % 64 == 0 && p.chunks == (b + 1023) / 1024 && p.nb == min b 1024 / 64
      && (p.chunks == 1 || (b % 1024 == 0 && isPow2 p.chunks))
    if !ok then throw s!"port {p.name}: not whole power-of-two chunk runs"
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
    pure (⟨kind, ← J.nat r "slot_log" ctx, ← J.nat r "pos0" ctx, ← J.nat r "count" ctx, role⟩ : Range)
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
  let c : Circuit := { name, sha, nets, ranges, ports, outPorts, g, kLog := ← J.nat mj "k_log" ctx, upv, unitNet,
                       leavesIn, leavesOut, key, dummyDigests, dummyOutputs, leafScheme, perVu, wires }
  c.checkLayout
  return c

end Circuit

end Flock
