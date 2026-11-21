/-
The circuit file of one block (`verity_flock.circuit.compose` for a block of bit rows): its unit, flat or templated, its
input rows and one zero-padded output row per output port, written as the bytes `HmOut.parseCommit` reads.
-/
module

public import Flock.HmRow
public import Flock.Tags
public import Flock.Lower
public import Flock.NetAt
public import Flock.TmplAt
public import Std.Data.HashMap

@[expose] public section

namespace Flock.Compose

open Lean (Json)
open HmRow

/-- A block's unit: laid out flat (a `flock-ir-unit/v2` section, the statement `Tags.circuit`'s), or a template read at
the layout mode `layout` (`Tmpl.modeOf`; a `flock-ir-tmpl/v2` section, `Tags.circuitTmpl`'s). -/
inductive Body where
  | flat (l : HmNets.Laid)
  | tmpl (t : Tmpl.Tmpl) (layout : String)
  deriving Inhabited

/-- A block for the file: its unit `body`, whose input group is one 16-bit leaf per input word in `ins`' order and whose
output group is one 16-bit leaf per output word in `outs`' order; `ins` are the input rows (name, length in bits), `outs`
the output ports (name, u16 words); `programDigests`, META's `program_digests` when not empty: the template the units are
instances of, by its descriptor id, which a template query's partition reads (`HmRow.circuitTemplate`). -/
structure Block where
  name : String
  body : Body
  ins : Array (String × Nat)
  outs : Array (String × Nat)
  programDigests : List (String × String) := []
  deriving Inhabited

def SLOT_LOG : Nat := 18
def MASK_LOG : Nat := 14
def K_MIN : Nat := 20

def groupText (gs : Array Group) : String :=
  gs.foldl (fun s g => g.bits.foldl (fun s b => s ++ s!" {b}") (s ++ s!" {g.col} {g.words} {g.bits.size}")) s!" {gs.size}"

/-- A `flock-ir-unit/v2` section's lines (`ir_lower.circuit_description_pieces` without `LEAVES`), each ending in a
newline: the header, then one line `|A| A… |B| B…` per row. -/
def Laid.write (name : String) (l : HmNets.Laid) : String := Id.run do
  let mut out := s!"flock-ir-unit/v2 {name} {l.useful} {l.constPos}" ++ groupText l.inGroups ++ groupText l.outGroups ++ "\n"
  for i in [0:l.useful] do
    let a := l.ra.getD i #[]
    let b := l.rb.getD i #[]
    out := a.foldl (fun s c => s ++ s!" {c}") (out ++ toString a.size)
    out := b.foldl (fun s c => s ++ s!" {c}") (out ++ s!" {b.size}") ++ "\n"
  return out

/-- A `CIRCUIT` section of `tags`' files: its header line, then `l`'s lines. -/
def section_ (nm : String) (l : HmNets.Laid) (unitName : String := nm) : String :=
  s!"{Tags.circuit.blockKeyword} {nm} {1 + l.useful}\n" ++ Laid.write unitName l

set_option compiler.extract_closed false in
/-- The `sha512x3` and `hm96` sections, the same in every file: built once per run, when a file first needs them. -/
def rowNets : Thunk String :=
  Thunk.mk fun _ => section_ SHA_NET (HmNets.shaLaid ()) ++ section_ HM_NET (HmNets.hmLaid ())

/-- The words at the bit columns of `gs`' ports (16 bits each) in the column values `z`. -/
def words (gs : Array Group) (z : Array Bool) : Array Nat :=
  (portCols gs).map fun (at_, w) => (List.range w).foldl (fun acc t => acc + if z.getD (at_ + t) false then 2 ^ t else 0) 0

/-- The flat unit's output words on the all-zero input: each row `(Σ_A z)(Σ_B z)` over GF(2), the constant 1, every
input row 0. -/
def zeroOutputs (l : HmNets.Laid) : Array Nat := Id.run do
  let nIn := (l.inGroups.back?.map fun g => g.col + g.words).getD 0 * WORD
  let mut z : Array Bool := (Array.replicate l.useful false).set! l.constPos true
  for i in [nIn:l.useful] do
    if i != l.constPos then
      let s := fun (r : Array Nat) => r.foldl (fun acc c => acc ^^ z.getD c false) false
      z := z.set! i (s (l.ra.getD i #[]) && s (l.rb.getD i #[]))
  return words l.outGroups z

abbrev Memo := Std.HashMap (Nat × Array Bool) (Array Bool)

/-- Definition `d`'s own columns on the input values `inp`, the constant 1 (`Tmpl.eval`'s values): in column order, a
row's `(⊕A) ∧ (⊕B)`, a copy's its callee's on its bound values, else 0; a callee's columns are computed once for each
distinct bound values (`Memo`). `fuel` bounds the call depth. -/
def colsOf (defs : Array Tmpl.Def) : Nat → Nat → Array Bool → StateM Memo (Array Bool)
  | 0, _, _ => pure #[]
  | fuel + 1, d, inp => do
    if let some v := (← get)[(d, inp)]? then return v
    let df := defs.getD d default
    let ref := fun (v : Array Bool) (x : Nat) =>
      if x < df.cols then v.getD x false else if x < df.cols + df.ins then inp.getD (x - df.cols) true else true
    let sum := fun (v : Array Bool) (r : Array Nat) => r.foldl (fun acc x => acc ^^ ref v x) false
    let evs : Array (Nat × Option (Tmpl.Call × Nat)) :=
      (df.rows.map fun r => (r.1, none)) ++ df.calls.flatMap fun c => (Array.range c.count).map fun j => (c.base + j * c.stride, some (c, j))
    let rowAt : Std.HashMap Nat (Array Nat × Array Nat) := df.rows.foldl (fun m r => m.insert r.1 r.2) {}
    let mut v := Array.replicate df.cols false
    for (y, e) in evs.qsort (·.1 < ·.1) do
      match e with
      | none =>
        let (a, b) := rowAt.getD y (#[], #[])
        v := v.set! y (sum v a && sum v b)
      | some (c, j) =>
        let bound := c.segs.flatMap fun s => (Array.range s.reps).flatMap fun r =>
          (Array.range s.len).map fun t => ref v (s.at_ + r * s.rstride + j * s.step + t)
        let w ← colsOf defs fuel c.def_ bound
        for t in [0:w.size] do
          v := v.set! (y + t) w[t]!
    modify (·.insert (d, inp) v)
    return v

/-- The template's output words on the all-zero input: its root's columns (its input rows 0, as their self rows give). -/
def zeroOutputsT (t : Tmpl.Tmpl) : Array Nat :=
  words t.outGroups ((colsOf t.defs t.defs.size t.root #[]).run' {})

/-- A bit row of `l` bits (`sha512/row/v2`), as `rowPorts` reads it from META. -/
def bitPort (name : String) (l : Nat) : Port :=
  { name, words := (l + 15) / 16, chunks := ((l + 7) / 8 + 17 + 127) / 128 - 1, nb := 0, bits := l, v2 := true }

/-- An output port's row: a bit row of exactly its words, the row the period commits for it (`rows.Member.out_rows` of
a statement of bit rows, `circuit._out_ports`' `v2_out_rows`). -/
def outPort (name : String) (words : Nat) : Port := bitPort name (16 * words)

/-- `circuit._ranges`: the unpacked ranges largest first, each aligned to its power-of-two span, then the packed ones
aligned to their slot; and the block's `k_log`. -/
def ranges (items : Array (String × Nat × Nat × List (String × Json))) : Array Json × Nat := Id.run do
  let packed := fun (i : Nat) => items[i]!.2.2.2.any (·.1 == "packed")
  let key := fun (i : Nat) => nextPow2 items[i]!.2.2.1 <<< items[i]!.2.1
  let plain := ((Array.range items.size).filter (!packed ·)).qsort fun i j => key i > key j || (key i == key j && i < j)
  let mut at_ := 0
  let mut out := Array.replicate items.size Json.null
  let one := fun (typ : String) (sl n at_ : Nat) (extra : List (String × Json)) =>
    Json.mkObj ([("type", Json.str typ), ("slot_log", Lean.toJson sl), ("pos0", Lean.toJson (at_ >>> sl)),
      ("count", Lean.toJson n)] ++ extra)
  for i in plain do
    let (typ, sl, n, extra) := items[i]!
    let span := nextPow2 n <<< sl
    at_ := (at_ + span - 1) / span * span
    out := out.set! i (one typ sl n at_ extra)
    at_ := at_ + span
  for i in (Array.range items.size).filter packed do
    let (typ, sl, n, extra) := items[i]!
    at_ := (at_ + (1 <<< sl) - 1) >>> sl <<< sl
    out := out.set! i (one typ sl n at_ extra)
    at_ := at_ + (n <<< sl)
  return (out, max K_MIN (log2ceil at_))

def nat (n : Nat) : Json := Lean.toJson n
def str (s : String) : Json := Json.str s
def nats (a : Array Nat) : Json := Json.arr (a.map nat)

/-- What the file takes from the unit: its statement's tags, its groups, its net's size and slot size, its outputs on
the all-zero input, its section's text and META's keys of its own. -/
structure UnitPart where
  tags : Tags
  inGroups : Array Group
  outGroups : Array Group
  useful : Nat
  unitLog : Nat
  zero : Array Nat
  text : String
  extra : List (String × Json)

def Body.part (name : String) : Body → V UnitPart
  | .flat l => pure {
      tags := Tags.circuit, inGroups := l.inGroups, outGroups := l.outGroups, useful := l.useful,
      unitLog := max 7 (log2ceil l.useful), zero := zeroOutputs l, text := section_ "unit" l name, extra := [] }
  | .tmpl t layout => do
    let some mode := Tmpl.modeOf layout | throw s!"{name}: {layout} is not a layout mode"
    let tu ← TmplUnit.read t mode
    let lines := Lower.renderLines t
    pure {
      tags := Tags.circuitTmpl, inGroups := t.inGroups, outGroups := t.outGroups, useful := tu.net.useful,
      unitLog := tu.net.unitLog, zero := zeroOutputsT t,
      text := lines.foldl (fun s l => s ++ l ++ "\n") s!"TEMPLATE {lines.size}\n", extra := [("layout", str layout)] }

/-- The block's circuit file: one VU of one unit, META in canonical JSON, the unit's section (a flat body's `CIRCUIT
unit`, a template's `TEMPLATE`), then the `sha512x3` and `hm96` sections.
Refuses a unit whose groups are not the rows' and ports' 16-bit leaves, or that fills its slot. -/
def compose (b : Block) (programSha512 partitionDigest : String) : V ByteArray := do
  let u ← b.body.part b.name
  let nIn := b.ins.foldl (fun a p => a + (p.2 + 15) / 16) 0
  let nOut := b.outs.foldl (fun a p => a + p.2) 0
  if u.inGroups.size != 1 || u.outGroups.size != 1 || u.inGroups[0]!.bits != Array.replicate nIn 16
      || u.outGroups[0]!.bits != Array.replicate nOut 16 then
    throw s!"{b.name}: the unit's groups are not one 16-bit leaf per input row word and one per output word"
  let ul := u.unitLog
  if u.useful ≥ 2 ^ ul then throw s!"{b.name}: the unit fills its slot; a zero leaf needs a forced-zero row"
  let ins := b.ins.map fun p => bitPort p.1 p.2
  let orows := b.outs.map fun p => outPort p.1 p.2
  let ports := ins ++ orows
  let (nSha, _) := rowSlots ports
  if nSha == 0 then throw s!"{b.name}: no row has a compression before its last block"
  let nHm := nextPow2 ports.size
  let (rs, kLog) := ranges #[
    ("unit", SLOT_LOG, nSha, [("net", str SHA_NET), ("packed", Json.bool true)]),
    ("unit", SLOT_LOG, nHm, [("net", str HM_NET)]),
    ("unit", ul, 1, [("net", str "unit")]),
    ("mask", MASK_LOG, 1, [("net", str "mask")])]
  if kLog > MAX_K_LOG then throw s!"{b.name}: an instance needs a block of 2^{kLog} bits (at most 2^{MAX_K_LOG})"
  let mut at_ := 0
  let mut bcs := ins.map zeroRowBc
  for (p, (_, w)) in orows.zip b.outs do
    let row := (Array.range w).foldl (fun r i => let v := u.zero.getD (at_ + i) 0
      r.push (v % 256).toUInt8 |>.push (v / 256).toUInt8) ByteArray.empty
    bcs := bcs.push (HmOut.rowBcZeroSalt p row)
    at_ := at_ + w
  let netName := fun (n : Nat) => if n == 1 then SHA_NET else if n == 2 then HM_NET else "unit"
  let wires := (rowWires ports 1 2).map fun w =>
    Json.arr #[str (netName w.dst.1), nat w.dst.2.1, nat w.dst.2.2, str (netName w.src.1), nat w.src.2.1, nat w.src.2.2]
  let mj := Json.mkObj ([
    ("format", str u.tags.fileFormat), ("statement", str u.tags.statement), ("scheme", str SCHEME),
    ("row_schemas", Json.arr (ports.map fun p => Json.arr #[str ROW_SCHEMA_V2, nat p.bits])),
    ("vus_per_block", nat 1), ("k_log", nat kLog),
    ("ports", Json.arr (ports.map fun p => Json.arr #[str p.name, nat p.words, nat p.chunks])),
    ("input_ports", nat ins.size), ("out_ports", Json.arr (b.outs.map fun (nm, w) => Json.arr #[str nm, nat w])),
    ("ranges", Json.arr rs),
    ("rows", Json.arr (ports.map fun p => Json.mkObj [("prefix", str (hexOf (prefixOf p))),
      ("midstate", Json.arr ((midstateOf p).map (str ∘ hex16))), ("pad", str (hexOf (tailOf p)))])),
    ("units_per_vu", nat 1), ("unit_log", nat ul),
    ("leaves_in", Json.arr #[nats (Array.range nIn)]), ("leaves_out", Json.arr #[nats (Array.range nOut)]),
    ("row_key_sha512", str (hexOf (Sha512.hash Hm96.Default512.key))),
    ("dummy", Json.mkObj [("bc", Json.arr (bcs.map (str ∘ hexOf))), ("outputs", nats u.zero)]),
    ("leaf_scheme", Tags.leafSchemeHm96Os), ("mask_log", nat MASK_LOG),
    ("per_vu", Json.mkObj [("unit", nat 1), (SHA_NET, nat nSha), (HM_NET, nat nHm)]),
    ("lookups", Json.mkObj []), ("wires", Json.arr wires)] ++
    (if b.programDigests.isEmpty then [] else [("program_digests", Json.mkObj (b.programDigests.map fun (k, v) => (k, str v)))]) ++
    [("program_sha512", str programSha512),
    ("partition", Json.mkObj [("rule", str PARTITION_V1), ("digest", str partitionDigest)])] ++ u.extra)
  return (s!"{u.tags.fileFormat} {b.name}\nMETA {← canon mj}\n" ++ u.text ++ rowNets.get).toUTF8

end Flock.Compose
