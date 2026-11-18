/-
The circuit file of one block (`verity_flock.circuit.compose` for a flat block of bit rows): a unit laid out as one slot
type, its input rows and one zero-padded output row per output port, written as the bytes `HmOut.parseCommit` reads.
-/
module

public import Flock.HmRow
public import Flock.Tags

@[expose] public section

namespace Flock.Compose

open Lean (Json)
open HmRow

/-- A block for the file: its unit `laid`, whose input group is one 16-bit leaf per input word in `ins`' order and whose
output group is one 16-bit leaf per output word in `outs`' order; `ins` are the input rows (name, length in bits), `outs`
the output ports (name, u16 words). -/
structure Block where
  name : String
  laid : HmNets.Laid
  ins : Array (String × Nat)
  outs : Array (String × Nat)
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

/-- The unit's output words on the all-zero input: each row `(Σ_A z)(Σ_B z)` over GF(2), the constant 1, every input
row 0. -/
def zeroOutputs (l : HmNets.Laid) : Array Nat := Id.run do
  let nIn := (l.inGroups.back?.map fun g => g.col + g.words).getD 0 * WORD
  let mut z : Array Bool := (Array.replicate l.useful false).set! l.constPos true
  for i in [nIn:l.useful] do
    if i != l.constPos then
      let s := fun (r : Array Nat) => r.foldl (fun acc c => acc ^^ z.getD c false) false
      z := z.set! i (s (l.ra.getD i #[]) && s (l.rb.getD i #[]))
  return (portCols l.outGroups).map fun (at_, w) =>
    (List.range w).foldl (fun acc t => acc + if z.getD (at_ + t) false then 2 ^ t else 0) 0

/-- A bit row of `l` bits (`sha512/row/v2`), as `rowPorts` reads it from META. -/
def bitPort (name : String) (l : Nat) : Port :=
  { name, words := (l + 15) / 16, chunks := ((l + 7) / 8 + 17 + 127) / 128 - 1, nb := 0, bits := l, v2 := true }

/-- An output port's row: its words zero-padded to whole SHA-512 blocks, a bit row of all of them (`circuit.out_row`). -/
def outPort (name : String) (words : Nat) : Port := bitPort name (16 * ((words + 63) / 64 * 64))

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

/-- The block's circuit file under `Tags.circuit`: one VU of one unit, the unit then the `sha512x3` and `hm96` sections,
META in canonical JSON. Refuses a unit whose groups are not the rows' and ports' 16-bit leaves, or that fills its slot. -/
def compose (b : Block) (programSha512 partitionDigest : String) : V ByteArray := do
  let l := b.laid
  let nIn := b.ins.foldl (fun a p => a + (p.2 + 15) / 16) 0
  let nOut := b.outs.foldl (fun a p => a + p.2) 0
  if l.inGroups.size != 1 || l.outGroups.size != 1 || l.inGroups[0]!.bits != Array.replicate nIn 16
      || l.outGroups[0]!.bits != Array.replicate nOut 16 then
    throw s!"{b.name}: the unit's groups are not one 16-bit leaf per input row word and one per output word"
  let ul := max 7 (log2ceil l.useful)
  if l.useful == 2 ^ ul then throw s!"{b.name}: the unit fills its slot; a zero leaf needs a forced-zero row"
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
  let outputs := zeroOutputs l
  let mut at_ := 0
  let mut bcs := ins.map zeroRowBc
  for (p, (_, w)) in orows.zip b.outs do
    let row := (Array.range w).foldl (fun r i => let v := outputs.getD (at_ + i) 0
      r.push (v % 256).toUInt8 |>.push (v / 256).toUInt8) ByteArray.empty
    bcs := bcs.push (HmOut.rowBcZeroSalt p row)
    at_ := at_ + w
  let netName := fun (n : Nat) => if n == 1 then SHA_NET else if n == 2 then HM_NET else "unit"
  let wires := (rowWires ports 1 2).map fun w =>
    Json.arr #[str (netName w.dst.1), nat w.dst.2.1, nat w.dst.2.2, str (netName w.src.1), nat w.src.2.1, nat w.src.2.2]
  let mj := Json.mkObj [
    ("format", str Tags.circuit.fileFormat), ("statement", str Tags.circuit.statement), ("scheme", str SCHEME),
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
    ("dummy", Json.mkObj [("bc", Json.arr (bcs.map (str ∘ hexOf))), ("outputs", nats outputs)]),
    ("leaf_scheme", Tags.leafSchemeHm96Os), ("mask_log", nat MASK_LOG),
    ("per_vu", Json.mkObj [("unit", nat 1), (SHA_NET, nat nSha), (HM_NET, nat nHm)]),
    ("lookups", Json.mkObj []), ("wires", Json.arr wires), ("program_sha512", str programSha512),
    ("partition", Json.mkObj [("rule", str PARTITION_V1), ("digest", str partitionDigest)])]
  let sec := fun (nm : String) (s : HmNets.Laid) =>
    s!"{Tags.circuit.blockKeyword} {nm} {1 + s.useful}\n" ++ Laid.write (if nm == "unit" then b.name else nm) s
  let text := s!"{Tags.circuit.fileFormat} {b.name}\nMETA {← canon mj}\n" ++ sec "unit" l
    ++ sec SHA_NET (HmNets.shaLaid ()) ++ sec HM_NET (HmNets.hmLaid ())
  return text.toUTF8

end Flock.Compose
