import Flock.HmRow
import VStar.Stream
import VStar.Unit

/-!
# The `flock-circuit` file of a unit (`verity_flock.circuit.compose`)

For a template lowered to one unit per instance with no cut, no public input and `sha512/row/v1` rows (V*'s statements):
line 1 `flock-circuit <unit>`, line 2 `META` and its JSON (`pyDumps`), then the sections `CIRCUIT <name> <lines>` of the unit,
`sha512x3` and `hm96` (`ir_lower.netlist` without `LEAVES`). META holds the rows (each port's words and compressions, its
prefix, midstate and padding), the slot ranges and the block size they give (`placeRanges`, `vusPerBlock`), the leaf maps,
the all-zero instance's outputs and each row's `b ‖ c` under a zero salt (`dummy`), the rows' wires, and the statement's
bindings (`Bindings`), which come from core's descriptor codec and are taken as given here.
-/

namespace VStar

open Lean (Json)
open Flock.HmNets

def FORMAT_CIRCUIT : String := "flock-circuit"
def STATEMENT : String := "verity/flock-circuit"
def SCHEME : String := "frame-v3-sha512/hm96-sha512"
def ROW_SCHEMA : String := "hm96-sha512/row/v1"
def SHA_NET : String := "sha512x3"
def HM_NET : String := "hm96"
def PER_SLOT : Nat := 3
def SHA_LOG : Nat := 18
def HM_LOG : Nat := 18
def MASK_LOG : Nat := 14
def K_MAX : Nat := 27
def PARTITION_V1 : String := "verity/partition/v1"
def SALT_BYTES : Nat := 192
def KEY_SHA512 : String :=
  "afc5a60c85d8aa75cb4c7f726588d299a1712e1c6a621f09ca64809a5fedd744a0f6b0201d05e82b29cd5b274e6c29670cd4e4ab10831911fb1f9e0c7019df22"

/-- `circuit.LEAF_SCHEME`: `hm96-sha512/v1` over the column digest, under the default key. -/
def leafScheme : Json := Json.mkObj [
  ("id", "hm96-sha512/v1"), ("inner", "sha512 column digest"), ("salt_bytes", (192 : Nat)),
  ("salt_source", "os bytes (getrandom), 192 per leaf, each tree's read once, never reused or stored"),
  ("key", "default_key, a statement constant for every tree"), ("key_sha512", KEY_SHA512)]

/-- What `compose` reads of a subcircuit: its id, its template's, and its ports' names. -/
structure Sub where
  id : String
  template : String
  inputs : List String
  outputs : List String

/-- META's bindings: the `verity/partition/v1` digest, the Program's SHA-512 and `program_digests` (descriptor id to
digest), core's codec's, as given. -/
structure Bindings where
  partition : String
  program : String
  programDigests : Json

def isSha512Hex (s : String) : Bool := s.length == 128 && s.all fun c => c.isDigit || ('a' ≤ c && c ≤ 'f')

/-- `int.bit_length`. -/
def bitLength (n : Nat) : Nat := if n = 0 then 0 else Nat.log2 n + 1

/-- `circuit._pow2`: the smallest power of two at least `x` (`x ≥ 1`). -/
def pow2 (x : Nat) : Nat := 2 ^ bitLength (x - 1)

/-- `circuit._slot_log`. -/
def slotLog (useful : Nat) : Nat := max 7 (bitLength (useful - 1))

/-- A `sha512/row/v1` row: its name and words. -/
structure Row where
  name : String
  words : Nat

/-- `Port.comps`: compressions after the constant first block, all of the message's but the last (the hm96 slot's). -/
def Row.comps (p : Row) : Nat := (2 * p.words + 17 + 127) / 128 - 1

/-- `circuit._port`'s refusal for a `sha512/row/v1` row: whole SHA-512 blocks. -/
def checkRow (p : Row) : Except String Unit :=
  if p.words == 0 || (2 * p.words) % 128 != 0 then
    throw s!"port {p.name}: {2 * p.words}-byte rows are not whole SHA-512 blocks (sha512/row/v1)"
  else pure ()

/-- `circuit._row_consts`: the row's prefix, the chaining value after it, SHA-512's padding after its bytes. -/
def rowConsts (p : Row) : Json := Json.mkObj [
  ("prefix", Flock.hexOf (Flock.HmRow.rowPrefix p.words)),
  ("midstate", Json.arr ((Flock.HmRow.midstate p.words).map fun w => (Flock.HmRow.hex16 w : Json))),
  ("pad", Flock.hexOf (Flock.HmRow.padBlock (128 + 2 * p.words)))]

/-! ## Slots and ranges -/

/-- `circuit._row_slots`'s placement: each row's compressions as `(slot, sub)`, three a slot, rows in order. -/
def rowSlots (ports : List Row) : Nat × List (List (Nat × Nat)) :=
  let r := ports.foldl (fun (acc : Nat × List (List (Nat × Nat))) p =>
    (acc.1 + p.comps, acc.2 ++ [(List.range p.comps).map fun k => ((acc.1 + k) / PER_SLOT, (acc.1 + k) % PER_SLOT)])) (0, [])
  ((r.1 + PER_SLOT - 1) / PER_SLOT, r.2)

/-- `circuit._row_wires`: each row's chain (compression `k`'s `cv` from compression `k - 1`'s `out`), then its hm96 slot's
`cv` from its last. -/
def rowWires (ports : List Row) : List Json :=
  let w := (rowSlots ports).2
  let wire := fun (dn : String) (di dp : Nat) (sn : String) (si sp : Nat) =>
    Json.arr #[(dn : Json), (di : Json), (dp : Json), (sn : Json), (si : Json), (sp : Json)]
  (w.zipIdx).flatMap fun (ws, p) =>
    let a := ws.toArray
    ((List.range a.size).drop 1).map (fun k =>
      wire SHA_NET (a.getD k (0, 0)).1 (2 * (a.getD k (0, 0)).2) SHA_NET (a.getD (k - 1) (0, 0)).1 (a.getD (k - 1) (0, 0)).2) ++
    (if a.size = 0 then [] else [wire HM_NET p 0 SHA_NET (a.getD (a.size - 1) (0, 0)).1 (a.getD (a.size - 1) (0, 0)).2])

/-- A range to place: its type, slot log, count, net, and whether no region reads it (`packed`). -/
structure Item where
  typ : String
  sl : Nat
  n : Nat
  net : String
  packed : Bool
  deriving Inhabited

/-- `circuit._ranges`: the plain ranges largest first (ties in order), each aligned to its power-of-two span, then the packed
ones, each aligned to its slot and spanning its count; each range's `pos0`, and the block's log size. -/
def placeRanges (items : Array Item) : Array Nat × Nat := Id.run do
  let span := fun (it : Item) => pow2 it.n * 2 ^ it.sl
  let plain := ((Array.range items.size).filter fun i => !(items.getD i default).packed).qsort fun i j =>
    let si := span (items.getD i default)
    let sj := span (items.getD j default)
    si > sj || (si == sj && i < j)
  let mut at_ := 0
  let mut pos := Array.replicate items.size 0
  for i in plain do
    let it := items.getD i default
    let s := span it
    at_ := (at_ + s - 1) / s * s
    pos := pos.set! i (at_ / 2 ^ it.sl)
    at_ := at_ + s
  for i in [0:items.size] do
    let it := items.getD i default
    if it.packed then
      at_ := (at_ + 2 ^ it.sl - 1) / 2 ^ it.sl * 2 ^ it.sl
      pos := pos.set! i (at_ / 2 ^ it.sl)
      at_ := at_ + it.n * 2 ^ it.sl
  return (pos, max 20 (bitLength (at_ - 1)))

def rangeJson (it : Item) (pos0 : Nat) : Json :=
  Json.mkObj ([("type", (it.typ : Json)), ("slot_log", (it.sl : Json)), ("pos0", (pos0 : Json)), ("count", (it.n : Json)),
    ("net", (it.net : Json))] ++ if it.packed then [("packed", (true : Json))] else [])

/-- The ranges of `G` VUs a block: the sha512x3 slots (packed), the hm96 slots, the unit's, the mask's. -/
def items (G nSha nHm ul : Nat) : Array Item :=
  #[⟨"unit", SHA_LOG, G * nSha, SHA_NET, true⟩, ⟨"unit", HM_LOG, G * nHm, HM_NET, false⟩, ⟨"unit", ul, G, "unit", false⟩,
    ⟨"mask", MASK_LOG, 1, "mask", false⟩]

/-- `compose`'s VUs per block: doubled while at most 4096 a block and the block no larger than `max(k1, 22)`. -/
def vusPerBlock (nSha nHm ul : Nat) : Nat := Id.run do
  let k1 := (placeRanges (items 1 nSha nHm ul)).2
  let mut G := 1
  for _ in [0:12] do
    if 2 * G ≤ 4096 && (placeRanges (items (2 * G) nSha nHm ul)).2 ≤ max k1 22 then G := 2 * G
    else break
  return G

/-! ## The dummy instance -/

def evLin (z : Array Bool) (f : Lin) : Bool := f.s.foldl (fun acc v => acc ^^ z.getD v false) f.c

/-- The all-zero instance's output words: every committed bit evaluated in creation order from zero inputs. -/
def zeroOutputs (u : UnitC) : Array Nat := Id.run do
  let mut z := Array.replicate u.st.next false
  for r in u.st.rows.reverse do
    z := z.set! r.out (evLin z r.a && evLin z r.b)
  let bits := u.outs.map (evLin z)
  return (Array.range u.outWords).map fun j =>
    (List.range 16).foldl (fun acc b => if bits.getD (16 * j + b) false then acc + 2 ^ b else acc) 0

/-- A row's `b ‖ c` under a zero salt: `hm96-sha512/v1` of its `sha512/row/v1` digest. -/
def zeroSaltBc (p : Row) (ws : Array Nat) : ByteArray :=
  let row := (Array.range p.words).foldl (fun b i => let w := ws.getD i 0; (b.push (w % 256).toUInt8).push (w / 256).toUInt8)
    ByteArray.empty
  Flock.Hm96.Default512.commitString (Flock.Sha512.hash (Flock.HmRow.rowPrefix p.words ++ row))
    (ByteArray.mk (Array.replicate SALT_BYTES 0))

/-! ## META and the file -/

/-- The unit's lowering text (`Lowering.text`: its netlist with `LEAVES`): SHA-256, bytes. -/
def unitDigests (u : UnitC) (l : Laid) : Except String Digests := do
  let lv ← pyDumps u.leaves
  pure ((emitNet hashPut u.name l (some lv)).run {}).2

/-- The file's first two lines. -/
def metaHead (u : UnitC) (l : Laid) (sub : Sub) (b : Bindings) (unitSha256 : String) : Except String String := do
  for (nm, v) in [("partition", b.partition), ("program", b.program)] do
    unless isSha512Hex v do throw s!"{nm}: a SHA-512 digest is 128 lowercase hex digits"
  unless sub.inputs.length == u.inPorts.length && sub.outputs.length == u.outPorts.length do
    throw s!"{u.name}: the subcircuit's ports are not the unit's"
  let ins := (sub.inputs.zip u.inPorts).map fun (nm, w) => (⟨nm, w⟩ : Row)
  for p in ins do checkRow p
  let outs := sub.outputs.zip u.outPorts
  let orows := outs.map fun (nm, w) => (⟨nm, (w + 63) / 64 * 64⟩ : Row)
  let ports := ins ++ orows
  let ul := slotLog l.useful
  if l.useful == 2 ^ ul then throw s!"{u.name}: a unit fills its slot; a zero leaf needs a forced-zero row"
  let nSha := (rowSlots ports).1
  if nSha == 0 then throw s!"{u.name}: no row has a compression before its last block"
  let nHm := pow2 ports.length
  let G := vusPerBlock nSha nHm ul
  let its := items G nSha nHm ul
  let (pos, kLog) := placeRanges its
  if kLog > K_MAX then throw s!"{u.name}: an instance needs a block of 2^{kLog} bits (at most 2^{K_MAX})"
  let words := zeroOutputs u
  let outRow := fun (j : Nat) =>
    let at_ := (u.outPorts.take j).foldl (· + ·) 0
    (Array.range (u.outPorts.getD j 0)).map fun i => words.getD (at_ + i) 0
  let bcs := ins.map (fun p => zeroSaltBc p #[]) ++ (List.range orows.length).map fun j => zeroSaltBc (orows.getD j ⟨"", 0⟩) (outRow j)
  let idx := fun (n : Nat) => Json.arr #[Json.arr ((Array.range n).map fun (i : Nat) => (i : Json))]
  let nat := fun (n : Nat) => (n : Json)
  let str := fun (s : String) => (s : Json)
  let m := Json.mkObj [
    ("format", str FORMAT_CIRCUIT), ("statement", str STATEMENT), ("scheme", str SCHEME), ("row_schema", str ROW_SCHEMA),
    ("template", str sub.template), ("subcircuit", str sub.id), ("unit", str u.name), ("unit_sha256", str unitSha256),
    ("vus_per_block", nat G), ("k_log", nat kLog),
    ("ports", Json.arr (ports.toArray.map fun p => Json.arr #[str p.name, nat p.words, nat p.comps])),
    ("input_ports", nat ins.length),
    ("out_ports", Json.arr (outs.toArray.map fun (nm, w) => Json.arr #[str nm, nat w])),
    ("ranges", Json.arr ((Array.range its.size).map fun i => rangeJson (its.getD i default) (pos.getD i 0))),
    ("rows", Json.arr (ports.toArray.map rowConsts)), ("units_per_vu", nat 1), ("unit_log", nat ul),
    ("leaves_in", idx u.inWords), ("leaves_out", idx u.outWords), ("row_key_sha512", str KEY_SHA512),
    ("dummy", Json.mkObj [("bc", Json.arr (bcs.toArray.map fun x => str (Flock.hexOf x))),
                          ("outputs", Json.arr (words.map nat))]),
    ("leaf_scheme", leafScheme), ("mask_log", nat MASK_LOG),
    ("per_vu", Json.mkObj [("unit", nat 1), (SHA_NET, nat nSha), (HM_NET, nat nHm)]), ("lookups", Json.mkObj []),
    ("wires", Json.arr (rowWires ports).toArray), ("program_digests", b.programDigests), ("program_sha512", str b.program),
    ("partition", Json.mkObj [("rule", str PARTITION_V1), ("digest", str b.partition)])]
  pure (s!"{FORMAT_CIRCUIT} {u.name}\nMETA " ++ (← pyDumps m) ++ "\n")

/-- The file after its head: each section's `CIRCUIT` line and text. -/
def emitSections {m : Type → Type} [Monad m] (put : String → m Unit) (u : UnitC) (l : Laid) : m Unit := do
  for (nm, net, lay) in [("unit", u.name, l), (SHA_NET, SHA_NET, shaLaid ()), (HM_NET, HM_NET, hmLaid ())] do
    put s!"CIRCUIT {nm} {netLines lay}\n"
    emitNet put net lay none

/-- The whole file, line by line. -/
def emitCircuit {m : Type → Type} [Monad m] (put : String → m Unit) (u : UnitC) (l : Laid) (sub : Sub) (b : Bindings) :
    Except String (m Unit) := do
  let ud ← unitDigests u l
  let head ← metaHead u l sub b (Flock.hexOf ud.sha256.finish)
  pure (do put head; emitSections put u l)

end VStar
