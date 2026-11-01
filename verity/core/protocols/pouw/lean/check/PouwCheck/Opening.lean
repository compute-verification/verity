import Flock.FV3
import PouwCheck.Inputs
import PouwCheck.Pass
import PouwWindow.Record
import PouwWindow.Side
import Identity.Domains

/-!
# A unit's opening (`core_check.py`, "a unit's opening")

`opening`: the tile proof's public file opens the unit's rows under the registered window root. The unit is drawn
(unless it is opened undrawn, the filler's). A certified window (its context names a PoUW window, `windowOf`) opens only
a salted template's units (`Tile.salted`), under the 64-byte certified salt the verifier holds, and any other window no
salt and no salted unit (`saltReason`). The registered entries root's domain is one-stage's served domain of the
registered program and partition over the check's P entries (`windowDomain?`). The opening's entry, under its salt, is
that root's leaf at its index (`verifyPath`). The unit is one of the entry's tiles (`tileIn?`). The proof's header names one
instance, over the circuit the verifier pins for the check's template (`tile?`), whose first two ports, `a0` and `b0`
(`PC8_PORTS`), are an A row and a B row of the fp8-p schemas (`opened`), read by name. Each of those rows' b ‖ c, at the head of the body, is the entry's side tree's leaf at the unit's row.
C-Flock reads port k's b ‖ c at the body's k-th place, in the circuit's port order, so the header's keys, in key order
(serde's), must list the opened side rows first and in port order (`portsOf`): a header that orders them otherwise is
refused, never read in its own order.
A salted unit's tile (i, j) is A rows R·i + k and B rows C·j + k, each opened the same way, and the b ‖ c after them is
its public row's, which must be `publicBc` of the window's context, the certified salt, the unit's position and the
entry's side roots.
Python's reading is kept where its outcome depends on it: hex as `bytes.fromhex` reads it, integers as `struct`
packs them (a boolean is one), an entry as `Entry.from_json` reads it.
-/

namespace PouwCheck

open Lean (Json)
open PouwWindow
open PouwCommit (Domain)

def SERVED_DOMAIN : String := Domains.ONE_STAGE_SERVED_DOMAIN
/-- A tile proof's first two rows: its A row and its B row, each side's fp8-p row schema. -/
def opened : List (Side × String) := [(.a, Flock.HmRow.ROW_SCHEMA_SEG), (.b, Flock.HmRow.ROW_SCHEMA_V2)]
/-- The ports of those rows, the pinned circuit's first two (`served_zk/stage_tile.py`'s `in_rows`). -/
def PC8_PORTS : List String := ["a0", "b0"]

/-- `_ports`: the side rows a tile proof opens, by port name in its circuit's port order: A rows, then B rows. -/
def portsOf : Option Salted → List String
  | Option.none => PC8_PORTS
  | some l => (List.range l.r).map (s!"a{·}") ++ (List.range l.c).map (s!"b{·}")
def saltBytes : Nat := 192
def bcBytes : Nat := 128
/-- `PUBLIC_ROW_BITS`: a salted unit's public row, the certified salt, the unit's position (a u64) and two roots. -/
def PUBLIC_ROW_BITS : Nat := 8 * (SALT_BYTES + 8 + 2 * 64)
/-- `PUBLIC_ROW_J10R_BITS`: a J10-R unit's public row, `PUBLIC_ROW_BITS` and then its call's first position and its
two blocks, each a u64. -/
def PUBLIC_ROW_J10R_BITS : Nat := PUBLIC_ROW_BITS + 8 * 3 * 8
def PUBLIC_ROW_SALT_DOMAIN : ByteArray := "verity/pouw/public-row-salt/v1".toUTF8.push 0

/-- `public_row`: the salt, the position as a u64 big-endian and the two roots (and, with `j10r` = (call, b₀, b₁), a
J10-R unit's call and blocks, each a u64 big-endian), each 4 bytes reversed (a u32 low bit first, as
`ml.circuits.seeds` lays the bits out). -/
def publicRow (salt : ByteArray) (position : Nat) (roots : ByteArray × ByteArray)
    (j10r : Option (Nat × Nat × Nat) := none) : ByteArray :=
  let tail := match j10r with
    | some (c, b0, b1) => PouwCommit.be8 c ++ PouwCommit.be8 b0 ++ PouwCommit.be8 b1
    | none => ByteArray.empty
  let x := salt ++ PouwCommit.be8 position ++ roots.1 ++ roots.2 ++ tail
  ByteArray.mk ((Array.range x.size).map fun i => x.get! (4 * (i / 4) + 3 - i % 4))

/-- `public_row_salt`: three SHA-512 blocks of the domain ‖ u64be(len) ‖ context ‖ u64be(position) ‖ u8(i). -/
def publicRowSalt (context : String) (position : Nat) : ByteArray :=
  let ctx := context.toUTF8
  let head := PUBLIC_ROW_SALT_DOMAIN ++ PouwCommit.be8 ctx.size ++ ctx ++ PouwCommit.be8 position
  (List.range (saltBytes / 64)).foldl (fun acc i => acc ++ Flock.Sha512.hash (head.push i.toUInt8)) ByteArray.empty

/-- `public_bc`: the public row's b ‖ c, hm96-sha512 over its `sha512/row/v2` digest under `publicRowSalt`. -/
def publicBc (context : String) (salt : ByteArray) (position : Nat) (roots : ByteArray × ByteArray) : ByteArray :=
  Flock.Hm96.Default512.commitString
    (Flock.Sha512.hash (Flock.HmRow.rowPrefixV2 PUBLIC_ROW_BITS ++ publicRow salt position roots))
    (publicRowSalt context position)

/-- `_salt_reason`: a certified window opens only a salted template's units, under a 64-byte salt; any other window, no
salt and no salted unit. -/
def saltReason (p : Params) (t : Option Tile) (salt : Option ByteArray) : Option String :=
  let salted := t.bind (·.salted)
  if (windowOf p.context).isSome then
    if salted.isNone then
      some "the window is certified, and its template's unit reads no salt: no drawn proof of it is held to the window's certified salt"
    else if (salt.filter (·.size == SALT_BYTES)).isNone then
      some s!"the window is certified, and the verifier holds no {SALT_BYTES}-byte certified salt for it"
    else none
  else if salt.isSome then some "the window is not certified, so it has no certified salt for a unit to read"
  else if salted.isSome then some "the window's template's unit reads a salt, which only a certified window has"
  else none

/-- `window_domain`: the window root's domain, one-stage's window domain (`registration.window_domain`) of the
registered program and partition over the check's P entries in the window's context. -/
def windowDomain? (p : Params) (c : Registration) : Except String Domain := do
  let n := p.check.entries
  let binding ← Identity.identityDigestSha512 SERVED_DOMAIN (Json.mkObj [("program", c.program),
    ("partition", c.partition), ("port", PouwWindow.port), ("schema", entrySchema), ("leaves", n), ("run", p.context)])
  Domain.mk? .sha512 binding owner n

/-- `verify_path`: leaf `index` with siblings `sibs`, leaf level first, climbs to `root` (`Flock.FV3.Sha512.climbL`; the
check's domains are SHA-512). -/
def verifyPath (dom : Domain) (root : ByteArray) (index : Int) (leaf : ByteArray) (sibs : List ByteArray) : Bool :=
  if !(0 ≤ index && index < dom.count) || sibs.length != Flock.log2ceil dom.count then false
  else Flock.FV3.Sha512.climbL dom.id leaf index.toNat 0 sibs == root

/-- `_fromhex`: `bytes.fromhex` of a string of `n` bytes' hex. -/
def fromHex? (n : Nat) : Json → Option ByteArray
  | .str s => match PouwCommit.pyFromHex s with
    | .ok b => if b.size == n then some b else none
    | .error _ => none
  | _ => none

/-- `_path`: a list of SHA-512 digests' hex. -/
def pathOf? : Json → Option (List ByteArray)
  | .arr a => a.toList.mapM (fromHex? 64)
  | _ => none

/-- `_sides`: a salted unit's `n` rows of one side, `[{"index", "path"}, …]`. -/
def sidesOf? (o : Option Json) (n : Nat) : Option (List (Int × List ByteArray)) := do
  let some (.arr a) := o | Option.none
  if a.size != n then Option.none
  a.toList.mapM fun x => match x with
    | .obj _ => do pure (← (pyGet? x "index").bind strictInt?, ← (pyGet? x "path").bind pathOf?)
    | _ => Option.none

/-- `Entry.from_json(x)` with its `row()`: every field present, the kind's name, the digests' hex of 64 bytes, the
integers what `struct` packs as u64s. -/
def entryOfJson? (x : Json) : Option Entry := do
  let int (k : String) : Option Nat :=
    ((pyGet? x k).bind pyInt?).bind fun i => if 0 ≤ i && i < 2 ^ 64 then some i.toNat else Option.none
  let hex (k : String) : Option ByteArray := (pyGet? x k).bind (fromHex? 64)
  let kind ← match pyGet? x "kind" with
    | some (.str s) => Kind.ofName? s
    | _ => Option.none
  let (lo, hi, nA, nB, tile0, cols, m) := (← int "lo", ← int "hi", ← int "n_a", ← int "n_b", ← int "tile0", ← int "cols",
    ← int "m")
  let (rootA, rootB, rootH3, record) := (← hex "root_a", ← hex "root_b", ← hex "root_h3", ← hex "record")
  pure { kind, lo, hi, nA, nB, rootA, rootB, rootH3, record, tile0, cols, m }

/-- The header's object, as `json.loads` reads the bytes before the body's first newline (UTF-8, no `NaN`). -/
def header? (pub : ByteArray) : Option (Json × ByteArray) := do
  let nl ← (List.range pub.size).find? (pub.get! · == 10)
  let s ← String.fromUTF8? (pub.extract 0 nl)
  let h ← (Json.parse s).toOption
  match h with
  | .obj _ => some (h, pub.extract (nl + 1) pub.size)
  | _ => Option.none

/-- `opening`: none when the proof's public file `pub` opens unit `unit`'s rows under the registered window root, else
why not, `core_check.opening`'s reason. `salt` is a certified window's salt (`certified_salt`). -/
def opening (p : Params) (c : Registration) (units : Option (List Int)) (unit : Int) (pub : ByteArray) (op : Json)
    (salt : Option ByteArray := Option.none) : Except String (Option String) := do
  if let some us := units then
    if !us.contains unit then return some s!"unit {unit} is not drawn"
  let t := tile? p.check.template
  if let some why := saltReason p t salt then return some why
  let lay := t.bind (·.salted)
  let dom ← windowDomain? p c
  let entries := c.roots.toList.reverse.find? (·.port == PouwWindow.port)
  if (entries.map (·.domain)) != some (Flock.hexOf dom.id) then
    return some "the registered window domain is not the window's served domain"
  let some root := entries.bind (fun r => fromHex? 64 (.str r.root))
    | return some "the registered window root is not a SHA-512 digest"
  let some entry := (pyGet? op "entry_value").bind entryOfJson? | return some "the opening names no entry"
  let some e := (pyGet? op "entry").bind strictInt? | return some "the opening names no entry"
  let esalt := (pyGet? op "entry_salt").bind (fromHex? saltBytes)
  let sibs := (pyGet? op "window_path").bind pathOf?
  let rows : Option (List (List (Int × List ByteArray))) := match lay with
    | Option.none => opened.mapM fun (s, _) => match pyGet? op s.name with
      | some o@(.obj _) => do pure [(← (pyGet? o "index").bind strictInt?, ← (pyGet? o "path").bind pathOf?)]
      | _ => Option.none
    | some l => [sidesOf? (pyGet? op "a") l.r, sidesOf? (pyGet? op "b") l.c].mapM id
  let (some esalt, some sibs, some rows) := (esalt, sibs, rows)
    | return some s!"the opening's salt is not {saltBytes} bytes, or a path or a side's index is not one (paths of SHA-512 digests)"
  let bc ← entry.bc? esalt
  let leaf ← entryLeaf? dom e.toNat bc
  if !(0 ≤ e && e < dom.count && verifyPath dom root e leaf sibs) then
    return some s!"entry {e} is not the registered window's entry"
  let (i, j) ← match entry.tileIn? unit with
    | .ok ij => pure ij
    | .error x => return some s!"entry {e}: {x}"
  let wanted : List (List Nat) := match lay with
    | Option.none => [[i], [j]]
    | some l => [(List.range l.r).map (l.r * i + ·), (List.range l.c).map (l.c * j + ·)]
  let some (h, body) := header? pub | return some "the proof's public file has no JSON header"
  let some n := (pyGet? h "instances").bind strictInt? | return some "the proof's header has no instance count"
  if n != 1 then return some s!"the proof has {n} instances, not one tile"
  unless t.any (fun t => pyGet? h "circuit_sha512" == some (.str t.circuit)) do
    return some "the proof's circuit is not the one the verifier pins for the check's template"
  let fv := (pyGet? h "frame_v3").bind (pyGet? · "schemas")
  let ports := portsOf lay
  match lay with
  | Option.none =>
    let ok : Bool := match fv with
      | some s@(.obj _) => ports.map (pyGet? s) == opened.map fun (_, x) => some (Json.str x)
      | _ => false
    if !ok then return some "the proof's first rows are not an A row and a B row of the registered schemas"
    if body.size < 2 * bcBytes then return some "the proof's body holds no A row and B row"
  | some l =>
    let names := ports ++ ["pub"]
    let bad : Bool := match fv with
      | some s@(.obj _) => names.any fun x => pyGet? s x != some (.str Flock.HmRow.ROW_SCHEMA_V2)
      | _ => true
    if bad then return some s!"the proof's rows {", ".intercalate names} are not {Flock.HmRow.ROW_SCHEMA_V2} rows"
    if body.size < (l.r + l.c + 1) * bcBytes then
      return some "the proof's body holds no A rows, B rows and public row"
  if (fv.bind keys).map (·.take ports.length) != some ports then
    return some s!"the proof's header does not list its rows {", ".intercalate ports} first, in its circuit's port order"
  let mut k := 0
  for ((side, schema), (wants, opens)) in opened.zip (wanted.zip rows) do
    let sdom ← sideDomain? dom.id side.name
    for (want, (idx, path)) in wants.zip opens do
      if idx != (want : Int) then
        return some s!"side {side.name}: the opening is of row {idx}, not the unit's row {want}"
      let schema := if lay.isNone then schema else Flock.HmRow.ROW_SCHEMA_V2
      let leaf := sdom.leaf want schema (Flock.Hm96.Default512.treeLeaf (body.extract (k * bcBytes) ((k + 1) * bcBytes)))
      if !(want < sdom.count && verifyPath sdom (entry.root side) want leaf path) then
        return some s!"side {side.name}: the proof's committed row is not entry {e}'s row {want}"
      k := k + 1
  if lay.isSome && body.extract (k * bcBytes) ((k + 1) * bcBytes) !=
      publicBc p.context (salt.getD ByteArray.empty) unit.toNat (entry.rootA, entry.rootB) then
    return some s!"the proof's public row is not the window's certified salt, unit {unit}'s position and entry {e}'s side roots"
  return Option.none

/-! Each separator below keeps its literal while a lock reads its definition, whose record hashes the value as
written: tied here to its entry of the table (`Identity.Domains`), it reads the entry at its next record change. -/
example : PUBLIC_ROW_SALT_DOMAIN = Domains.POUW_PUBLIC_ROW_SALT.toUTF8.push 0 := rfl

end PouwCheck

/-!
# The lead's window (`circuit/lead.py`, "the window")

The lead (`verity/pouw/v2`) registers before its window's salt: its 64-byte record in the certifier's `commit_weights`
slot, in `pouwRecord`'s place, is `leadRecord?`, SHA-512 of `RECORD_TAG ‖ root(weights) ‖ stream ‖ prover ‖ keyLeaf ‖
u64be(calls)`, `calls` its calls a forward.
Each call's A and each forward's words are an instance at a position of their own: A_j of forward f at f · calls + j
(`aPosition`), the forward's words at 2^63 + f (`wordsPosition`). An instance is a frame-v3-sha512 tree bound to salt_w
and its position (`instanceDomain?`), leaf r at rank and position r: A_j's rows · K bytes, row-major, in `A_SEGMENT`-byte
leaves of schema `A_SCHEMA`, or each of the forward's calls' committed words in `CHUNK_WORDS`-word leaves of schema
`WORDS_SCHEMA`, a call's leaves after the call before it's. A call's committed words are the kernel's export of its MMA
tiles in (row tile, column tile, step, column, row) order (`Shape.tile`), every tile past its padded rows or columns
skipped, and a tile's 128 words in wgmma's fragment order (`fragment`). So a unit's opening (`unitOpening`) is a leaf and
an offset for each step the unit reads, and an A run for each of its rows.

`windowRefusal` reads the certifier's record of the window. The record opens to the window's weights' root, stream,
prover, keyLeaf and call count. salt_w is drawn once, before any A root. Forwards are serial: a forward opens at its
first A root, and its words root closes it under the certifier's verdict on its egress (`egressRefused`, its count kept
under the certifier's boundary hold, `forwardCount`). Each call has one position and a shape the verifier admits
(`Shape.refuse`).
-/

namespace PouwCheck.Lead

open Lean (Json)
open PouwWindow (pyGet? strictInt? bitLength)
open PouwCommit (Domain)

/-- The lead's registration record's tag. -/
def RECORD_TAG : ByteArray := "verity/pouw-lead-record/v1".toUTF8
/-- An instance's tree binding's tag, and its leaves' schemas. -/
def INSTANCE_TAG : ByteArray := "pouw/lead/instance/v1".toUTF8
def WORDS_SCHEMA : String := "pouw/lead/words/v1"
def A_SCHEMA : String := "pouw/lead/a/v1"
/-- The first words instance's position: A_j's instances are below it, a forward's words at it and above. -/
def WORDS_POSITION : Nat := 2 ^ 63
/-- A words leaf's words (512 MMA tiles), and an A leaf's bytes. -/
def CHUNK_WORDS : Nat := 2 ^ 16
def A_SEGMENT : Nat := 2 ^ 10
/-- An MMA tile's rows and columns, the kernel's row tile and the column tiles it runs a call at. -/
def TM : Nat := 16
def TN : Nat := 8
def TILE_WORDS : Nat := TM * TN
def ROW_TILE : Nat := 64
def BNS : List Int := [128, 256]
/-- k32 steps a unit, and data entries of K at step 0 and at each later step. -/
def BLOCK : Nat := 32
def LANES0 : Nat := 21
def LANES : Nat := 28
/-- The calls the verifier admits: K from 1 to 2^16, at most 2^15 rows and 2^20 columns. -/
def MAX_K : Nat := 2 ^ 16
def MAX_ROWS : Nat := 2 ^ 15
def MAX_COLS : Nat := 2 ^ 20
/-- The certifier's verdict on a forward within its egress bits; any other is a refusal. -/
def ACCEPTED : String := "accepted"
def DIGEST : Nat := 64

/-- `lead_record`: SHA-512(`RECORD_TAG` ‖ root(weights) ‖ stream ‖ prover ‖ keyLeaf ‖ u64be(calls)), every part
64 bytes and 1 ≤ calls < 2^63. -/
def leadRecord? (weightsRoot stream prover keyLeaf : ByteArray) (calls : Int) : Except String ByteArray :=
  let parts := [weightsRoot, stream, prover, keyLeaf]
  if parts.any (·.size != DIGEST) || !(1 ≤ calls && calls < (WORDS_POSITION : Int)) then
    throw "a record binds a 64-byte weights root, stream, prover and keyLeaf and 1 .. 2^63 - 1 calls a forward"
  else pure (Flock.Sha512.hash (parts.foldl (· ++ ·) RECORD_TAG ++ PouwCommit.be8 calls.toNat))

/-- `position_refusal`: why call `call` of forward `forward` has no position in a window of `calls` calls a forward. -/
def positionRefusal (forward calls call : Int) : Option String :=
  if 0 ≤ call && call < calls && 0 ≤ forward && forward * calls + call < (WORDS_POSITION : Int) then none
  else some s!"call {call} of forward {forward} has no A position in a window of {calls} calls a forward"

/-- `a_position`: A_j's position, f · calls + j, below every words position. -/
def aPosition (forward calls call : Int) : Except String Nat :=
  match positionRefusal forward calls call with
  | some why => throw why
  | none => pure (forward * calls + call).toNat

/-- `words_position`: the forward's words position, 2^63 + f. -/
def wordsPosition (forward : Int) : Except String Nat :=
  if 0 ≤ forward && forward < (WORDS_POSITION : Int) then pure (WORDS_POSITION + forward.toNat)
  else throw s!"forward {forward} has no words position"

/-- `instance_binding`: SHA-512(`INSTANCE_TAG` ‖ salt_w ‖ u64be(position)). -/
def instanceBinding? (salt : ByteArray) (position : Int) : Except String ByteArray :=
  if salt.size != DIGEST || !(0 ≤ position && position < (2 ^ 64 : Int)) then throw "a 64-byte salt_w and a u64 position"
  else pure (Flock.Sha512.hash (INSTANCE_TAG ++ salt ++ PouwCommit.be8 position.toNat))

/-- `instance_domain`: the instance's tree domain over `count` leaves, a boundary domain (owner -1). -/
def instanceDomain? (salt : ByteArray) (position : Int) (count : Nat) : Except String Domain := do
  Domain.mk? .sha512 (← instanceBinding? salt position) (-1) count

/-- `steps_for`: 1 + ⌈(K - 21) / 28⌉, K - 21 truncated at 0. -/
def stepsFor (K : Nat) : Nat := 1 + (K - LANES0 + LANES - 1) / LANES

/-- `k_range`: step t's data entries [lo, hi) of K, [0, 21) at step 0 and then 28 a step. -/
def kRange (t K : Nat) : Nat × Nat :=
  let lo := if t == 0 then 0 else LANES0 + LANES * (t - 1)
  (min lo K, min (lo + if t == 0 then LANES0 else LANES) K)

/-- A call's public shape: its rows of A, output columns, inner dimension K, and the column tile `bn` the kernel ran it
at, which orders its committed words. -/
structure Shape where
  rows : Int
  cols : Int
  K : Int
  bn : Int
  deriving BEq, Repr

namespace Shape

/-- Why the verifier doesn't admit the call, else none. The rest of `Shape` reads an admitted call. -/
def refuse (s : Shape) : Option String :=
  if !(1 ≤ s.K && s.K ≤ (MAX_K : Int)) then
    some s!"K = {s.K} is outside [1, {MAX_K}], the inner dimensions the verifier admits"
  else if !(1 ≤ s.rows && s.rows ≤ (MAX_ROWS : Int) && 1 ≤ s.cols && s.cols ≤ (MAX_COLS : Int)) then
    some s!"a call of {s.rows} rows and {s.cols} columns is outside the [1, {MAX_ROWS}] rows and [1, {MAX_COLS}] columns the verifier admits"
  else if !BNS.contains s.bn then
    some s!"the kernel runs a call at a column tile of {" or ".intercalate (BNS.map toString)}, not {s.bn}"
  else none

/-- Its rows and columns padded to the MMA grid, and its steps. -/
def m (s : Shape) : Nat := TM * ((s.rows.toNat + TM - 1) / TM)
def n (s : Shape) : Nat := TN * ((s.cols.toNat + TN - 1) / TN)
def S (s : Shape) : Nat := stepsFor s.K.toNat

/-- Its committed words' leaves, ⌈m n S / `CHUNK_WORDS`⌉, and A_j's, ⌈rows K / `A_SEGMENT`⌉. -/
def chunks (s : Shape) : Nat := (s.m * s.n * s.S + CHUNK_WORDS - 1) / CHUNK_WORDS
def segments (s : Shape) : Nat := (s.rows.toNat * s.K.toNat + A_SEGMENT - 1) / A_SEGMENT

/-- Its MACs a token, rows · n · K: its share of its forward's egress bits. -/
def macs (s : Shape) : Nat := s.rows.toNat * s.n * s.K.toNat

/-- The index in its committed words of the MMA tile at row R0, column C0 and step t: row tile rt of 64 rows, column
tile ct of `bn` columns, and in them the tile's column ni and row w, in (rt, ct, t, ni, w) order, each row or column tile
holding only its tiles within m and n. -/
def tile (s : Shape) (R0 C0 t : Nat) : Nat :=
  let q := ROW_TILE / TM
  let (R, Cn, nni) := (s.m / TM, s.n / TN, s.bn.toNat / TN)
  let (rt, w, ct, ni) := (R0 / TM / q, R0 / TM % q, C0 / TN / nni, C0 / TN % nni)
  let (rw, cw) := (min q (R - rt * q), min nni (Cn - ct * nni))
  rt * q * Cn * s.S + rw * (nni * ct * s.S + cw * t + ni) + w

end Shape

/-- Word (r, c) of an MMA tile's 128 committed words, in wgmma's fragment order: 16 g + 4 q + 2 eh + el, with
r = 8 eh + g and c = 2 q + el. -/
def fragment (r c : Nat) : Nat := 16 * (r % 8) + 4 * (c / 2) + 2 * (r / 8) + c % 2

/-- `Unit.refuse`: why the verifier has no unit at row `R0`, column `C0` and block `BLK` holding `ROWS` × `COLS` of a
call at K. -/
def unitRefusal (K R0 C0 ROWS COLS BLK : Int) : Option String :=
  let S : Int := stepsFor K.toNat
  if !(1 ≤ K && K ≤ (MAX_K : Int)) then
    some s!"K = {K} is outside [1, {MAX_K}], the inner dimensions the verifier admits"
  else if R0 < 0 || R0 % (TM : Int) != 0 || C0 < 0 || C0 % (TN : Int) != 0 then
    some s!"the tile at row {R0}, column {C0} is not on the {TM} × {TN} grid"
  else if R0 + (TM : Int) > (MAX_ROWS : Int) || C0 + (TN : Int) > (MAX_COLS : Int) then
    some s!"the tile at row {R0}, column {C0} passes the {MAX_ROWS} rows and {MAX_COLS} columns the verifier admits"
  else if !(1 ≤ ROWS && ROWS ≤ (TM : Int) && 1 ≤ COLS && COLS ≤ (TN : Int)) then
    some s!"a tile holds 1 .. {TM} of the call's rows and 1 .. {TN} of its columns, not {ROWS} and {COLS}"
  else if !(0 ≤ (BLOCK : Int) * BLK && (BLOCK : Int) * BLK < S) then
    some s!"block {BLK} starts past the call's {S} steps"
  else none

/-- A unit's opening: each step it reads as (step, leaf, first word), its rows' A runs [lo, hi), and their A leaves. -/
structure UnitOpening where
  words : List (Nat × Nat × Nat)
  a : List (Nat × Nat)
  segments : List Nat

/-- `unit_opening`: where call `call` of a forward of `shapes` keeps what its unit at row `R0`, column `C0` and block
`BLK` reads. For each step it reads (the step before the block on a block after the first, then the block's), the leaf
of the forward's words instance holding its tile and the tile's first word there, its word (r, c) at that plus
`fragment r c`. For each of its rows, A_j's bytes at the block's K entries, and the A leaves holding them. Refused: a
forward holding a shape the verifier doesn't admit, a call not in it, or a unit the verifier doesn't have. -/
def unitOpening (shapes : List Shape) (call R0 C0 BLK : Int) : Except String UnitOpening := do
  for (s, k) in shapes.zipIdx do
    if let some why := s.refuse then throw s!"call {k}: {why}"
  let some sh := (if 0 ≤ call then shapes[call.toNat]? else none)
    | throw s!"call {call} is not one of the forward's {shapes.length}"
  let ROWS := min (sh.rows - R0) (TM : Int)
  if let some why := unitRefusal sh.K R0 C0 ROWS (min (sh.cols - C0) (TN : Int)) BLK then throw why
  let (K, R0, C0, ROWS, BLK) := (sh.K.toNat, R0.toNat, C0.toNat, ROWS.toNat, BLK.toNat)
  let t0 := BLOCK * BLK
  let t1 := min (t0 + BLOCK) sh.S
  let k0 := (kRange t0 K).1
  let kb := (kRange (t1 - 1) K).2 - k0
  let base := ((shapes.take call.toNat).map Shape.chunks).sum
  let tiles := CHUNK_WORDS / TILE_WORDS
  let first := if BLK > 0 then t0 - 1 else t0
  let words := (List.range (t1 - first)).map fun d =>
    let x := sh.tile R0 C0 (first + d)
    (first + d, base + x / tiles, x % tiles * TILE_WORDS)
  let a := (List.range ROWS).map fun r => ((R0 + r) * K + k0, (R0 + r) * K + k0 + kb)
  let segments := (a.flatMap fun (lo, hi) =>
    (List.range ((hi - 1) / A_SEGMENT + 1 - lo / A_SEGMENT)).map (lo / A_SEGMENT + ·)).eraseDups
  pure { words, a, segments }

/-- `egress_refused`: whether a forward's egress count passes its bits, Σ cap · rows · n · K / fwdMacs over its calls,
compared in integers. -/
def egressRefused (count : Nat) (shapes : List Shape) (cap fwdMacs : Nat) : Bool :=
  count * fwdMacs > cap * (shapes.map Shape.macs).sum

/-- `forward_count`: a forward's egress count as the certifier keeps it. Each bucket a boundary link ticks while the
forward is open, at allocation r with its real frames' content bits, counts ⌈log2(r + 1)⌉ and those bits (the hold keeps
r = 0, so a held bucket counts nothing), and the release at the boundary, at allocation `release`, counts its
⌈log2(r + 1)⌉ once. -/
def forwardCount (buckets : List (Nat × List Nat)) (release : Nat) : Nat :=
  (buckets.map fun (r, frames) => bitLength r + frames.sum).sum + bitLength release

/-- `window_refusal`: why the verifier refuses the lead's window from the certifier's record, else none. `record` is the
window's signed `commit_weights` record, which must be `leadRecord?` of its weights' root, stream, prover, keyLeaf and
call count, and `events` the certifier's record in its order: `{"kind": "salt"}` (salt_w drawn, once), `{"kind":
"a_root", "forward", "call", "rows", "cols", "K", "bn"}` (a call's A root and shape, which opens its forward), or
`{"kind": "words_root", "forward", "verdict"}` (the forward's words root, which closes it, and the certifier's verdict
on its egress). -/
def windowRefusal (record weightsRoot stream prover keyLeaf : ByteArray) (calls : Int) (events : List Json) :
    Option String := Id.run do
  if !(1 ≤ calls && calls < (WORDS_POSITION : Int)) then return some s!"a window of {calls} calls a forward"
  if [record, weightsRoot, stream, prover, keyLeaf].any (·.size != DIGEST) then
    return some "the record, the weights' root, the stream, the prover and keyLeaf are 64 bytes each"
  unless (leadRecord? weightsRoot stream prover keyLeaf calls).toOption == some record do
    return some "the window's record does not open to its weights' root, stream, prover, keyLeaf and call count"
  let mut salted := false
  let mut opened : Option Int := none
  let mut seen : List (Int × Int) := []
  let mut closed : List Int := []
  for (e, k) in events.zipIdx do
    let int (x : String) : Option Int := (pyGet? e x).bind strictInt?
    match pyGet? e "kind" with
    | some (.str "salt") =>
      if salted then return some s!"event {k}: a second salt_w"
      salted := true
    | some (.str "a_root") =>
      let some (f, j, rows, cols, K, bn) :=
          (do pure (← int "forward", ← int "call", ← int "rows", ← int "cols", ← int "K", ← int "bn") : Option _)
        | return some s!"event {k}: the a_root event's fields are malformed"
      if !salted then return some s!"event {k}: forward {f}'s call {j} has an A root recorded before salt_w"
      if let some why := (positionRefusal f calls j).orElse fun _ => Shape.refuse ⟨rows, cols, K, bn⟩ then
        return some s!"event {k}: {why}"
      if let some o := opened then
        if o != f then return some s!"event {k}: forward {f}'s A root while forward {o} is open: the forwards overlap"
      if closed.contains f then return some s!"event {k}: forward {f} is closed"
      if seen.contains (f, j) then return some s!"event {k}: call {j} of forward {f} has an A root already"
      seen := (f, j) :: seen
      opened := some f
    | some (.str "words_root") =>
      let (some f, some (.str verdict)) := (int "forward", pyGet? e "verdict")
        | return some s!"event {k}: the words_root event's fields are malformed"
      if opened != some f then return some s!"event {k}: a words root of forward {f}, which is not open"
      if verdict != ACCEPTED then return some s!"event {k}: the certifier refused forward {f}'s egress: {verdict}"
      closed := f :: closed
      opened := none
    | _ => return some s!"event {k}: not a salt, an A root or a words root"
  if let some f := opened then return some s!"forward {f} is open when the record ends"
  return none

end PouwCheck.Lead
