import Flock.FV3
import PouwCheck.Inputs
import PouwCheck.Pass
import PouwWindow.Record
import PouwWindow.Side

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

def SERVED_DOMAIN : String := "verity/one-stage/served-domain/v1"
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

end PouwCheck
