import PouwCheck.Inputs
import PouwWindow.Record
import PouwWindow.Side

/-!
# A unit's opening (`core_check.py`, "a unit's opening")

`opening`: the tile proof's public file opens the unit's rows under the registered window root. The unit is drawn
(unless it is opened undrawn, the filler's). The registered entries root's domain is one-stage's served domain of the
registered program and partition over the check's P entries (`windowDomain?`). The opening's entry, under its salt, is
that root's leaf at its index (`verifyPath`). The unit is one of the entry's tiles (`tileIn?`). The proof's header names one
instance whose first two rows, in key order (serde's, as C-Flock reads it), are an A row and a B row of the fp8-p schemas
(`opened`). Each of those rows' b ‖ c, at the head of the body, is the entry's side tree's leaf at the unit's row.
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
def saltBytes : Nat := 192
def bcBytes : Nat := 128

/-- `window_domain`: the window root's domain, one-stage's served domain (`registration.served_domain`) of the
registered program and partition over the check's P entries in the window's context. -/
def windowDomain? (p : Params) (c : Registration) : Except String Domain := do
  let n := p.check.entries
  let binding ← Identity.identityDigestSha512 SERVED_DOMAIN (Json.mkObj [("program", c.program),
    ("partition", c.partition), ("port", port), ("schema", entrySchema), ("leaves", n), ("run", p.context)])
  Domain.mk? .sha512 binding owner n

/-- `verify_path`: leaf `index` with siblings `sibs`, leaf level first, hashes up to `root`. -/
def verifyPath (dom : Domain) (root : ByteArray) (index : Int) (leaf : ByteArray) (sibs : List ByteArray) : Bool :=
  if !(0 ≤ index && index < dom.count) || sibs.length != PouwCommit.depth dom.count then false
  else
    let (digest, _) := sibs.zipIdx.foldl (fun (acc : ByteArray × Nat) (s, level) =>
      let (d, i) := acc
      (if i % 2 == 0 then dom.node level (i / 2) d s else dom.node level (i / 2) s d, i / 2)) (leaf, index.toNat)
    digest == root

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

/-- `[s[k] for k in sorted(s)[:2]]`. -/
def firstTwo (s : Json) : Option (List Json) := do
  let ks ← keys s
  ks.take 2 |>.mapM (pyGet? s)

/-- `opening`: none when the proof's public file `pub` opens unit `unit`'s rows under the registered window root, else
why not, `core_check.opening`'s reason. -/
def opening (p : Params) (c : Registration) (units : Option (List Int)) (unit : Int) (pub : ByteArray) (op : Json) :
    Except String (Option String) := do
  if let some us := units then
    if !us.contains unit then return some s!"unit {unit} is not drawn"
  let dom ← windowDomain? p c
  let entries := c.roots.toList.reverse.find? (·.port == port)
  if (entries.map (·.domain)) != some (Flock.hexOf dom.id) then
    return some "the registered window domain is not the window's served domain"
  let some root := entries.bind (fun r => fromHex? 64 (.str r.root))
    | return some "the registered window root is not a SHA-512 digest"
  let some entry := (pyGet? op "entry_value").bind entryOfJson? | return some "the opening names no entry"
  let some e := (pyGet? op "entry").bind strictInt? | return some "the opening names no entry"
  let salt := (pyGet? op "entry_salt").bind (fromHex? saltBytes)
  let sibs := (pyGet? op "window_path").bind pathOf?
  let sides := opened.map fun (s, _) => pyGet? op s.name
  let paths := sides.map fun o => match o with
    | some o@(.obj _) => (pyGet? o "path").bind pathOf?
    | _ => Option.none
  let indices := sides.map fun o => (o.bind (pyGet? · "index")).bind strictInt?
  let (some salt, some sibs) := (salt, sibs)
    | return some s!"the opening's salt is not {saltBytes} bytes, or a path or a side's index is not one (paths of SHA-512 digests)"
  if paths.any (·.isNone) || indices.any (·.isNone) then
    return some s!"the opening's salt is not {saltBytes} bytes, or a path or a side's index is not one (paths of SHA-512 digests)"
  let bc ← entry.bc? salt
  let leaf ← entryLeaf? dom e.toNat bc
  if !(0 ≤ e && e < dom.count && verifyPath dom root e leaf sibs) then
    return some s!"entry {e} is not the registered window's entry"
  let (i, j) ← match entry.tileIn? unit with
    | .ok ij => pure ij
    | .error x => return some s!"entry {e}: {x}"
  let some (h, body) := header? pub | return some "the proof's public file has no JSON header"
  let some n := (pyGet? h "instances").bind strictInt? | return some "the proof's header has no instance count"
  if n != 1 then return some s!"the proof has {n} instances, not one tile"
  let schemas := ((pyGet? h "frame_v3").bind (pyGet? · "schemas")).bind firstTwo
  if schemas != some (opened.map fun (_, s) => Json.str s) then
    return some "the proof's first rows are not an A row and a B row of the registered schemas"
  if body.size < 2 * bcBytes then return some "the proof's body holds no A row and B row"
  for ((side, schema), k) in opened.zipIdx do
    let row := if k == 0 then i else j
    let idx := (indices[k]!).getD 0
    if idx != (row : Int) then
      return some s!"side {side.name}: the opening is of row {idx}, not the unit's row {row}"
    let sdom ← sideDomain? dom.id side.name
    let leaf := sdom.leaf row schema (Flock.Hm96.Default512.treeLeaf (body.extract (k * bcBytes) ((k + 1) * bcBytes)))
    if !(row < sdom.count && verifyPath sdom (entry.root side) row leaf ((paths[k]!).getD [])) then
      return some s!"side {side.name}: the proof's committed row is not entry {e}'s row {row}"
  return Option.none

end PouwCheck
