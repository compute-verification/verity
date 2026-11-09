module

public import PouwPearlPlus.Check
public import Flock.Bytes

/-!
# The prover's openings from a pass's files (`verify_run.Opened`), and the verdict as JSON

`opening` reads each opened tile's rows, leaf and y rows from the pass's files as `verify_run.Opened` does: each file is
held to its manifest entry's size and, where the entry has one, its SHA-256 (`verify_run.Files`); rows are BF16
widened to the committed FP32 words, each followed by its hm96 salt under the manifest's salt key (nonces: the call's
index for A, 2^62 + the weight for B, 2^63 for the transcript, 2^61 + the call's index for y), and a path is read from a
tree file laid out as the device writes it (level 0 padded to a power of two, then each level above it, 64 bytes a node).
A y file that fails is the y opening's failure (`Opened.open_y`), met only once the tile has passed. A y row is the words
its file holds of it (`f32Slice`, `f32_row`'s slice): a file that ends inside a row opens a row shorter than the call's
width, as the Python's does, which `Rule.yReason` refuses where it ends before the tile's columns. A's and B's rows are
read at the call's k, a byte past the file's end as 0, and `Words.parseRow` holds each to its k words.
-/

@[expose] public section

namespace PouwCheck.PearlPlus

open Lean PouwCommit

/-- `verify_run.pow2`: `1 << (n - 1).bit_length()`. -/
def pow2 (n : Nat) : Nat := if n ≤ 1 then 1 else 2 ^ ((n - 1).log2 + 1)

/-- `verify_run.opening`: position `pos`'s path in a tree file of `count` positions. -/
def pathOf (tree : ByteArray) (count pos : Nat) : List ByteArray := Id.run do
  let mut path := []
  let mut off := 0
  let mut width := pow2 count
  let mut cur := pos
  while width > 1 do
    let s := off + (cur ^^^ 1)
    path := path ++ [tree.extract (64 * s) (64 * s + 64)]
    off := off + width
    width := width / 2
    cur := cur / 2
  return path

/-- Row `i` of a BF16 row file of width `k` as its FP32 words, little-endian. -/
def f32Row (data : ByteArray) (k i : Nat) : ByteArray := Id.run do
  let mut out := ByteArray.emptyWithCapacity (4 * k)
  let base := 2 * k * i
  for j in [0:k] do
    out := out.push 0 |>.push 0 |>.push data[base + 2 * j]! |>.push data[base + 2 * j + 1]!
  return out

/-- `verify_run.f32_row` on a y file: row `i` is the words the file holds of it (`data[2ki : 2k(i + 1)]`, widened), fewer
than `k` where the file ends inside the row or before it, so the verifier sees the row the prover's file holds and not one
padded to the call's width; half a word is numpy's refusal. -/
def f32Slice (data : ByteArray) (k i : Nat) : Except String ByteArray := do
  let row := data.extract (2 * k * i) (2 * k * (i + 1))
  if row.size % 2 != 0 then throw "ValueError: buffer size must be a multiple of element size"
  pure (f32Row row (row.size / 2) 0)

def str? (j : Json) (k : String) : Option String := match get? j k with
  | some (.str s) => some s
  | _ => none

def unhex (s : String) : Option ByteArray := Id.run do
  let cs := s.toList.toArray
  if cs.size % 2 != 0 then return none
  let v (c : Char) : Option Nat :=
    if c.isDigit then some (c.toNat - 48) else if 'a' ≤ c && c ≤ 'f' then some (c.toNat - 87) else none
  let mut out := ByteArray.empty
  for i in [0:cs.size / 2] do
    let (some h, some l) := (v cs[2 * i]!, v cs[2 * i + 1]!) | return none
    out := out.push (16 * h + l).toUInt8
  return some out

/-- The manifest entries of the files tile `tid` of pass `p` (manifest `man`) opens from. -/
def filesOf (man : Json) (p : Pass) (tid : Nat) : List Json := Id.run do
  let some (pos, _, _) := p.tile? tid | return []
  let some (.arr us) := get? man "matmuls" | return []
  let u := us[pos]!
  let w := p.records[pos]!.matmul.weight
  let wt := ((get? man "weights").bind (get? · (toString w))).getD .null
  let spec (j : Json) (k : String) : List Json := (get? j k).toList
  let y := (get? u "y").getD .null
  return spec u "rows" ++ spec u "tree" ++ spec wt "rows" ++ spec wt "tree" ++ spec man "leaves" ++ spec man "tree_t" ++
    spec y "rows" ++ spec y "tree"

/-- `verify_run.Files`: the bytes `data` read at manifest entry `spec`'s path, held to its `bytes` and, where it has one,
its `sha256`. -/
def held (spec : Json) (data : ByteArray) : Except String ByteArray := do
  let key := (str? spec "path").getD ""
  let .ok n := spec.getObjValAs? Nat "bytes" | throw "KeyError: 'bytes'"
  if data.size != n then throw s!"ValueError: {key}: {data.size} bytes; the manifest says {n}"
  if let some d := str? spec "sha256" then
    if Flock.hexOf (Flock.Sha256.hash data) != d then throw s!"ValueError: {key}: its SHA-256 is not the manifest's"
  pure data

/-- `Opened.open` and `open_y` on the pass's files: `file spec` is the file at manifest entry `spec`, `held` to it. -/
def opening (man : Json) (p : Pass) (file : Json → Except String ByteArray) (tid : Nat) : Except String TileOpening := do
  let some (pos, rows, cols) := p.tile? tid | throw "KeyError: no such tile"
  let some (.arr us) := get? man "matmuls" | throw "KeyError: 'matmuls'"
  let uj := us[pos]!
  let r := p.records[pos]!
  let u := r.matmul
  let wt := ((get? man "weights").bind (get? · (toString u.weight))).getD .null
  let some key := (str? man "hm96_salt_key").bind unhex | throw "KeyError: 'hm96_salt_key'"
  let read (j : Json) (k : String) : Except String ByteArray :=
    match get? j k with
    | some spec => if (str? spec "path").isSome then file spec else throw "KeyError: 'path'"
    | none => throw s!"KeyError: '{k}'"
  let aRows ← read uj "rows"
  let aTree ← read uj "tree"
  let bRows ← read wt "rows"
  let bTree ← read wt "tree"
  let leaves ← read man "leaves"
  let treeT ← read man "tree_t"
  let a := rows.map fun i => (⟨i, f32Row aRows u.k i ++ hm96Salt key u.index i, pathOf aTree u.m i⟩ : LeafOpening)
  let b := cols.map fun j => (⟨j, f32Row bRows u.k j ++ hm96Salt key (2 ^ 62 + u.weight) j, pathOf bTree u.n j⟩ : LeafOpening)
  let leaf : LeafOpening := ⟨tid, leaves.extract (64 * tid) (64 * tid + 64) ++ hm96Salt key (2 ^ 63) tid, pathOf treeT p.count tid⟩
  let y : Except String (List LeafOpening) := match get? uj "y" with
    | none => pure []
    | some yj => do
      let yRows ← read yj "rows"
      let yTree ← read yj "tree"
      (rows.filter (· < r.realRows)).mapM fun i => do
        pure (⟨i, (← f32Slice yRows u.n i) ++ hm96Salt key (2 ^ 61 + u.index) i, pathOf yTree r.realRows i⟩ : LeafOpening)
  pure ⟨a, b, leaf, y⟩

def jstr (s : String) : Json := .str s
def jopt (s : Option String) : Json := s.map jstr |>.getD .null

def output (ans : RunAnswer) : Json :=
  Json.mkObj [
    ("verdict", jstr (if ans.accept then "accept" else "reject")),
    ("reason", jopt ans.reason),
    ("refused", match ans.refused with | some (c, w) => .arr #[jstr c, jstr w] | none => .null),
    ("commitment", jopt (ans.plan.map (hex ·.root))),
    ("source", jopt (ans.plan.map (hex ·.source))),
    ("drawn", match ans.plan with | some pl => (pl.drawn : Nat) | none => .null),
    ("over_cap", jstr ans.overCap.name),
    ("verdicts", .arr (ans.verdicts.map fun v => Json.mkObj [("pass", (v.pass : Nat)), ("tile", (v.tile : Nat)),
      ("ok", .bool (v.ok ans.overCap)), ("reason", jopt v.reason)]).toArray),
    ("work", match ans.work with
      | some c => Json.mkObj [("credited", (c.credited : Nat)), ("credited_tiles", (c.creditedTiles : Nat)),
          ("macs", (c.macs : Nat)), ("rho", jstr (toString c.rho)), ("epsilon", jstr (toString c.epsilon)),
          ("over_cap_draws", (c.overCapDraws : Nat))]
      | none => .null),
    ("tie", match ans.tie with
      | some t => Json.mkObj (([("record", jstr t.record), ("tables", (t.tables : Nat))] : List (String × Json)) ++
          (t.weights.map fun u => [("weights", jstr u)]).getD [] ++
          (t.load.map fun c => [("load", Json.arr #[.num ⟨c.slots, 0⟩, .num ⟨c.prompt, 0⟩, .num ⟨c.outputs, 0⟩])]).getD [])
      | none => .null)]

end PouwCheck.PearlPlus
