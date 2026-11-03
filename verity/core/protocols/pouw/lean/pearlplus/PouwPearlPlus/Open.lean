import PouwPearlPlus.Check

/-!
# The prover's openings from a pass's files (`verify_run.Opened`), and the verdict as JSON

`opening` reads each opened tile's rows, leaf and y rows from the pass's files as `verify_run.Opened` does: rows are BF16
widened to the committed FP32 words, each followed by its hm96 salt under the manifest's salt key (nonces: the call's
index for A, 2^62 + the weight for B, 2^63 for the transcript, 2^61 + the call's index for y), and a path is read from a
tree file laid out as the device writes it (level 0 padded to a power of two, then each level above it, 64 bytes a node).
-/

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

/-- The files tile `tid` of pass `p` (manifest `man`) opens from, by their manifest paths. -/
def filesOf (man : Json) (p : Pass) (tid : Nat) : List String := Id.run do
  let some (pos, _, _) := p.tile? tid | return []
  let some (.arr us) := get? man "matmuls" | return []
  let u := us[pos]!
  let w := p.records[pos]!.matmul.weight
  let wt := ((get? man "weights").bind (get? · (toString w))).getD .null
  let spec (j : Json) (k : String) : List String := ((get? j k).bind (str? · "path")).toList
  let y := (get? u "y").getD .null
  return spec u "rows" ++ spec u "tree" ++ spec wt "rows" ++ spec wt "tree" ++ spec man "leaves" ++ spec man "tree_t" ++
    spec y "rows" ++ spec y "tree"

/-- `Opened.open` and `open_y` on the pass's files `file`. -/
def opening (man : Json) (p : Pass) (file : String → Except String ByteArray) (tid : Nat) : Except String TileOpening := do
  let some (pos, rows, cols) := p.tile? tid | throw "KeyError: no such tile"
  let some (.arr us) := get? man "matmuls" | throw "KeyError: 'matmuls'"
  let uj := us[pos]!
  let r := p.records[pos]!
  let u := r.matmul
  let wt := ((get? man "weights").bind (get? · (toString u.weight))).getD .null
  let some key := (str? man "hm96_salt_key").bind unhex | throw "KeyError: 'hm96_salt_key'"
  let path (j : Json) (k : String) : Except String String :=
    match (get? j k).bind (str? · "path") with
    | some s => pure s
    | none => throw s!"KeyError: '{k}'"
  let aRows ← file (← path uj "rows")
  let aTree ← file (← path uj "tree")
  let bRows ← file (← path wt "rows")
  let bTree ← file (← path wt "tree")
  let leaves ← file (← path man "leaves")
  let treeT ← file (← path man "tree_t")
  let a := rows.map fun i => (⟨i, f32Row aRows u.k i ++ hm96Salt key u.index i, pathOf aTree u.m i⟩ : Opening)
  let b := cols.map fun j => (⟨j, f32Row bRows u.k j ++ hm96Salt key (2 ^ 62 + u.weight) j, pathOf bTree u.n j⟩ : Opening)
  let leaf : Opening := ⟨tid, leaves.extract (64 * tid) (64 * tid + 64) ++ hm96Salt key (2 ^ 63) tid, pathOf treeT p.count tid⟩
  let y ← match get? uj "y" with
    | none => pure []
    | some yj => do
      let yRows ← file (← path yj "rows")
      let yTree ← file (← path yj "tree")
      pure ((rows.filter (· < r.realRows)).map fun i =>
        (⟨i, f32Row yRows u.n i ++ hm96Salt key (2 ^ 61 + u.index) i, pathOf yTree r.realRows i⟩ : Opening))
  pure ⟨a, b, leaf, y⟩

def jstr (s : String) : Json := .str s
def jopt (s : Option String) : Json := s.map jstr |>.getD .null

def output (ans : Answer) : Json :=
  Json.mkObj [
    ("verdict", jstr (if ans.accept then "accept" else "reject")),
    ("reason", jopt ans.reason),
    ("refused", match ans.refused with | some (c, w) => .arr #[jstr c, jstr w] | none => .null),
    ("commitment", jopt (ans.plan.map (hex ·.root))),
    ("source", jopt (ans.plan.map (hex ·.source))),
    ("drawn", match ans.plan with | some pl => (pl.drawn : Nat) | none => .null),
    ("verdicts", .arr (ans.verdicts.map fun v => Json.mkObj [("pass", (v.pass : Nat)), ("tile", (v.tile : Nat)),
      ("ok", .bool v.reason.isNone), ("reason", jopt v.reason)]).toArray)]

end PouwCheck.PearlPlus
