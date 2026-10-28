import Flock.Hm96
import Flock.RecLinks

/-!
# The link check's rows trees (`verity_flock.rec_link.Tree`)

A recursive session's link table (`Flock.RecLinks`) is checked one row an instance by V*'s `RecLink_v1{UB, PB}`, which
climbs the reader's and the writer's rows to two roots the verifier builds itself, from the commit strings `b ‖ c` the
inner statement publishes for each unit's rows: input row `p` of unit `u` is leaf `u · 2^PB + p` of the input rows' tree,
output row `o` of unit `u` leaf `u · 2^PB + o` of the output rows' tree. A leaf is the row's hm96 tree leaf
(`Hm96.Default512.treeLeaf`), a leaf with no row `EMPTY`, and a node `SHA-512(left ‖ right)`.

`rowsRoot` is one tree's root, `rowsPath` an opening of it. Each is linear in the rows it is given (one hash a row a level,
at most `D ≤ 32` levels): the public commitments the verifier already reads, not the circuit they commit.

`linksArg` reads `--rec-links ROOT:N`, the table's registered root and row count, as `rec-compose` takes them. Every
statement carrying them is still refused (`RecLinks.refuseLinks`): no guarantee yet covers a table.
-/

namespace Flock.RecLinkTree

def UB_MAX : Nat := 30
def PB_MAX : Nat := 16
def D_MAX : Nat := 32

/-- A leaf no row fills (`rec_link.EMPTY`): the SHA-512 of the empty string, which is no row leaf's input
(`leaf_prefix ‖ b ‖ c`), so a row leaf that opens it is a SHA-512 collision. -/
def EMPTY : ByteArray := Sha512.hash ByteArray.empty

def node (l r : ByteArray) : ByteArray := Sha512.hash (l ++ r)

/-- The root of an empty subtree of height `h`. -/
def empty : Nat → ByteArray
  | 0 => EMPTY
  | h + 1 => let e := empty h; node e e

/-- `rec_link._check`: `UB` at most 30, `PB` at most 16, `D = UB + PB` at most 32. -/
def checkShape (UB PB : Nat) : Except String Unit :=
  unless UB ≤ UB_MAX && PB ≤ PB_MAX && UB + PB ≤ D_MAX do
    throw s!"RecLink's UB is 0 to {UB_MAX} and PB 0 to {PB_MAX}, D = UB + PB at most {D_MAX}: not UB {UB}, PB {PB}"

/-- `rec_link.index`: unit `u`'s row `r` (an input port or an output row). -/
def index (u r PB : Nat) : Nat := u * 2 ^ PB + r

/-- `rec_link.leaf`: a row's leaf from its commit string. -/
def leaf (bc : ByteArray) : ByteArray := Hm96.Default512.treeLeaf bc

/-- The empty subtrees' roots, heights `0 .. D`, each computed once. -/
def empties (D : Nat) : Array ByteArray := ((List.range (D + 1)).map empty).toArray

/-- A leaf's index relative to the upper half of a subtree of height `h + 1`. -/
def shift (h : Nat) (x : Nat × ByteArray) : Nat × ByteArray := (x.1 - 2 ^ h, x.2)

/-- **A subtree's root**, height `h`, over `leaves` (indices relative to it, each below `2^h`, in increasing order), `e k`
the root of an empty subtree of height `k`: `e h` with no leaves, the leaf at height 0, else the node over its two halves,
the leaves below `2^(h-1)` and the rest. Each leaf is in one half a level, so the work is the leaves times `h`. -/
def sub (e : Nat → ByteArray) : Nat → List (Nat × ByteArray) → ByteArray
  | h, [] => e h
  | 0, (_, v) :: _ => v
  | h + 1, leaves@(_ :: _) =>
    let p := leaves.partition (·.1 < 2 ^ h)
    node (sub e h p.1) (sub e h (p.2.map (shift h)))

/-- Leaf `idx`'s siblings in that subtree, bottom up. -/
def subPath (e : Nat → ByteArray) : Nat → List (Nat × ByteArray) → Nat → List ByteArray
  | 0, _, _ => []
  | h + 1, leaves, idx =>
    let p := leaves.partition (·.1 < 2 ^ h)
    if idx < 2 ^ h then subPath e h p.1 idx ++ [sub e h (p.2.map (shift h))]
    else subPath e h (p.2.map (shift h)) (idx - 2 ^ h) ++ [sub e h p.1]

def increasing : List Nat → Bool
  | a :: tl@(b :: _) => a < b && increasing tl
  | _ => true

/-- The leaves in increasing index order, each below `2^D`: a tree's leaves as `Tree` takes them, a map. -/
def checkLeaves (D : Nat) (leaves : List (Nat × ByteArray)) : Except String Unit := do
  unless increasing (leaves.map (·.1)) do throw "a rows tree's leaves are in increasing index order, each once"
  if leaves.any (fun (i, _) => 2 ^ D ≤ i) then throw s!"a leaf index outside a tree of depth {D}"

/-- `rec_link.Tree(leaves, D).root`. -/
def root (D : Nat) (leaves : List (Nat × ByteArray)) : Except String ByteArray := do
  checkLeaves D leaves
  let es := empties D
  pure (sub (fun h => es.getD h EMPTY) D leaves)

/-- `rec_link.Tree(leaves, D).path(pos)`: leaf `pos`'s siblings, bottom up, `pos` below `2^D`. -/
def path (D : Nat) (leaves : List (Nat × ByteArray)) (pos : Nat) : Except String (List ByteArray) := do
  checkLeaves D leaves
  let es := empties D
  pure (subPath (fun h => es.getD h EMPTY) D leaves pos)

/-- A rows tree's leaves from each row's `((unit, row), b ‖ c)`, in increasing `(unit, row)` order: the leaf `index u r PB`
of the row's `leaf`. A unit past `UB` bits or a row past `PB` bits is refused. -/
def rowLeaves (UB PB : Nat) (rows : List ((Nat × Nat) × ByteArray)) : Except String (List (Nat × ByteArray)) := do
  checkShape UB PB
  if let some ((u, r), _) := rows.find? fun ((u, r), _) => !(u < 2 ^ UB && r < 2 ^ PB) then
    throw s!"row {r} of unit {u} does not fit UB {UB}, PB {PB}"
  pure (rows.map fun ((u, r), bc) => (index u r PB, leaf bc))

/-- **A rows tree's root**: the input rows' tree, or the output rows', of depth `UB + PB` over the inner statement's rows
`((unit, row), b ‖ c)` in increasing order. A root of 64 zero bytes is refused: it is the output of an instance whose row
fails, which the statement must never read as a root. -/
def rowsRoot (UB PB : Nat) (rows : List ((Nat × Nat) × ByteArray)) : Except String ByteArray := do
  let r ← root (UB + PB) (← rowLeaves UB PB rows)
  if r.data.all (· == 0) then throw "a rows tree's root is 64 zero bytes, a failed row's output"
  pure r

/-- An opening of a rows tree: row `r` of unit `u`'s siblings, bottom up. -/
def rowsPath (UB PB : Nat) (rows : List ((Nat × Nat) × ByteArray)) (u r : Nat) : Except String (List ByteArray) := do
  let leaves ← rowLeaves UB PB rows
  unless u < 2 ^ UB && r < 2 ^ PB do throw s!"row {r} of unit {u} does not fit UB {UB}, PB {PB}"
  path (UB + PB) leaves (index u r PB)

/-- `--rec-links ROOT:N`: the link table's registered root (64 bytes, lowercase hex) and its row count, `grid(L)` for some
`L ≥ 1` (a count `grid` leaves fixed: at most `GRID_BITS` significant bits); none without the flag. -/
def linksArg (args : List String) : Except String (Option (ByteArray × Nat)) := do
  match args.dropWhile (· != "--rec-links") with
  | [] => pure none
  | [_] => throw "--rec-links takes ROOT:N"
  | _ :: v :: rest =>
    if rest.contains "--rec-links" then throw "--rec-links given twice"
    let [r, n] := v.splitOn ":" | throw s!"--rec-links takes ROOT:N, not {v}"
    let t ← unhex r
    unless t.size == 64 do throw s!"--rec-links: the table's root is 64 bytes, not {t.size}"
    let some n := n.toNat? | throw s!"--rec-links: the row count is a natural number, not {n}"
    unless 1 ≤ n && RecLinks.grid n == n do throw s!"--rec-links: {n} rows is no table's grid(L)"
    pure (some (t, n))

end Flock.RecLinkTree
