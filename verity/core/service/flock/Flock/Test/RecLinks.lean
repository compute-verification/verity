/-
`RecLinks.check`: the prover's table of a chain of links whose rows agree passes at counts on and off the grid, and a row
cut short, a link after a filler, a table off its grid, a link outside the shape, a link between rows of different widths,
an input row linked twice and a link whose rows differ are refused, each with its code; `--rec-links` is refused. The rows
trees (`RecLinkTree`) open to their roots, and `--rec-links ROOT:N` is read only at a table's root and grid count.
-/
import Flock.RecLinks
import Flock.RecLinkTree

namespace Flock.Test.RecLinks

open Flock.RecLinks

def sh : Shape := { units := 40, ins := #[4, 3], outs := #[4, 2] }

/-- Unit `u`'s output row `p`: four words from `u` and `p`. -/
def outRow (u p : Nat) : Array Nat := #[u, p, 7 * u + p, 65535 - u]

/-- A chain: unit `u + 1`'s input row 0 is unit `u`'s output row 0; every other input row is its own. -/
def inRow (u p : Nat) : Array Nat := if p == 0 && u > 0 then outRow (u - 1) 0 else #[u, p, 1, 1]

def chain (n : Nat) : List LinkEntry := (List.range n).map fun u => ⟨u + 1, 0, u, 0⟩

def code (rows : Array ByteArray) : String :=
  match check sh inRow outRow rows with
  | .ok n => s!"ok {n}"
  | .error e => (e.splitOn ":").head!

#guard (table (chain 17)).size == 18
#guard code (table (chain 1)) == "ok 1"
#guard code (table (chain 16)) == "ok 16"
#guard code (table (chain 17)) == "ok 17"
#guard code (table (chain 39)) == "ok 39"
#guard code ((table (chain 4)).set! 2 ((table (chain 4))[2]!.extract 0 127)) == "row-size"
#guard code ((table (chain 17)).swapIfInBounds 3 17) == "order"
#guard code (table []) == "empty"
#guard code ((table (chain 17)).pop) == "count"
#guard code (table (chain 3 ++ [⟨40, 0, 1, 0⟩])) == "range"
#guard code (table (chain 3 ++ [⟨5, 1, 4, 0⟩])) == "width"
#guard code (table (chain 3 ++ [⟨2, 0, 1, 0⟩])) == "twice"
#guard code (table (chain 3 ++ [⟨5, 0, 7, 0⟩])) == "unequal"

/-! `--rec-links` is refused, with a value or bare, before any file is read. -/

def refused (args : List String) : Option String :=
  match refuseLinks args with
  | .error e => some e
  | .ok () => none

#guard refused ["--rec-links", "links.json", "--rec-inner", "inner.json"] == some NO_LINKS
#guard refused ["--rec-inner", "inner.json", "--rec-links"] == some NO_LINKS
#guard refused ["--rec-inner", "inner.json", "--rec-part", "0"] == none
#guard refused [] == none

/-! The rows trees (`RecLinkTree`): `rec_link`'s padded tree, an opening that climbs to the root, and the trees' and
`--rec-links`' refusals. -/

section Trees

open Flock.RecLinkTree

def x : ByteArray := ByteArray.mk (Array.replicate 64 1)
def bc (k : Nat) : ByteArray := ByteArray.mk (Array.replicate 128 k.toUInt8)
def dat (r : Except String ByteArray) : Option (Array UInt8) := r.toOption.map (·.data)

/-- `rec_link._climb_ref`: `start` climbed by `sibs` at `pos`. -/
def climb (start : ByteArray) (sibs : List ByteArray) (pos : Nat) : ByteArray :=
  (List.range sibs.length).foldl (fun cur h =>
    let s := sibs.getD h EMPTY
    if (pos >>> h) % 2 == 1 then node s cur else node cur s) start

#guard dat (root 2 [(1, x)]) == some (node (node EMPTY x) (node EMPTY EMPTY)).data
#guard ((path 2 [(1, x)] 1).toOption.map (·.map (·.data))) == some [EMPTY.data, (node EMPTY EMPTY).data]
#guard dat (root 0 []) == some EMPTY.data
#guard dat (root 0 [(0, x)]) == some x.data
#guard dat (root 3 []) == some (empty 3).data
#guard (root 1 [(1, x), (0, x)]).toOption.isNone
#guard (root 1 [(0, x), (0, x)]).toOption.isNone
#guard (root 1 [(2, x)]).toOption.isNone

def rows : List ((Nat × Nat) × ByteArray) := [((0, 1), bc 1), ((1, 0), bc 2), ((2, 1), bc 3)]

/-- Whether `start`, climbed by unit `u`'s row `r`'s opening in the tree over `rows`, reaches its root. -/
def opens (u r : Nat) (start : ByteArray) : Except String Bool := do
  let root ← rowsRoot 2 1 rows
  let p ← rowsPath 2 1 rows u r
  pure ((climb start p (index u r 1)).data == root.data)

#guard (opens 2 1 (leaf (bc 3))).toOption == some true
#guard (opens 0 1 (leaf (bc 1))).toOption == some true
#guard (opens 3 0 EMPTY).toOption == some true
#guard (opens 2 1 (leaf (bc 2))).toOption == some false
#guard (rowsRoot 2 1 [((4, 0), bc 1)]).toOption.isNone
#guard (rowsRoot 2 1 [((0, 2), bc 1)]).toOption.isNone
#guard (rowsPath 2 1 rows 4 0).toOption.isNone
#guard (rowsRoot 31 0 []).toOption.isNone
#guard (rowsRoot 16 17 []).toOption.isNone

def hex128 : String := "".pushn 'a' 128
def count (args : List String) : Option (Option Nat) := (linksArg args).toOption.map (·.map (·.2))

#guard count ["--rec-inner", "inner.json"] == some none
#guard count ["--rec-links", hex128 ++ ":18"] == some (some 18)
#guard count ["--rec-links", hex128 ++ ":1"] == some (some 1)
#guard count ["--rec-links", hex128 ++ ":17"] == none
#guard count ["--rec-links", hex128 ++ ":0"] == none
#guard count ["--rec-links", "".pushn 'a' 126 ++ ":18"] == none
#guard count ["--rec-links", hex128] == none
#guard count ["--rec-links"] == none
#guard count ["--rec-links", hex128 ++ ":18", "--rec-links", hex128 ++ ":18"] == none

end Trees

end Flock.Test.RecLinks
