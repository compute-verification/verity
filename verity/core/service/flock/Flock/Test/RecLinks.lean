/-
`RecLinks.check`: the prover's table of a chain of links whose rows agree passes at counts on and off the grid, and a row
cut short, a link after a filler, a table off its grid, a link outside the shape, a link between rows of different widths,
an input row linked twice and a link whose rows differ are refused, each with its code; `--rec-links` is refused.
-/
import Flock.RecLinks

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

end Flock.Test.RecLinks
