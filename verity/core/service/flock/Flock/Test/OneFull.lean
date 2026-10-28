/-
`RecSetup.oneFull` under F1's inner shape: one table holding its whole population passes, undrawn or drawn in full (a
population of four, four instances); a table drawn short of its population (three of four, as a sampled RecCap table
would be) is refused, and so are two tables and none. Without an inner shape it passes whatever the tables are.
-/
import Flock.RecSetup

namespace Flock.Test.OneFull

open Flock

def inner : RecVStar.Inner := { m := 27, kLog := 24, claims := 8, regions := 3 }

/-- A table of `n` instances, drawn from a registered population of `pop` when `pop` is some. -/
def tbl (n : Nat) (pop : Option Nat := none) : Stmt :=
  { (default : Stmt) with n := n, pub := { (default : Public) with n := n, population := pop.map (·, #[], #[]) } }

def why (i : Option RecVStar.Inner) (sts : Array Stmt) : Option String :=
  match RecSetup.oneFull i sts with
  | .ok () => none
  | .error e => some e

def tail : String := "the only form the recursion's proof covers"

#guard why inner #[tbl 4] == none
#guard why inner #[tbl 4 (some 4)] == none
#guard why inner #[tbl 3 (some 4)] ==
  some (": refused: a table of 3 of its population's 4 instances: a V* session's table holds every instance, " ++ tail)
#guard why inner #[tbl 0 (some 1)] ==
  some (": refused: a table of 0 of its population's 1 instances: a V* session's table holds every instance, " ++ tail)
#guard why inner #[tbl 2, tbl 2] == some ("refused: a V* session of 2 tables: a V* session is one table, " ++ tail)
#guard why inner #[tbl 4 (some 4), tbl 4 (some 4)] ==
  some ("refused: a V* session of 2 tables: a V* session is one table, " ++ tail)
#guard why inner #[] == some ("refused: a V* session of 0 tables: a V* session is one table, " ++ tail)
#guard why none #[tbl 3 (some 4)] == none
#guard why none #[tbl 2, tbl 2] == none
#guard why none #[] == none

end Flock.Test.OneFull
