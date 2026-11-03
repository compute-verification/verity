/-
`RecBudget` under F1's inner shape: a registration of `MAX_PORTS` ports of trees `MAX_DEPTH` deep, and a table of
`MAX_INSTANCES` instances, pass, as does a level's registration (`rec-acc` of 2048 rows, its tops, its `2 q` openings'
coefficients); one port more, one leaf past `2^MAX_DEPTH`, or one instance more is refused, each by its own message.
V\*'s list of `MAX_STATEMENTS = 16` statements passes and one more is refused (`count`), naming the soundness bound's
cap; a shape `RecVStar.statements` refuses (claims not two and two per region) is left to the checks that run it.
Without an inner shape nothing is refused.
-/
import Flock.RecBudget

namespace Flock.Test.RegBudget

open Flock RecBudget

def inner : RecVStar.Inner := { m := 27, kLog := 24, claims := 8, regions := 3 }

def port (name : String) (leaves : Nat) : Registered.Port := { (default : Registered.Port) with name, leaves }

def tbl (n : Nat) : Stmt := { (default : Stmt) with n := n, pub := { (default : Public) with n := n } }

def whyR (i : Option RecVStar.Inner) (r : Option (Array Registered.Port)) : Option String :=
  match regs i r with
  | .ok () => none
  | .error e => some e

def whyT (i : Option RecVStar.Inner) (sts : Array Stmt) : Option String :=
  match table i sts with
  | .ok () => none
  | .error e => some e

#guard whyR inner none == none
#guard whyR inner (some #[port "acc" 2048, port "out" 128, port "coef" 510]) == none
#guard whyR inner (some (Array.replicate MAX_PORTS (port "p" (2 ^ MAX_DEPTH)))) == none
#guard whyR inner (some (Array.replicate (MAX_PORTS + 1) (port "p" 1))) ==
  some "refused: a V* session's registered reads name 4097 ports, more than 4096: the recursion's call bound covers no more"
#guard whyR inner (some #[port "acc" 2048, port "big" (2 ^ MAX_DEPTH + 1)]) ==
  some ("refused: a V* session's registered port big has 18446744073709551617 leaves, a tree of 65 levels, more than 64: " ++
    "the recursion's call bound covers no more")
#guard whyR none (some (Array.replicate (MAX_PORTS + 1) (port "p" (2 ^ 80)))) == none
#guard whyT inner #[tbl 510] == none
#guard whyT inner #[tbl MAX_INSTANCES] == none
#guard whyT inner #[tbl (MAX_INSTANCES + 1)] ==
  some ": refused: a V* session's table of 65537 instances, more than 65536: the recursion's call bound covers no more"
#guard whyT none #[tbl (MAX_INSTANCES + 1)] == none

def whyC (n : Nat) : Option String :=
  match count n with
  | .ok () => none
  | .error e => some e

#guard MAX_STATEMENTS == 16
#guard whyC 10 == none
#guard whyC 16 == none
#guard whyC 17 ==
  some ("refused: V*'s list at the verifier's inner shape holds 17 statements, more than 16: the recursion's soundness " ++
    "bound covers at most 16 V* parts")
#guard whyC 4096 ==
  some ("refused: V*'s list at the verifier's inner shape holds 4096 statements, more than 16: the recursion's soundness " ++
    "bound covers at most 16 V* parts")
#guard (statements none).toOption == some ()
#guard (statements (some { inner with claims := 9 })).toOption == some ()

end Flock.Test.RegBudget
