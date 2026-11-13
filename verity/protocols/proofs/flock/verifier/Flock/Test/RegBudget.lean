/-
`RecBudget` under F1's inner shape: a registration of `MAX_PORTS` ports of trees `MAX_DEPTH` deep, and a table of
`MAX_INSTANCES` instances, pass, as does a level's registration (`rec-acc` of 2048 rows, its tops, its `2 q` openings'
coefficients); one port more, one leaf past `2^MAX_DEPTH`, or one instance more is refused, each by its own message.
V\*'s list of `MAX_STATEMENTS = 16` statements passes and one more is refused (`count`), naming the soundness bound's
cap; a shape `RecVStar.statements` refuses (claims not two and two per region) is left to the checks that run it.
Without an inner shape nothing is refused. A list recording `MAX_RECORDED` hm96 calls passes and one more is refused
(`recorded`); a level's statement records two calls at each of its openings' `out`, `coef` and `dirs` and four at each
`rec-acc` row its units read, R's level's at `out` and `rec-acc` alone (none of `rec-acc` without R's shape), a RecCap
four at each node, RecSparse two at each of its ports' two rows, and the link table's part none (`recordedAt`). The
inner statement's public file of `MAX_INNER_ROWS` instances, or of shared-row tables of `MAX_INNER_ROWS` rows, passes
`innerRows`, and one instance or row more is refused.
-/
module

public import Flock.RecBudget
meta import Flock.RecBudget

@[expose] public section

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

#guard (MAX_INSTANCES, MAX_PORTS, MAX_DEPTH) == (4096, 512, 64)
#guard MAX_STATEMENTS * (131072 + MAX_INSTANCES * MAX_PORTS * (2 + MAX_DEPTH)) < 2 ^ 32
#guard whyR inner none == none
#guard whyR inner (some #[port "acc" 2048, port "out" 128, port "coef" 510]) == none
#guard whyR inner (some (Array.replicate MAX_PORTS (port "p" (2 ^ MAX_DEPTH)))) == none
#guard whyR inner (some (Array.replicate (MAX_PORTS + 1) (port "p" 1))) ==
  some "refused: a V* session's registered reads name 513 ports, more than 512: the recursion's call bound covers no more"
#guard whyR inner (some #[port "acc" 2048, port "big" (2 ^ MAX_DEPTH + 1)]) ==
  some ("refused: a V* session's registered port big has 18446744073709551617 leaves, a tree of 65 levels, more than 64: " ++
    "the recursion's call bound covers no more")
#guard whyR none (some (Array.replicate (MAX_PORTS + 1) (port "p" (2 ^ 80)))) == none
#guard whyT inner #[tbl 510] == none
#guard whyT inner #[tbl MAX_INSTANCES] == none
#guard whyT inner #[tbl (MAX_INSTANCES + 1)] ==
  some ": refused: a V* session's table of 4097 instances, more than 4096: the recursion's call bound covers no more"
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

def whyW (n : Nat) : Option String :=
  match recorded n with
  | .ok () => none
  | .error e => some e

#guard whyW 0 == none
#guard whyW MAX_RECORDED == none
#guard whyW (MAX_RECORDED + 1) ==
  some ("refused: V*'s list at the verifier's inner shape records 4294967297 hm96 calls, more than 4294967296: " ++
    "the recursion's call bound covers no more")
#guard recordedAt inner (.recOpen 0 64 8 436) == 4360
#guard recordedAt { inner with stacked := some { m := 23, claims := 2 } } (.recOpenStacked 0 64 8 218) == 1308
#guard recordedAt inner (.recOpenStacked 0 64 8 218) == 436
#guard recordedAt inner (.cap 0 128) == 512
#guard recordedAt inner (.sparse { nu := 18, ell := 13, sLog := 6, tLog := 0 } 21) == 8
#guard recordedAt inner (.link 6 6 18) == 0
#guard recordedCalls inner #[.recOpen 0 64 8 436, .cap 1 128, .cap 0 128] == 5384

def innerSt (n : Nat) (counts : Option (Array Nat)) : Stmt :=
  let tables : Option Tables := counts.map fun c => { counts := c, starts := #[], refs := #[] }
  { (default : Stmt) with n := n, pub := { (default : Public) with n := n, tables := tables } }

def whyI (st : Stmt) : Option String :=
  match innerRows st with
  | .ok () => none
  | .error e => some e

#guard MAX_INNER_ROWS == 2 ^ 17
#guard 12 + 1024 * ((17 + 3) * 2 ^ 17) < 2 ^ 32
#guard whyI (innerSt 16 none) == none
#guard whyI (innerSt MAX_INNER_ROWS none) == none
#guard whyI (innerSt (MAX_INNER_ROWS + 1) none) ==
  some ("refused: the inner statement's public file holds 131073 instances, more than 131072: " ++
    "the recursion's call bound covers no more")
#guard whyI (innerSt 16 (some #[16, MAX_INNER_ROWS])) == none
#guard whyI (innerSt 16 (some #[16, MAX_INNER_ROWS + 1])) ==
  some ("refused: the inner statement's public file has a shared-row table of more than 131072 rows: " ++
    "the recursion's call bound covers no more")

end Flock.Test.RegBudget
