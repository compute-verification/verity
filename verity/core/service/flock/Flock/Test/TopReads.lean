/-
`RecCoins.checkTopReads`, R's level `l`'s `out` port: at level 1 with `H = 2`, openings at positions 5, 13 and 6 read R's
tops `rec-top-stacked-L1` at nodes 1, 3 and 1, as `rec_outer.TopValue.position` reads them at R's one rep. A reads file
without `out`, with `out` under another level's tops or the reps' `rec-top-L1`, reading another node, short of an opening,
or past the last one is refused; only the first `out` port is read.
-/
import Flock.RecCoins

namespace Flock.Test.TopReads

open Flock Flock.RecCoins

def port (name value : String) (reads : List Nat) : Registered.Port :=
  { name, value, root := .empty, leaves := 4, words := some 8, wordBits := 16, domain := .empty,
    reads := (reads.map some).toArray }

def pos : Array Nat := #[5, 13, 6]

def refused (own : Array Registered.Port) (q : Nat := 3) : Bool :=
  (checkTopReads "rec-open-stacked/l8-h2" own 1 q 2 pos).toOption.isNone

def honest : Registered.Port := port "out" "rec-top-stacked-L1" [1, 3, 1]

#guard !refused #[port "acc" "rec-acc" [0, 1, 2], honest]
#guard refused #[port "acc" "rec-acc" [0, 1, 2]]
#guard refused #[port "out" "rec-top-stacked-L0" [1, 3, 1]]
#guard refused #[port "out" "rec-top-L1" [1, 3, 1]]
#guard refused #[port "out" "rec-top-stacked-L1" [1, 3, 0]]
#guard refused #[port "out" "rec-top-stacked-L1" [5, 13, 6]]
#guard refused #[port "out" "rec-top-stacked-L1" [1, 3]]
#guard refused #[honest] (q := 2)
#guard refused #[port "out" "rec-top-stacked-L1" [1, 3, 0], honest]
#guard !refused #[honest, port "out" "rec-top-stacked-L1" [1, 3, 0]]
def why (own : Array Registered.Port) : Option String :=
  match checkTopReads "u" own 1 3 2 pos with
  | .error e => some e
  | .ok () => none

#guard why #[] == some "rec-coins: u: the verifier's registered reads hold no port out: R's level 1 tops would be the prover's"
#guard why #[port "out" "rec-top-stacked-L1" [1, 3, 0]] ==
  some "rec-coins: u: rec-top-stacked-L1: the reads file does not read opening j's top at node pos[j] >>> 2"

end Flock.Test.TopReads
