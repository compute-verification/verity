/-
`RecOpen.check` on a unit named `rec-open/l8-h2` with rec-open's input and output bits there (384 input words, 128 output
words): the statement's first five rows must be `row`, `aux`, `acc`, `coef` and `dirs` at 64, 64, 64, 128 and 64 words
(`out` and `acc_out` follow), and every copy must read their words in order. Each is refused before the net is built;
the honest rows and leaves pass both checks, at one copy or two.

`RecVStar.checkAccReads` at the inner shape of `test_lean_rec_coins`' class (m 25: four levels of 218, 106, 71 and 53
openings a rep) and, for R's levels, R at m 22 (218, 106 and 53): level 3's `acc` reads rows 395 to 447 for rep 0 and 843
to 895 for rep 1, as the recorded level's reads file does, and `acc_out` the next rows; R's level 1 reads rows from
`2 · 448 + 218`. A reads file whose `acc` reads its openings level-major, or whose `acc_out` reads `acc`'s rows, or one
short, is refused at the first unit it differs at.

`RecVStar.checkAcc` refuses a reads file that names a port twice (`checkNames`): `checkAccReads` reads a name's first
port, so a second `acc` reading level-major rows passes it unread.
-/
import Flock.RecOpen
import Flock.RecVStar

namespace Flock.Test.RecOpenLeaves

open Flock.RecOpen

def unit : Net :=
  { (default : Net) with name := "rec-open/l8-h2", inGroups := #[{ col := 0, words := 384, bits := Array.replicate 384 16 }],
                         outGroups := #[{ col := 0, words := 128, bits := Array.replicate 128 16 }] }

def port (name : String) (words : Nat) : Port := { name, words, chunks := 2 * words / 128, nb := 0 }

def honest : Array Port :=
  #[port "row" 64, port "aux" 64, port "acc" 64, port "coef" 128, port "dirs" 64, port "out" 64, port "acc_out" 64]

def inOrder : Array Int := (Array.range 384).map Int.ofNat

def circuit (ports : Array Port) (leaves : Array (Array Int)) : Circuit :=
  { (default : Circuit) with name := "t", nets := #[("unit", unit)], unitNet := 0, ports, leavesIn := leaves,
                             upv := leaves.size }

def refusal (ports : Array Port) (leaves : Array (Array Int)) : Option String :=
  match Flock.RecOpen.check (circuit ports leaves) with
  | .error e => some e
  | .ok () => none

def rowsRefused : Option String :=
  some "circuit t: rec-open/l8-h2's first five rows are not row, aux, acc, coef and dirs at its widths"

def leavesRefused : Option String := some "circuit t: a copy of rec-open/l8-h2 does not read its rows' words in order"

def words (a b : Nat) : Array Int := (Array.range (b - a)).map fun i => Int.ofNat (a + i)

#guard rowsOk (circuit honest #[inOrder]) 8 2 && leavesOk (circuit honest #[inOrder]) 8 2
#guard rowsOk (circuit honest #[inOrder, inOrder]) 8 2 && leavesOk (circuit honest #[inOrder, inOrder]) 8 2

#guard refusal #[port "row" 64, port "aux" 64, port "acc" 64, port "dirs" 64, port "coef" 128, port "out" 64,
  port "acc_out" 64] #[inOrder] == rowsRefused
#guard refusal (honest.set! 4 (port "dir" 64)) #[inOrder] == rowsRefused
#guard refusal (honest.set! 2 (port "acc" 128)) #[inOrder] == rowsRefused
#guard refusal (honest.set! 4 { port "dirs" 64 with bits := 1023 }) #[inOrder] == rowsRefused
#guard refusal (honest.extract 0 4) #[inOrder] == rowsRefused

#guard refusal honest #[words 0 192 ++ words 320 384 ++ words 192 320] == leavesRefused
#guard refusal honest #[words 64 128 ++ words 0 64 ++ words 128 384] == leavesRefused
#guard refusal honest #[inOrder.set! 383 (-1)] == leavesRefused
#guard refusal honest #[inOrder, inOrder.set! 200 199] == leavesRefused
#guard refusal honest #[inOrder.pop] == leavesRefused

/-! V*'s output layout (`outsOk`): `out` then `acc_out`, one output row each past the five input rows, every copy
returning words `0..127` in order. A circuit carrying it passes `outsOk`; one missing any part is refused there, after
the rows and leaves. -/

def circuitOut (ports : Array Port) (leaves : Array (Array Int)) : Circuit :=
  { circuit ports leaves with outPorts := #[("out", 64), ("acc_out", 64)], outRowPorts := #[1, 1],
                              leavesOut := leaves.map fun _ => Array.range 128 }

def refusalOut (ports : Array Port) (leaves : Array (Array Int)) : Option String :=
  match Flock.RecOpen.check (circuitOut ports leaves) with
  | .error e => some e
  | .ok () => none

def outsRefused : Option String :=
  some ("circuit t: rec-open/l8-h2's outputs are not out then acc_out, 64 words each, one output row " ++
    "each, every copy returning words 0..127")

#guard outsOk (circuitOut honest #[inOrder]) && outsOk (circuitOut honest #[inOrder, inOrder])
#guard refusal honest #[inOrder] == outsRefused
#guard refusalOut (honest.set! 5 (port "outs" 64)) #[inOrder] == outsRefused
#guard refusalOut (honest.set! 6 (port "acc_out" 32)) #[inOrder] == outsRefused
#guard !outsOk { circuitOut honest #[inOrder] with outPorts := #[("out", 64)] }
#guard !outsOk { circuitOut honest #[inOrder] with outRowPorts := #[2] }
#guard !outsOk { circuitOut honest #[inOrder] with leavesOut := #[(Array.range 128).reverse] }
#guard !outsOk { circuitOut honest #[inOrder, inOrder] with leavesOut := #[Array.range 128, Array.range 127] }

def inner : RecVStar.Inner := { m := 25, kLog := 16, claims := 8, regions := 3 }

def innerR : RecVStar.Inner := { inner with stacked := some { m := 22, claims := 2 } }

def accPort (name : String) (reads : Array (Option Nat)) : Registered.Port :=
  { name, value := "rec-acc", root := ByteArray.empty, leaves := 897, words := some 64, wordBits := 16,
    domain := ByteArray.empty, reads }

def level3 : VStar.Stmt := .recOpen 3 16 4 106

def rowsAt (s : VStar.Stmt) (i : RecVStar.Inner := inner) : Array (Option Nat) :=
  ((RecVStar.accReads i s).toOption.join).getD #[]

def accRefusal (i : RecVStar.Inner) (s : VStar.Stmt) (acc accOut : Array (Option Nat)) : Option String :=
  match RecVStar.checkAccReads i s #[accPort "acc" acc, accPort "acc_out" accOut] with
  | .error e => some e
  | .ok () => none

def after (r : Array (Option Nat)) : Array (Option Nat) := r.map (·.map (· + 1))

#guard RecVStar.opensBelow ((Flock.fast100 25).get!) 4 == 448 && RecVStar.opensBelow ((Flock.fast100 25).get!) 3 == 395
#guard rowsAt level3 == ((Array.range 53).map fun j => some (395 + j)) ++ (Array.range 53).map fun j => some (843 + j)
#guard (rowsAt (.recOpen 0 64 8 436)).toList.take 2 == [some 0, some 1] && (rowsAt (.recOpen 0 64 8 436))[218]! == some 448
#guard rowsAt (.recOpenStacked 1 16 7 106) innerR == (Array.range 106).map fun j => some (1114 + j)
#guard (RecVStar.accReads inner (.recOpenStacked 1 16 7 106)).toOption.isNone

#guard accRefusal inner level3 (rowsAt level3) (after (rowsAt level3)) == none
#guard accRefusal innerR (.recOpenStacked 1 16 7 106) (rowsAt (.recOpenStacked 1 16 7 106) innerR)
  (after (rowsAt (.recOpenStacked 1 16 7 106) innerR)) == none
#guard accRefusal inner level3 ((Array.range 106).map fun u => some (790 + u)) (after (rowsAt level3)) ==
  some "rec-open/l16-h4: its port acc reads rec-acc at 790 in unit 0, not at 395 (the chain runs rep-major, then by level and query)"
#guard accRefusal inner level3 (rowsAt level3) (rowsAt level3) ==
  some "rec-open/l16-h4: its port acc_out reads rec-acc at 395 in unit 0, not at 396 (the chain runs rep-major, then by level and query)"
#guard accRefusal inner level3 (rowsAt level3).pop (after (rowsAt level3)) ==
  some "rec-open/l16-h4: its port acc reads rec-acc at nothing in unit 105, not at 895 (the chain runs rep-major, then by level and query)"
#guard (RecVStar.checkAccReads inner level3 #[accPort "acc" (rowsAt level3)]) matches .error _

def honestAcc : Array Registered.Port := #[accPort "acc" (rowsAt level3), accPort "acc_out" (after (rowsAt level3))]

def twiceAcc : Array Registered.Port := honestAcc.push (accPort "acc" ((Array.range 106).map fun u => some (790 + u)))

def namesRefusal (ports : Array Registered.Port) : Option String :=
  match RecVStar.checkAcc level3 ports (some #[accPort "acc0" #[]]) with
  | .error e => some e
  | .ok () => none

#guard (RecVStar.checkAccReads inner level3 twiceAcc) matches .ok _
#guard namesRefusal honestAcc == none
#guard namesRefusal twiceAcc == some "rec-open/l16-h4: the registered reads name two ports acc, and each check reads a name's first"
#guard namesRefusal (#[accPort "acc_out" #[]] ++ honestAcc) ==
  some "rec-open/l16-h4: the registered reads name two ports acc_out, and each check reads a name's first"

end Flock.Test.RecOpenLeaves
