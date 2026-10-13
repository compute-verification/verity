/-
`RecOpen.check` on a unit named `rec-open/l8-h2` with rec-open's input and output bits there (384 input words, 128 output
words): the statement's first five rows must be `row`, `aux`, `acc`, `coef` and `dirs` at 64, 64, 64, 128 and 64 words
(`out` and `acc_out` follow), and every copy must read their words in order. Each is refused before the net is built;
the honest rows and leaves pass both checks, at one copy or two.
-/
import Flock.RecOpen

namespace Flock.Test.RecOpen

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

end Flock.Test.RecOpen
