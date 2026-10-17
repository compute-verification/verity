/-
`RecSparse.check` on a unit named `rec-sparse/n2-e6-s0-t1` with rec-sparse's input and output bits there: the statement's
first rows must be `ports` at their widths, `h` (512 words) first and `v` last (`out` follows), and every copy must read
their words in order. Each is refused before the net is built; the honest rows and leaves, as `rec_sparse.statement`
composes them, pass both checks, at one copy or two.
-/
import Flock.RecSparse

namespace Flock.Test.RecSparseRows

open Flock.RecSparse

def sh : RecSparse.Shape := ⟨2, 6, 0, 1⟩

def unit : Net :=
  { (default : Net) with name := unitName sh,
                         inGroups := #[{ col := 0, words := inWords sh, bits := Array.replicate (inWords sh) 16 }],
                         outGroups := #[{ col := 0, words := outWords sh, bits := Array.replicate (outWords sh) 16 }] }

def port (name : String) (words : Nat) : Port := { name, words, chunks := 2 * words / 128, nb := 0 }

def honest : Array Port := ((ports sh).map fun p => port p.1 p.2).toArray.push (port "out" (outWords sh))

def inOrder : Array Int := (Array.range (inWords sh)).map Int.ofNat

def circuit (ports : Array Port) (leaves : Array (Array Int)) : Circuit :=
  { (default : Circuit) with name := "t", nets := #[("unit", unit)], unitNet := 0, ports, leavesIn := leaves,
                             upv := leaves.size }

def refusal (ports : Array Port) (leaves : Array (Array Int)) : Option String :=
  match Flock.RecSparse.check (circuit ports leaves) with
  | .error e => some e
  | .ok () => none

def rowsRefused : Option String :=
  some "circuit t: rec-sparse/n2-e6-s0-t1's first rows are not rec-sparse's ports at their widths"

def leavesRefused : Option String := some "circuit t: a copy of rec-sparse/n2-e6-s0-t1 does not read its rows' words in order"

#guard (ports sh).head? == some ("h", 512) && (ports sh).getLast? == some ("v", portWords (nv sh))
#guard rowsOk (circuit honest #[inOrder]) sh && leavesOk (circuit honest #[inOrder]) sh
#guard rowsOk (circuit honest #[inOrder, inOrder]) sh && leavesOk (circuit honest #[inOrder, inOrder]) sh

#guard refusal ((honest.swapIfInBounds 0 1)) #[inOrder] == rowsRefused
#guard refusal (honest.set! 0 (port "g" 512)) #[inOrder] == rowsRefused
#guard refusal (honest.set! 0 (port "h" 448)) #[inOrder] == rowsRefused
#guard refusal (honest.set! 0 { port "h" 512 with bits := 8191 }) #[inOrder] == rowsRefused
#guard refusal (honest.extract 0 ((ports sh).length - 1)) #[inOrder] == rowsRefused

#guard refusal honest #[(Array.range 512).map (Int.ofNat ∘ (· + 64)) ++ (Array.range 64).map Int.ofNat ++
  inOrder.extract 576 (inWords sh)] == leavesRefused
#guard refusal honest #[inOrder.set! 0 (-1)] == leavesRefused
#guard refusal honest #[inOrder, inOrder.set! 511 510] == leavesRefused
#guard refusal honest #[inOrder.pop] == leavesRefused
#guard (Flock.RecSparse.check { circuit honest #[inOrder] with upv := 2 }) matches .error _

end Flock.Test.RecSparseRows
