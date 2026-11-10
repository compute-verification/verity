/-
`ProvedScope.portReadsB`, which compiled code runs as `portReadsBFast`: computed rows reading only port bits, or columns
past the input rows, pass; a row reading an input column outside every port fails, in A or in B.
-/
module

public import Flock.ProvedScope
meta import Flock.ProvedScope

@[expose] public section

namespace Flock.Test.PortReads

open Flock

/-- One input word whose port is its first `bits` columns, the constant at 131, and the computed rows 128 to 130:
`ra`'s and `rb`'s, after 128 empty input rows. -/
def net (bits : Nat) (ra rb : Array (Array Nat)) : Net :=
  let sp := fun (rows : Array (Array Nat)) =>
    match Sparse.ofRows (Array.replicate 128 #[] ++ rows ++ #[#[131]]) 256 with
    | .ok m => m
    | .error _ => default
  { name := "t", useful := 132, constPos := 131, inGroups := #[⟨0, 1, #[bits]⟩], outGroups := #[], inWords := 1,
    aT := .pure (sp ra), bT := .pure (sp rb), unitLog := 8 }

def ra : Array (Array Nat) := #[#[0, 1], #[128, 3], #[129, 130]]
def rb : Array (Array Nat) := #[#[2], #[131], #[]]

#guard ProvedScope.portReadsB (net 128 ra rb)
#guard ProvedScope.portReadsB (net 4 ra rb)
#guard !ProvedScope.portReadsB (net 3 ra rb)
#guard !ProvedScope.portReadsB (net 4 ra #[#[2], #[131], #[7]])
#guard !ProvedScope.portReadsB (net 4 #[#[0, 1], #[128, 3, 100], #[]] rb)

end Flock.Test.PortReads
