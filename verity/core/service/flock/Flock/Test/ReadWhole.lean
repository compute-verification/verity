/-
`RecResiduals.partlyRead` on small nets: the row ports' bits sit at `rowCol`'s columns (port `n` elements wide spans
`16 · portWordsOf n` columns), an element read in no bit or in all 128 passes, and the first element read in some but not
all is named with how many. `readIns` marks the input columns the computed rows read and nothing past the inputs.
-/
import Flock.RecResiduals

namespace Flock.Test.ReadWhole

open Flock

/-- A net of `nIn` input columns whose computed rows read, in A, the columns of `reads` (one row each, B empty). -/
def net (inWords : Nat) (reads : List (List Nat)) : Net :=
  let nIn := inWords * WORD
  let rows := Array.replicate nIn #[] ++ (reads.map List.toArray).toArray
  let w := nIn + reads.length + 1
  match Sparse.ofRows rows w, Sparse.ofRows (Array.replicate rows.size #[]) w with
  | .ok a, .ok b => { (default : Net) with constPos := nIn + reads.length, inWords, a, b }
  | _, _ => default

def partly (inWords : Nat) (ns : List Nat) (reads : List (List Nat)) : Option (Nat × Nat) :=
  RecResiduals.partlyRead (RecResiduals.readIns (net inWords reads)) ns

#guard RecResiduals.rowCol [1, 1] 0 5 == 5
#guard RecResiduals.rowCol [1, 1] 0 (128 + 5) == 16 * 64 + 5
#guard partly 16 [1, 1] [] == none
#guard partly 16 [1, 1] [List.range 128] == none
#guard partly 16 [1, 1] [List.range 64, List.range' 64 64] == none
#guard partly 16 [1, 1] [List.range 64] == some (0, 64)
#guard partly 16 [1, 1] [List.range 128, [1024 + 7]] == some (1, 1)
#guard partly 16 [1, 1] [List.range 128, List.range' 1024 128] == none
#guard partly 16 [1, 1] [List.range' 1024 127] == some (1, 127)
#guard (RecResiduals.readIns (net 16 [[3, 2048 + 1]])).size == 2048
#guard (RecResiduals.readIns (net 16 [[2048]])).toList.all (!·)

end Flock.Test.ReadWhole
