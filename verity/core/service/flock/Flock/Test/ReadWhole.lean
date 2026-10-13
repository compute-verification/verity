/-
`RecResiduals.partlyRead` on small nets: the row ports' bits sit at `rowCol`'s columns (port `n` elements wide spans
`16 · portWordsOf n` columns), an element read in no bit or in all 128 passes, and the first element read in some but not
all is named with how many, wherever it sits among held elements read in no bit. `readIns` marks the input columns the
computed rows read and nothing past the inputs. `RecOpening.rsUnread` on m = 22's shape: w's slot 0 is a claim's ring
switch's (`.rs 0 0`) and slots 362–365 the running sums' (`.acc0 0` to `.acc1 1`), each named unless read in all 128
bits, and slot 256 (Ligerito's first) is neither.
-/
import Flock.RecOpening

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

def rsUnread (W : Array Nat) (reads : List (List Nat)) : Option (Nat × Nat) :=
  match RecOpening.shape? 22 2 with
  | some sh => RecOpening.rsUnread sh W (RecResiduals.readIns (net (8 * W.size) reads)) (W.toList.map fun _ => 1)
  | none => some (0, 0)

#guard RecResiduals.rowCol [1, 1] 0 5 == 5
#guard RecResiduals.rowCol [1, 1] 0 (128 + 5) == 16 * 64 + 5
#guard partly 16 [1, 1] [] == none
#guard partly 16 [1, 1] [List.range 128] == none
#guard partly 16 [1, 1] [List.range 64, List.range' 64 64] == none
#guard partly 16 [1, 1] [List.range 64] == some (0, 64)
#guard partly 16 [1, 1] [List.range 128, [1024 + 7]] == some (1, 1)
#guard partly 16 [1, 1] [List.range 128, List.range' 1024 128] == none
#guard partly 16 [1, 1] [List.range' 1024 127] == some (1, 127)
#guard partly 24 [1, 1, 1] [List.range 128, List.range' 2048 128] == none
#guard partly 24 [1, 1, 1] [List.range 128, List.range' 2048 100] == some (2, 100)
#guard (RecResiduals.readIns (net 16 [[3, 2048 + 1]])).size == 2048
#guard (RecResiduals.readIns (net 16 [[2048]])).toList.all (!·)
#guard rsUnread #[0] [List.range 128] == none
#guard rsUnread #[0] [List.range 100] == some (0, 100)
#guard rsUnread #[0] [] == some (0, 0)
#guard rsUnread #[256, 0] [List.range' 1024 128] == none
#guard rsUnread #[256, 0] [List.range 128] == some (1, 0)
#guard (RecOpening.shape? 22 2).map (fun sh => [.acc0 0, .acc0 1, .acc1 0, .acc1 1].map (RecOpening.wi sh)) ==
  some [some 362, some 363, some 364, some 365]
#guard rsUnread #[362] [List.range 128] == none
#guard rsUnread #[363] [List.range 100] == some (0, 100)
#guard rsUnread #[364] [] == some (0, 0)
#guard rsUnread #[256, 365] [List.range 128] == some (1, 0)
#guard rsUnread #[362, 363, 364, 365] [List.range 128, List.range' 1024 128, List.range' 2048 128] == some (3, 0)

end Flock.Test.ReadWhole
