/-
`Sparse.foldT` runs its unboxed twin (`foldT_eq_fast`), so these guards hold it to `foldRows` on a boxed accumulator,
the code before the twin. A row whose weight is zero adds nothing, and so do a column at or past the width (which
`ofRows` refuses but a `Sparse` can hold) and a row past `e`'s length. A column whose sum is zero reads `0`, and a width
of 0 gives `#[]`.
-/
module

public import Flock.Comp
meta import Flock.Comp

@[expose] public section

namespace Flock.Test.FoldT

open Flock

/-- `foldT`'s own body, the fold on a boxed accumulator. -/
def boxed (m : Sparse) (e : Array F128) : Array F128 :=
  m.foldRows e (min m.rows e.size) 0 (Array.replicate m.width 0)

def lcg (s : UInt64) : UInt64 := s * 6364136223846793005 + 1442695040888963407

/-- `rows` rows of `k` columns each below `bound`, from seed `s`, built without `ofRows`'s check against `width`. -/
def mat (rows k width bound : Nat) (s : UInt64) : Sparse := Id.run do
  let mut data := ByteArray.empty
  let mut start := #[]
  let mut len := #[]
  let mut x := s
  for _ in [0:rows] do
    start := start.push data.size
    len := len.push k
    for _ in [0:k] do
      x := lcg x
      data := pushLe32 data ((x >>> 33).toNat % bound)
  return ⟨rows, width, data, start, len⟩

/-- Weights for `n` rows from seed `s`, every third one zero. -/
def weights (n : Nat) (s : UInt64) : Array F128 :=
  (Array.range n).map fun i => if i % 3 == 0 then 0 else ⟨lcg (s + i.toUInt64), lcg (lcg (s + i.toUInt64))⟩

#guard (mat 40 5 64 64 1).foldT (weights 40 2) == boxed (mat 40 5 64 64 1) (weights 40 2)
#guard (mat 2000 12 3000 3000 3).foldT (weights 2000 4) == boxed (mat 2000 12 3000 3000 3) (weights 2000 4)
#guard (mat 40 5 64 80 5).foldT (weights 40 6) == boxed (mat 40 5 64 80 5) (weights 40 6)
#guard (mat 40 5 64 64 7).foldT (weights 25 8) == boxed (mat 40 5 64 64 7) (weights 25 8)
#guard (mat 0 0 0 1 9).foldT #[] == #[]
#guard (mat 5 3 0 4 9).foldT (weights 5 10) == #[]

/-- Two rows holding only column 3, with the same weight, so that column 3's sum is zero. -/
def twice : Sparse := ⟨2, 8, pushLe32 (pushLe32 ByteArray.empty 3) 3, #[0, 4], #[1, 1]⟩

#guard twice.foldT #[⟨5, 6⟩, ⟨5, 6⟩] == Array.replicate 8 0
#guard twice.foldT #[⟨5, 6⟩, ⟨5, 7⟩] == (Array.replicate 8 0).set! 3 ⟨0, 1⟩

end Flock.Test.FoldT
