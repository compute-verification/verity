import Definitions.Core.NaiveStrassen
import Proofs.Core.NaiveStrassen.Grid

/-!
# The tables (NaiveStrassen's kernel tables)

Level `l`'s tables bound, at each pair of grid cells `(i, j)`, the least cost at depth budget `l` of a schedule whose
operands range over at least `g i` and `g j` values: `lo` over every schedule and `has` over those with a fast
scheme. Each is a packed `sz l × sz l` table (`Pack.lean`); row and column 0 are zero, and entries beyond the table read
as zero.

This file only computes them, by the options of the model: an atom of the RTX PRO 6000's data sheet, a scheme over the
previous level, and digit splits, relaxed by a fixed number of passes (`passes`) of splits of `A` followed by a
symmetrisation (splits of `B`). Nothing here is trusted: `Check.lean` checks the result, and only the check is proved
sound.
-/

namespace Verity.Accel.NaiveStrassen.Packed

open Verity.Accel.NaiveStrassen

/-- Fieldwise minimum of two rows of `n` fields below `2^63`. -/
def minRow (n X Y : Nat) : Nat :=
  let m := (((X + guards n) - Y) &&& guards n) >>> 63
  let k := m * 18446744073709551615
  X - (X &&& k) + (Y &&& k)

/-- `T` with row `i` replaced by `R`. -/
def setRow (T n i R : Nat) : Nat := T - (row T n i <<< (64 * n * i)) + (R <<< (64 * n * i))

/-- `f` folded over `k ∈ [a, b)`. -/
def foldRange {α : Type} (a b : Nat) (f : Nat → α → α) (init : α) : α :=
  Nat.rec (motive := fun _ => α) init (fun k acc => f (a + k) acc) (b - a)

/-- `f` applied `k` times. -/
def iter {α : Type} (k : Nat) (f : α → α) (x : α) : α := Nat.rec (motive := fun _ => α) x (fun _ acc => f acc) k

/-- An entry no table reaches (`2^56 - 1`). -/
def CAP : Nat := 72057594037927935
/-- The atom table's entry where no atom holds. -/
def BIG : Nat := 65536

/-- The least weight of an atom of `fs` that holds operands over `a` and `b` values (`BIG` if none). -/
def atomMin (fs : List Atom) (a b : Nat) : Nat :=
  fs.foldl (fun acc f => if f.holds a b then min acc f.weight8 else acc) BIG

/-- The atom table on the whole grid, on the RTX PRO 6000's data sheet. -/
@[irreducible] def atomT : Nat := packRows (fun i => packF (fun j => atomMin rtx6000Sheet (g i) (g j)) gsz) gsz gsz

/-- Strassen's products' `(‖u‖₁, ‖v‖₁)`. -/
def strassenP : List (Nat × Nat) := [(2, 2), (2, 1), (1, 2), (1, 2), (2, 1), (2, 2), (2, 2)]
/-- Winograd's products' `(‖u‖₁, ‖v‖₁)`. -/
def winogradP : List (Nat × Nat) := [(1, 1), (1, 1), (4, 1), (1, 4), (2, 2), (3, 3), (2, 2)]

/-- The previous level's `lo` (`m` rows), with each row's fields gathered at `j ↦ kc j μ`, `n` fields a row. -/
def gatherT (prev m n μ : Nat) : Nat := packRows (fun k => packF (fun j => fld (row prev m k) (kc j μ)) n) n m

/-- Row `i` of a scheme's option: the sum over its products of the gathered rows at the kid cells. -/
def schemeRow (prev m n i : Nat) (prof : List (Nat × Nat)) : Nat :=
  prof.foldl (fun acc p => acc + row (gatherT prev m n p.2) n (kc i p.1)) 0

/-- `CAP` in each of `n` fields. -/
def capRow (n : Nat) : Nat := CAP * packF (fun _ => 1) n

/-- A row with field 0 cleared. -/
def zero0 (R : Nat) : Nat := R - (R &&& 18446744073709551615)

/-- Level `l`'s tables (size `n`) before relaxation, from the previous level's (size `m`). -/
def initLevel (l m n : Nat) (prev : Nat × Nat) : Nat × Nat :=
  let lo := packRows (fun i =>
    if i = 0 then 0 else zero0 <|
    let a := minRow n (capRow n) (8 ^ l * (row atomT gsz i &&& rowMask n))
    if l = 0 then a else
    let ps := row prev.1 m i &&& rowMask n
    minRow n (minRow n a (8 * ps))
      (minRow n (schemeRow prev.1 m n i strassenP) (schemeRow prev.1 m n i winogradP))) n n
  let has := packRows (fun i =>
    if i = 0 then 0 else zero0 <|
    if l = 0 then capRow n else
    let ps := row prev.1 m i &&& rowMask n
    let ph := row prev.2 m i &&& rowMask n
    minRow n (capRow n) (minRow n (7 * ps + ph)
      (minRow n (schemeRow prev.1 m n i strassenP) (schemeRow prev.1 m n i winogradP)))) n n
  (lo, has)

/-- One pass of splits of `A` over rows `2 ≤ i < n`, each row against rows `e` and `rest i e` for `1 ≤ e < i`. -/
def splitPass (n : Nat) (t : Nat × Nat) : Nat × Nat :=
  foldRange 2 n (fun i (s : Nat × Nat) =>
    let r := foldRange 1 i (fun e (acc : Nat × Nat) =>
      let te := row s.1 n e
      let tr := row s.1 n (rest i e)
      let he := row s.2 n e
      let hr := row s.2 n (rest i e)
      (minRow n acc.1 (te + tr), minRow n (minRow n acc.2 (he + tr)) (te + hr))) (row s.1 n i, row s.2 n i)
    let h := minRow n r.2 (r.1 + row s.2 n 1)
    (setRow s.1 n i r.1, setRow s.2 n i h)) t

/-- The transpose of an `n × n` table. -/
def transpose (T n : Nat) : Nat := packRows (fun i => packF (fun j => fld (row T n j) i) n) n n

/-- The fieldwise minimum of a table and its transpose (splits of `B`). -/
def symm (T n : Nat) : Nat :=
  let U := transpose T n
  packRows (fun i => minRow n (row T n i) (row U n i)) n n

/-- One relaxation pass. -/
def pass (n : Nat) (t : Nat × Nat) : Nat × Nat :=
  let s := splitPass n t
  (symm s.1 n, symm s.2 n)

/-- Level `l`'s tables from the previous level's. -/
@[irreducible] def level (l : Nat) (prev : Nat × Nat) : Nat × Nat :=
  iter (passes.getD l 0) (pass (sz l)) (initLevel l (sz (l - 1)) (sz l) prev)

end Verity.Accel.NaiveStrassen.Packed
