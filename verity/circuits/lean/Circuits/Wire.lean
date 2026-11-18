/-
A wire of a circuit and its positions in closed form: the one arithmetic SPOT's wiring (`Spot.Wires`) and Flock's
registered reads (`Flock.Registered`) read a wire's reader rows and writers by.

A wire states, once for many units, that reader template `reader`'s units read rows of their parameter `port` whole as
row `out` of a unit of template `writer`. Reader index `i < units` (the product of `radix`) is reader unit
`start + i / cuts`, row `cut0 + i % cuts`; its writer is unit `base + Σ_k d_k · step_k` of `writer`, `d` the digits of `i`
in mixed radix `radix` (the first digit the slowest). Each index counts its own template's units, and a writer unit is
also the writer's leaf in its template's output tree of `out`.
-/
module

@[expose] public section

namespace Circuits

/-- A wire (the module docstring); `start` is the record's `at`. -/
structure Wire where
  reader : String
  port : String
  writer : String
  out : String
  start : Nat
  radix : List Nat
  base : Int
  step : List Int
  cut0 : Nat
  cuts : Nat
  deriving Repr, BEq

namespace Wire

/-- The digits' radices and steps, the fastest digit first. -/
def pairs (w : Wire) : List (Nat × Int) := (w.radix.zip w.step).reverse

/-- The product of the radices of `ps`. -/
def prodR (ps : List (Nat × Int)) : Nat := ps.foldr (fun p n => p.1 * n) 1

/-- The reader rows the wire covers: the product of its radices. -/
def units (w : Wire) : Nat := prodR w.pairs

/-- `Σ_k d_k · step_k` for index `i`, `ps` the radices and steps fastest first (`pairs`). -/
def accum : List (Nat × Int) → Nat → Int
  | [], _ => 0
  | (r, s) :: ps, i => ((i % r : Nat) : Int) * s + accum ps (i / r)

/-- Reader index `i`'s writer unit. -/
def writerAt (w : Wire) (i : Nat) : Int := w.base + accum w.pairs i

/-- `Σ_k min 0 (step_k · (radix_k − 1))`. -/
def lowOf : List (Nat × Int) → Int
  | [] => 0
  | (r, s) :: ps => min 0 (s * ((r : Int) - 1)) + lowOf ps

/-- `Σ_k max 0 (step_k · (radix_k − 1))`. -/
def highOf : List (Nat × Int) → Int
  | [] => 0
  | (r, s) :: ps => max 0 (s * ((r : Int) - 1)) + highOf ps

/-- The least and the greatest writer the wire reads, in closed form. -/
def span (w : Wire) : Int × Int := (w.base + lowOf w.pairs, w.base + highOf w.pairs)

/-- A lower bound on `accum ps i` over indices `lo ≤ i ≤ hi`, digit by digit fastest first: where every such index has
the same higher digits, its digit's least term and theirs; else the digit's least term over all its values and the bound
on the higher digits. -/
def boxLow : List (Nat × Int) → Nat → Nat → Int
  | [], _, _ => 0
  | (r, s) :: ps, lo, hi =>
    if lo / r = hi / r then min (((lo % r : Nat) : Int) * s) (((hi % r : Nat) : Int) * s) + accum ps (lo / r)
    else min 0 (s * ((r : Int) - 1)) + boxLow ps (lo / r) (hi / r)

/-- The upper bound `boxLow` mirrors. -/
def boxHigh : List (Nat × Int) → Nat → Nat → Int
  | [], _, _ => 0
  | (r, s) :: ps, lo, hi =>
    if lo / r = hi / r then max (((lo % r : Nat) : Int) * s) (((hi % r : Nat) : Int) * s) + accum ps (lo / r)
    else max 0 (s * ((r : Int) - 1)) + boxHigh ps (lo / r) (hi / r)

/-- Bounds on the writers of reader indices `lo … hi`, in closed form (`writerAt_mem_box`). -/
def box (w : Wire) (lo hi : Nat) : Int × Int := (w.base + boxLow w.pairs lo hi, w.base + boxHigh w.pairs lo hi)

/-- Reader index `i`'s reader unit and row. -/
def readerAt (w : Wire) (i : Nat) : Nat × Nat := (w.start + i / w.cuts, w.cut0 + i % w.cuts)

/-- The reader units the wire covers. -/
def readerUnits (w : Wire) : Nat := w.units / w.cuts

/-- The first and the last reader index of the wire's reader units in units `[p0, p0 + n)` of its reader. -/
def indicesIn (w : Wire) (p0 n : Nat) : Nat × Nat :=
  ((max p0 w.start - w.start) * w.cuts, (min (p0 + n) (w.start + w.readerUnits) - w.start) * w.cuts - 1)

/-- Units `(t, at, n)` meet the wire's reader units. -/
def readsIn (w : Wire) (b : String × Nat × Nat) : Bool :=
  b.1 == w.reader && decide (max b.2.1 w.start < min (b.2.1 + b.2.2) (w.start + w.readerUnits))

/-- Whether wire `w` covers reader row `(tid, u, port, row)`, and if so its index. -/
def indexOf (w : Wire) (tid : String) (u : Nat) (port : String) (row : Nat) : Option Nat :=
  if w.reader == tid && w.port == port && w.start ≤ u && u < w.start + w.readerUnits && w.cut0 ≤ row &&
     row < w.cut0 + w.cuts then some ((u - w.start) * w.cuts + (row - w.cut0)) else none

/-- The leaf of its writer's output tree that reader unit `u`'s row `row` of the wire's port reads: its writer unit,
none where the wire doesn't cover that row or names a writer below unit 0. -/
def leafAt (w : Wire) (u row : Nat) : Option Nat :=
  (w.indexOf w.reader u w.port row).bind fun i =>
    let v := w.writerAt i
    if 0 ≤ v then some v.toNat else none

/-- Every reader row the wire covers, `(template, unit, port, row)`, with its writer `(template, unit, out)`, index by
index. -/
def entries (w : Wire) : List ((String × Nat × String × Nat) × (String × Int × String)) :=
  (List.range w.units).map fun i =>
    ((w.reader, (w.readerAt i).1, w.port, (w.readerAt i).2), (w.writer, w.writerAt i, w.out))

end Wire

end Circuits
