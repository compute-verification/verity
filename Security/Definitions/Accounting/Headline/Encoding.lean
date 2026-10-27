import Definitions.Accounting.TS.Routes
import Definitions.Accounting.TS.Sheet

/-!
# Exact data for a headline instance (trusted)

A family's headline guarantee (`Definitions.Accounting.Guarantees.Headline`) is about one device's machine, one design's TS
model on it and its floor program ♭, each given as exact data the machine model's exporter writes
(`benchmarks/pouw/lean_headline.py`): natural numbers and rationals, never a float.

* `Sparse`: counts per class as `(class index, numerator, denominator)` triples, each entry the rational
  `numerator / denominator`; `Sparse.counts` reads them over `Fin n`, an index at or past `n` read as nothing.
* `SheetData`: a data sheet (`Accounting.TS.Sheet`) as lists by class index; `SheetData.sheet n` is the sheet over
  `Fin n`.
* `rowsMachine`: a machine whose rows are sparse lists of rational coefficients (a measured vector).
* `DesignData`: a design's writer kinds (class and `ω`), vector classes, fold routes, SCALAR-ROUTE floors and committed
  words a unit; `DesignData.model` is its TS model on a machine, with no side unit, one table level priced at
  `b_T = 0`, no filler and both slacks 0 (FOLD-OPT and FILL-OPT at `μ = 0`).
* `labelled`: a labelled program from its four parts.
-/

namespace Accounting.TS.Headline

open Accounting.TS

/-- Counts per class: `(class index, numerator, denominator)` triples. -/
abbrev Sparse := List (ℕ × ℕ × ℕ)

/-- An entry's rational, `numerator / denominator`. -/
def Sparse.ratio (e : ℕ × ℕ × ℕ) : ℚ := (e.2.1 : ℚ) / e.2.2

/-- The count of class `i`: the sum of its entries. -/
def Sparse.coeff (v : Sparse) (i : ℕ) : ℚ := (v.map fun e => if e.1 = i then Sparse.ratio e else 0).sum

/-- The counts over `Fin n`. -/
noncomputable def Sparse.counts {n : ℕ} (v : Sparse) : Fin n → ℝ := fun k => (Sparse.coeff v k.val : ℝ)

/-- Class index `i` in `Fin n` (taken mod `n`; the exporter writes indices below `n`). -/
def idx (n : ℕ) [NeZero n] (i : ℕ) : Fin n := ⟨i % n, Nat.mod_lt _ (Nat.pos_of_ne_zero (NeZero.ne n))⟩

/-- **A data sheet as lists by class index**, over pipes `P`: `Accounting.TS.Sheet`'s fields, a class past a list's end
read as `0` (or no pipe). -/
structure SheetData (P : Type) where
  fmax : ℕ
  macs : List ℕ
  peak : List ℕ
  pipe : List (Option P)
  width : P → ℕ
  smem : List ℕ

/-- The sheet over `Fin n`. -/
noncomputable def SheetData.sheet {P : Type} (D : SheetData P) (n : ℕ) : Sheet (Fin n) P where
  fmax := D.fmax
  macs k := D.macs.getD k.val 0
  peak k := D.peak.getD k.val 0
  pipe k := D.pipe.getD k.val none
  width p := D.width p
  smem k := D.smem.getD k.val 0

/-- **A machine from its rows**: row `r` the sparse coefficients `rows[r]` over `Fin n`, every facet SM-clocked. -/
noncomputable def rowsMachine (n m : ℕ) (rows : List Sparse) (fmax : ℕ) : Machine (Fin n) (Fin m) where
  a r k := (Sparse.coeff (rows.getD r.val []) k.val : ℝ)
  smClocked _ := true
  fmax := fmax

/-- **A design's data**: its writer kinds as `(class index, ω)`, its vector classes, its fold routes (warp
instructions a word, by class), the class counting the table level's bytes, SCALAR-ROUTE's floors `q` and `q_T`, and
`N_w`, the committed words a unit of honest work. -/
structure DesignData where
  writers : List (ℕ × ℕ)
  vec : List ℕ
  folds : List Sparse
  bytes : ℕ
  q : ℚ
  qT : ℚ
  Nw : ℕ

/-- **The design's TS model on machine `G`**, over `n` classes, `nt` writer kinds and `nf` fold routes: no side unit,
one table level at `b_T = 0`, the fold routes' demands their loads on `G`, no filler (its menu one route of demand
`0`), and `μ_f = μ_x = 0`. -/
noncomputable def DesignData.model {R : Type} (D : DesignData) (n nt nf : ℕ) [NeZero n] (G : Machine (Fin n) R) :
    Model (Fin n) R (Fin nt) Unit Empty (Fin nf) Unit where
  G := G
  cls t := idx n (D.writers.getD t.val (0, 0)).1
  omega t := ((D.writers.getD t.val (0, 0)).2 : ℝ)
  vec := (D.vec.map (idx n)).toFinset
  bytes _ := idx n D.bytes
  q := D.q
  qT := D.qT
  bT := 0
  side u := u.elim
  fold φ := G.load (Sparse.counts (D.folds.getD φ.val []))
  fill _ _ := 0
  Nw := D.Nw
  muF := 0
  muX := 0

/-- A labelled program over `Fin n` from its word production, fold, filler and skippable work. -/
noncomputable def labelled (n : ℕ) (nw nf nx ns : Sparse) : Labelled (Fin n) where
  nw := Sparse.counts nw
  nf := Sparse.counts nf
  nx := Sparse.counts nx
  ns := Sparse.counts ns

end Accounting.TS.Headline
