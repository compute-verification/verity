import Definitions.Accounting.Headline.Encoding

/-!
# Reading the headline encoding

The lemmas a headline instance's proof needs to read its exact data (`Definitions.Accounting.Headline.Encoding`) as
the model's reals:

* `load_counts`: a load of sparse counts, on a machine whose rows are rationals `c r`, is the rational `dot (c r) n v`,
  a sum over the list's entries (each entry's class below `n`), which `decide` evaluates.
* `bound_eq`: the bound `B(n)` is the facet `r0`'s load when no facet's is larger.
* `sheet_a`: a sheet's machine's rows are `sheetCoef`'s rationals.
* `labelled_n`: a labelled program's counts are the counts of its four lists' concatenation.
* `demandPrice_onFacet`, `isCert_onFacet`: the certificate on one facet prices a demand at that facet's.
-/

namespace Accounting.SecurityProofs.TS.Headline

open Accounting.TS Accounting.TS.Headline

/-- `Σ_{i<n} c_i · v_i` over a sparse list's entries: each entry below `n` once, at its rational. -/
def dot (c : ℕ → ℚ) (n : ℕ) (v : Sparse) : ℚ := (v.map fun e => if e.1 < n then c e.1 * Sparse.ratio e else 0).sum

theorem sum_range_mul_coeff (n : ℕ) (c : ℕ → ℚ) (v : Sparse) :
    ∑ i ∈ Finset.range n, c i * Sparse.coeff v i = dot c n v := by
  induction v with
  | nil => simp [Sparse.coeff, dot]
  | cons e v ih =>
    rw [show dot c n (e :: v) = (if e.1 < n then c e.1 * Sparse.ratio e else 0) + dot c n v by
      simp only [dot, List.map_cons, List.sum_cons], ← ih]
    simp only [Sparse.coeff, List.map_cons, List.sum_cons, mul_add, Finset.sum_add_distrib]
    congr 1
    rcases Nat.lt_or_ge e.1 n with h | h
    · simp [mul_ite, h]
    · simp [mul_ite, not_lt.mpr h]

theorem sum_mul_coeff (n : ℕ) (c : ℕ → ℚ) (v : Sparse) :
    ∑ k : Fin n, c k * Sparse.coeff v k = dot c n v :=
  (Fin.sum_univ_eq_sum_range (fun i => c i * Sparse.coeff v i) n).trans (sum_range_mul_coeff n c v)

/-- A load of sparse counts on rational rows. -/
theorem load_counts {n : ℕ} {R : Type} [Fintype R] (G : FacetCosts (Fin n) R) (c : R → ℕ → ℚ)
    (hG : ∀ r k, G.a r k = (c r k.val : ℝ)) (v : Sparse) (r : R) :
    G.load (Sparse.counts v) r = (dot (c r) n v : ℝ) := by
  rw [← sum_mul_coeff]
  simp only [FacetCosts.load, hG, Sparse.counts, Rat.cast_sum, Rat.cast_mul]

theorem coeff_nonneg (v : Sparse) (i : ℕ) : 0 ≤ Sparse.coeff v i := by
  unfold Sparse.coeff
  refine List.sum_nonneg fun x hx => ?_
  obtain ⟨e, -, rfl⟩ := List.mem_map.mp hx
  split_ifs
  · exact div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)
  · exact le_rfl

theorem load_counts_nonneg {n : ℕ} {R : Type} [Fintype R] (G : FacetCosts (Fin n) R) (hG : ∀ r k, 0 ≤ G.a r k)
    (v : Sparse) (r : R) : 0 ≤ G.load (Sparse.counts v) r := by
  unfold FacetCosts.load
  exact Finset.sum_nonneg fun k _ => mul_nonneg (hG r k) (Rat.cast_nonneg.mpr (coeff_nonneg v k.val))

theorem coeff_append (v w : Sparse) (i : ℕ) : Sparse.coeff (v ++ w) i = Sparse.coeff v i + Sparse.coeff w i := by
  simp only [Sparse.coeff, List.map_append, List.sum_append]

/-- A labelled program's counts: its four lists' concatenation. -/
theorem labelled_n (n : ℕ) (nw nf nx ns : Sparse) :
    (labelled n nw nf nx ns).n = Sparse.counts (nw ++ nf ++ nx ++ ns) := by
  funext k
  simp only [Labelled.n, labelled, Sparse.counts, coeff_append, Rat.cast_add]

/-- A sheet's rows as rationals, by class index: `Sheet.machine`'s, at `SheetData.sheet`'s lists. -/
def sheetCoef {P : Type} [DecidableEq P] (D : SheetData P) : SheetRow P → ℕ → ℚ
  | .issue, _ => 1
  | .tensor, i => (D.macs.getD i 0 : ℚ) / ((D.peak.getD i 0 : ℚ) / 4)
  | .pipe p, i => if D.pipe.getD i none = some p then 128 / (D.width p : ℚ) else 0
  | .smem, i => (D.smem.getD i 0 : ℚ) / 32

theorem sheet_a {P : Type} [DecidableEq P] (D : SheetData P) (n : ℕ) (r : SheetRow P) (k : Fin n) :
    (D.sheet n).machine.a r k = (sheetCoef D r k.val : ℝ) := by
  cases r with
  | pipe p =>
    show (if D.pipe.getD k.val none = some p then (128 : ℝ) / (D.width p : ℝ) else 0) =
      ((if D.pipe.getD k.val none = some p then (128 : ℚ) / (D.width p : ℚ) else 0 : ℚ) : ℝ)
    by_cases h : D.pipe.getD k.val none = some p
    · rw [if_pos h, if_pos h, Rat.cast_div, Rat.cast_ofNat, Rat.cast_natCast]
    · rw [if_neg h, if_neg h, Rat.cast_zero]
  | issue => simp only [Sheet.machine, SheetData.sheet, sheetCoef] <;> simp
  | tensor => simp only [Sheet.machine, SheetData.sheet, sheetCoef] <;> simp
  | smem => simp only [Sheet.machine, SheetData.sheet, sheetCoef] <;> simp

theorem sheetCoef_nonneg {P : Type} [DecidableEq P] (D : SheetData P) (r : SheetRow P) (i : ℕ) :
    0 ≤ sheetCoef D r i := by
  cases r <;> simp only [sheetCoef] <;> (try split_ifs) <;> positivity

/-- The bound is facet `r0`'s load when no facet's is larger. -/
theorem bound_eq {n : ℕ} {R : Type} [Fintype R] [Nonempty R] (G : FacetCosts (Fin n) R) (c : R → ℕ → ℚ)
    (hG : ∀ r k, G.a r k = (c r k.val : ℝ)) (v : Sparse) (r0 : R) (h : ∀ r, dot (c r) n v ≤ dot (c r0) n v) :
    G.bound (Sparse.counts v) = (dot (c r0) n v : ℝ) := by
  unfold FacetCosts.bound
  simp only [load_counts G c hG v]
  exact le_antisymm (ciSup_le fun r => Rat.cast_le.mpr (h r))
    (le_ciSup (f := fun r => ((dot (c r) n v : ℚ) : ℝ)) (Set.finite_range _).bddAbove r0)

theorem demandPrice_onFacet {R : Type} [Fintype R] [DecidableEq R] (r0 : R) (d : R → ℝ) :
    demandPrice (onFacet r0) d = d r0 := by
  simp [demandPrice, onFacet, ite_mul]

theorem isCert_onFacet {R : Type} [Fintype R] [DecidableEq R] (r0 : R) : IsCert (onFacet r0) := by
  refine ⟨fun r => ?_, ?_⟩
  · simp only [onFacet]
    split_ifs <;> norm_num
  · simp [onFacet]

end Accounting.SecurityProofs.TS.Headline
