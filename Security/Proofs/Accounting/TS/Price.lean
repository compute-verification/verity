import Models.Accounting.Guarantees.TS

/-!
# The tensor–scalar model's price lemmas

The certificate's price against the bound, the stars' signs, (RD) in price form and the identity of Theorem D(a)
(EPS1-SPEC §1.1, §2.2): gap list G1–G6 of note:20261017T0801Z-draft-lean-stmt.
-/

namespace Accounting.SecurityProofs.TS

open Accounting.TS Accounting.Assumptions.TS Accounting.Guarantees.TS Pouw.RelativeSecurity

-- The frozen statements take every section `Fintype` binder, whether or not their proofs use it.
set_option linter.unusedSectionVars false

section Lemmas

variable {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F]
  [Fintype X]

private theorem vecMin_nonneg (Z : Model K R T L U F X) (ha : ∀ r k, 0 ≤ Z.G.a r k) (r : R) :
    0 ≤ Z.vecMin r :=
  Real.iInf_nonneg fun k => ha r k

private theorem vecMin_le (Z : Model K R T L U F X) (r : R) {k : K} (hk : k ∈ Z.vec) :
    Z.vecMin r ≤ Z.G.a r k :=
  ciInf_le (Set.finite_range fun k : Z.vec => Z.G.a r k).bddBelow ⟨k, hk⟩

private theorem price_add (G : Machine K R) (lam : R → ℝ) (f g : K → ℝ) :
    G.price lam (fun k => f k + g k) = G.price lam f + G.price lam g := by
  simp only [Machine.price, Machine.load, mul_add, Finset.sum_add_distrib]

private theorem price_sum {ι : Type} (G : Machine K R) (lam : R → ℝ) (s : Finset ι) (f : ι → K → ℝ) :
    G.price lam (fun k => ∑ i ∈ s, f i k) = ∑ i ∈ s, G.price lam (f i) := by
  induction s using Finset.cons_induction with
  | empty => simp [Machine.price, Machine.load]
  | cons a s ha ih => simp only [Finset.sum_cons]; rw [price_add, ih]

private theorem price_mono (G : Machine K R) (lam : R → ℝ) {f g : K → ℝ} (ha : ∀ r k, 0 ≤ G.a r k)
    (hlam : ∀ r, 0 ≤ lam r) (hfg : ∀ k, f k ≤ g k) : G.price lam f ≤ G.price lam g :=
  Finset.sum_le_sum fun r _ => mul_le_mul_of_nonneg_left
    (Finset.sum_le_sum fun k _ => mul_le_mul_of_nonneg_left (hfg k) (ha r k)) (hlam r)

/-- `Σ_i c_i (π·d_i) = Σ_r λ_r Σ_i c_i d_{i,r}`. -/
private theorem sum_demandPrice {ι : Type} [Fintype ι] (lam : R → ℝ) (c : ι → ℝ) (d : ι → R → ℝ) :
    ∑ i, c i * demandPrice lam (d i) = ∑ r, lam r * ∑ i, c i * d i r := by
  simp only [demandPrice, Finset.mul_sum]
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun _ _ => Finset.sum_congr rfl fun _ _ => by ring

/-- Step 1 of Theorem D(a): a certificate's price is at most the bound, `π·n ≤ B(n)`. -/
theorem price_le_bound [Nonempty R] (G : Machine K R) (lam : R → ℝ) (n : K → ℝ) (h : IsCert lam) :
    G.price lam n ≤ G.bound n := by
  obtain ⟨hlam, hsum⟩ := h
  have hle : ∀ r, G.load n r ≤ G.bound n := fun r =>
    le_ciSup (Set.finite_range fun r => G.load n r).bddAbove r
  calc G.price lam n = ∑ r, lam r * G.load n r := rfl
    _ ≤ ∑ r, lam r * G.bound n :=
        Finset.sum_le_sum fun r _ => mul_le_mul_of_nonneg_left (hle r) (hlam r)
    _ = G.bound n := by rw [← Finset.sum_mul, hsum, one_mul]

/-- The demands, `w*`, `φ*` and `χ*` are `≥ 0` on a well-formed model at a certificate. -/
theorem stars_nonneg [Nonempty F] [Nonempty X] (Z : Model K R T L U F X) (P : Labelled K) (lam : R → ℝ)
    (hZ : Z.Wf P) (h : IsCert lam) :
    (∀ hd r, 0 ≤ Z.demand hd r) ∧ 0 ≤ Z.wStar lam ∧ 0 ≤ Z.phiStar lam ∧ 0 ≤ Z.chiStar lam := by
  sorry

/-- Each head's words dominate its route facet by facet, from (I2)/(I4) `WordsWritten`, (I3) `ScalarRoute` and (I5)
`SideUnits` (the per-head part of (RD)). -/
theorem head_dominates (Z : Model K R T L U F X) (P : Labelled K) (A : Anatomy K T L U) (hZ : Z.Wf P)
    (hA : A.Split) (hW : WordsWritten Z A) (hS : ScalarRoute Z A) (hU : SideUnits Z A) :
    ∀ hd r, A.words hd * Z.demand hd r ≤ Z.G.load (A.nword hd) r := by
  sorry

/-- (RD) in price form from the parts (I2)–(I6): at every certificate,
`fixed·w* + (1 − μ_f)N_w φ* + (1 − μ_x)χ* ≤ π·n(M)`. -/
theorem price_of_parts [Nonempty F] [Nonempty X] (Z : Model K R T L U F X) (P : Labelled K)
    (A : Anatomy K T L U) (lam : R → ℝ) (h : IsCert lam) (hZ : Z.Wf P) (hA : A.Split)
    (hW : WordsWritten Z A) (hS : ScalarRoute Z A) (hU : SideUnits Z A) (hN : NoSum A) (hF : FoldOpt Z A)
    (hX : FillOpt Z A) :
    A.fixed * Z.wStar lam + (1 - Z.muF) * Z.Nw * Z.phiStar lam + (1 - Z.muX) * Z.chiStar lam ≤
      Z.G.price lam A.run.n := by
  sorry

/-- (RD) in its per-facet form gives the same price bound (step 2 of Theorem D(a)). -/
theorem price_of_rd [Nonempty F] [Nonempty X] (Z : Model K R T L U F X) (P : Labelled K) (n : K → ℝ)
    (fixed : ℝ) (lam : R → ℝ) (h : IsCert lam) (hZ : Z.Wf P) (hRD : Z.RouteDominated n fixed) :
    fixed * Z.wStar lam + (1 - Z.muF) * Z.Nw * Z.phiStar lam + (1 - Z.muX) * Z.chiStar lam ≤ Z.G.price lam n := by
  sorry

/-- Step 4 of Theorem D(a), the identity: `Φ(π)/B(L) = 1 − ε_cert − ε_skip − ε_route − ε_mand − ε_luck − ε_adapt`
for any labelled program `L`, with `Φ(π) = (1 − d̄ − ā)N_w w* + (1 − μ_f)N_w φ* + (1 − μ_x)χ*`. -/
theorem identity (Z : Model K R T L U F X) (P : Labelled K) (lam : R → ℝ) (dbar abar : ℝ) (hZ : Z.Wf P) :
    ((1 - dbar - abar) * Z.Nw * Z.wStar lam + (1 - Z.muF) * Z.Nw * Z.phiStar lam +
        (1 - Z.muX) * Z.chiStar lam) / Z.bnd P =
      1 - Z.epsCert P lam - Z.epsSkip P lam - Z.epsRoute P lam - Z.epsMand P lam - Z.epsLuck P lam dbar -
        Z.epsAdapt P lam abar := by
  sorry

end Lemmas

end Accounting.SecurityProofs.TS
