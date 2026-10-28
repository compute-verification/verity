import Definitions.Accounting.Guarantees.TS

/-!
# The tensor–scalar model's price lemmas

The certificate's price against the bound, the stars' signs, (RD) in price form and the identity of Theorem D(a)
(EPS1-SPEC §1.1, §2.2).
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

private theorem price_add (G : FacetCosts K R) (lam : R → ℝ) (f g : K → ℝ) :
    G.price lam (fun k => f k + g k) = G.price lam f + G.price lam g := by
  simp only [FacetCosts.price, FacetCosts.load, mul_add, Finset.sum_add_distrib]

private theorem price_sum {ι : Type} (G : FacetCosts K R) (lam : R → ℝ) (s : Finset ι) (f : ι → K → ℝ) :
    G.price lam (fun k => ∑ i ∈ s, f i k) = ∑ i ∈ s, G.price lam (f i) := by
  induction s using Finset.cons_induction with
  | empty => simp [FacetCosts.price, FacetCosts.load]
  | cons a s ha ih => simp only [Finset.sum_cons]; rw [price_add, ih]

private theorem price_mono (G : FacetCosts K R) (lam : R → ℝ) {f g : K → ℝ} (ha : ∀ r k, 0 ≤ G.a r k)
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
theorem price_le_bound [Nonempty R] (G : FacetCosts K R) (lam : R → ℝ) (n : K → ℝ) (h : IsCert lam) :
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
  obtain ⟨ha, -, -, -, hom, hq, -, hbT, hside, hfold, hfill, -⟩ := hZ
  obtain ⟨hlam, -⟩ := h
  have hd : ∀ hd r, 0 ≤ Z.demand hd r := by
    intro hd r
    cases hd with
    | writer t => exact div_nonneg (ha r _) (hom t).le
    | vector => exact mul_nonneg (div_nonneg hq (by norm_num)) (vecMin_nonneg Z ha r)
    | table l => exact le_max_of_le_right (mul_nonneg hbT (ha r _))
    | side u => exact Finset.sum_nonneg fun k _ => mul_nonneg (ha r k) (hside u k)
  have hp : ∀ d : R → ℝ, (∀ r, 0 ≤ d r) → 0 ≤ demandPrice lam d := fun d hd =>
    Finset.sum_nonneg fun r _ => mul_nonneg (hlam r) (hd r)
  exact ⟨hd, Real.iInf_nonneg fun hd' => hp _ (hd hd'), Real.iInf_nonneg fun φ => hp _ (hfold φ),
    Real.iInf_nonneg fun ψ => hp _ (hfill ψ)⟩

/-- Each head's words dominate its route facet by facet, from (I2)/(I4) `WordsWritten`, (I3) `ScalarRoute` and (I5)
`SideUnits` (the per-head part of (RD)). -/
theorem head_dominates (Z : Model K R T L U F X) (P : Labelled K) (A : Anatomy K T L U) (hZ : Z.Wf P)
    (hA : A.Split) (hW : WordsWritten Z A) (hS : ScalarRoute Z A) (hU : SideUnits Z A) :
    ∀ hd r, A.words hd * Z.demand hd r ≤ Z.G.load (A.nword hd) r := by
  obtain ⟨ha, -, -, -, hom, -⟩ := hZ
  obtain ⟨hnw, -, -, hwd, -⟩ := hA
  have hterm : ∀ hd r k, Z.G.a r k * A.nword hd k ≤ Z.G.load (A.nword hd) r := fun hd r k =>
    Finset.single_le_sum (f := fun k => Z.G.a r k * A.nword hd k)
      (fun k _ => mul_nonneg (ha r k) (hnw hd k)) (Finset.mem_univ k)
  have hvec : ∀ hd r, (∑ k ∈ Z.vec, A.nword hd k) * Z.vecMin r ≤ Z.G.load (A.nword hd) r := by
    intro hd r
    calc (∑ k ∈ Z.vec, A.nword hd k) * Z.vecMin r = ∑ k ∈ Z.vec, A.nword hd k * Z.vecMin r := by
          rw [Finset.sum_mul]
      _ ≤ ∑ k ∈ Z.vec, Z.G.a r k * A.nword hd k := Finset.sum_le_sum fun k hk => by
          rw [mul_comm]; exact mul_le_mul_of_nonneg_right (vecMin_le Z r hk) (hnw hd k)
      _ ≤ ∑ k, Z.G.a r k * A.nword hd k :=
          Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _)
            fun k _ _ => mul_nonneg (ha r k) (hnw hd k)
  have hvm : ∀ r, 0 ≤ Z.vecMin r / 32 := fun r => div_nonneg (vecMin_nonneg Z ha r) (by norm_num)
  intro hd r
  cases hd with
  | writer t =>
    simp only [Model.demand]
    calc A.words (.writer t) * (Z.G.a r (Z.cls t) / Z.omega t)
        ≤ Z.omega t * A.nword (.writer t) (Z.cls t) * (Z.G.a r (Z.cls t) / Z.omega t) :=
          mul_le_mul_of_nonneg_right (hW t) (div_nonneg (ha r _) (hom t).le)
      _ = Z.G.a r (Z.cls t) * A.nword (.writer t) (Z.cls t) := by
          rw [mul_comm (Z.omega t), mul_assoc, mul_div_cancel₀ _ (hom t).ne', mul_comm]
      _ ≤ _ := hterm _ r _
  | vector =>
    simp only [Model.demand]
    calc A.words .vector * (Z.q / 32 * Z.vecMin r) = Z.q * A.words .vector * (Z.vecMin r / 32) := by ring
      _ ≤ 32 * (∑ k ∈ Z.vec, A.nword .vector k) * (Z.vecMin r / 32) :=
          mul_le_mul_of_nonneg_right hS.1 (hvm r)
      _ = (∑ k ∈ Z.vec, A.nword .vector k) * Z.vecMin r := by ring
      _ ≤ _ := hvec _ r
  | table l =>
    simp only [Model.demand]
    obtain ⟨hql, hbl⟩ := hS.2 l
    rw [mul_max_of_nonneg _ _ (hwd _)]
    refine max_le ?_ ?_
    · calc A.words (.table l) * (Z.qT / 32 * Z.vecMin r)
          = Z.qT * A.words (.table l) * (Z.vecMin r / 32) := by ring
        _ ≤ 32 * (∑ k ∈ Z.vec, A.nword (.table l) k) * (Z.vecMin r / 32) :=
            mul_le_mul_of_nonneg_right hql (hvm r)
        _ = (∑ k ∈ Z.vec, A.nword (.table l) k) * Z.vecMin r := by ring
        _ ≤ _ := hvec _ r
    · calc A.words (.table l) * (Z.bT * Z.G.a r (Z.bytes l))
          = Z.bT * A.words (.table l) * Z.G.a r (Z.bytes l) := by ring
        _ ≤ A.nword (.table l) (Z.bytes l) * Z.G.a r (Z.bytes l) :=
            mul_le_mul_of_nonneg_right hbl (ha r _)
        _ = Z.G.a r (Z.bytes l) * A.nword (.table l) (Z.bytes l) := mul_comm _ _
        _ ≤ _ := hterm _ r _
  | side u =>
    simp only [Model.demand, FacetCosts.load, Finset.mul_sum]
    exact Finset.sum_le_sum fun k _ =>
      calc A.words (.side u) * (Z.G.a r k * Z.side u k) = Z.G.a r k * (A.words (.side u) * Z.side u k) := by
            ring
        _ ≤ Z.G.a r k * A.nword (.side u) k := mul_le_mul_of_nonneg_left (hU u k) (ha r k)

/-- (RD) in price form from the parts (I2)–(I6), FOLD-OPT per word folded at the certificate: at every certificate
`π` where it holds, `fixed·w* + (1 − μ_f)·folded·φ* + (1 − μ_x)χ* ≤ π·n(M)`. -/
theorem price_of_parts_folded [Nonempty F] [Nonempty X] (Z : Model K R T L U F X) (P : Labelled K)
    (A : Anatomy K T L U) (lam : R → ℝ) (folded : ℝ) (h : IsCert lam) (hZ : Z.Wf P) (hA : A.Split)
    (hW : WordsWritten Z A) (hS : ScalarRoute Z A) (hU : SideUnits Z A) (hN : NoSum A)
    (hF : FoldOptFoldedAt Z A folded lam) (hX : FillOpt Z A) :
    A.fixed * Z.wStar lam + (1 - Z.muF) * folded * Z.phiStar lam + (1 - Z.muX) * Z.chiStar lam ≤
      Z.G.price lam A.run.n := by
  obtain ⟨-, hw0, -, -⟩ := stars_nonneg Z P lam hZ h
  have hdom := head_dominates Z P A hZ hA hW hS hU
  have hfold : (1 - Z.muF) * folded * Z.phiStar lam ≤ Z.G.price lam A.nfold := hF
  have hfill := hX lam h
  obtain ⟨ha, -⟩ := hZ
  obtain ⟨hlam, -⟩ := h
  obtain ⟨-, -, -, hwd, hsplit⟩ := hA
  have hword : ∀ hd, A.words hd * Z.wStar lam ≤ Z.G.price lam (A.nword hd) := by
    intro hd
    calc A.words hd * Z.wStar lam ≤ A.words hd * demandPrice lam (Z.demand hd) :=
          mul_le_mul_of_nonneg_left
            (ciInf_le (Set.finite_range fun hd => demandPrice lam (Z.demand hd)).bddBelow hd) (hwd hd)
      _ = ∑ r, lam r * (A.words hd * Z.demand hd r) := by
          simp only [demandPrice, Finset.mul_sum]
          exact Finset.sum_congr rfl fun _ _ => by ring
      _ ≤ ∑ r, lam r * Z.G.load (A.nword hd) r :=
          Finset.sum_le_sum fun r _ => mul_le_mul_of_nonneg_left (hdom hd r) (hlam r)
      _ = Z.G.price lam (A.nword hd) := rfl
  calc A.fixed * Z.wStar lam + (1 - Z.muF) * folded * Z.phiStar lam + (1 - Z.muX) * Z.chiStar lam
      ≤ (∑ hd, A.words hd) * Z.wStar lam + Z.G.price lam A.nfold + Z.G.price lam A.nfill :=
        add_le_add (add_le_add (mul_le_mul_of_nonneg_right hN hw0) hfold) hfill
    _ = ∑ hd, A.words hd * Z.wStar lam + Z.G.price lam A.nfold + Z.G.price lam A.nfill := by
        rw [Finset.sum_mul]
    _ ≤ ∑ hd, Z.G.price lam (A.nword hd) + Z.G.price lam A.nfold + Z.G.price lam A.nfill :=
        add_le_add (add_le_add (Finset.sum_le_sum fun hd _ => hword hd) le_rfl) le_rfl
    _ = Z.G.price lam (fun k => ∑ hd, A.nword hd k + A.nfold k + A.nfill k) := by
        rw [price_add, price_add, price_sum]
    _ ≤ Z.G.price lam A.run.n := price_mono Z.G lam ha hlam hsplit

/-- (RD) in price form from the parts (I2)–(I6): at every certificate,
`fixed·w* + (1 − μ_f)N_w φ* + (1 − μ_x)χ* ≤ π·n(M)`. -/
theorem price_of_parts [Nonempty F] [Nonempty X] (Z : Model K R T L U F X) (P : Labelled K)
    (A : Anatomy K T L U) (lam : R → ℝ) (h : IsCert lam) (hZ : Z.Wf P) (hA : A.Split)
    (hW : WordsWritten Z A) (hS : ScalarRoute Z A) (hU : SideUnits Z A) (hN : NoSum A) (hF : FoldOpt Z A)
    (hX : FillOpt Z A) :
    A.fixed * Z.wStar lam + (1 - Z.muF) * Z.Nw * Z.phiStar lam + (1 - Z.muX) * Z.chiStar lam ≤
      Z.G.price lam A.run.n :=
  price_of_parts_folded Z P A lam Z.Nw h hZ hA hW hS hU hN (hF lam h) hX

/-- (RD) in its per-facet form gives the same price bound (step 2 of Theorem D(a)). -/
theorem price_of_rd [Nonempty F] [Nonempty X] (Z : Model K R T L U F X) (P : Labelled K) (n : K → ℝ)
    (fixed : ℝ) (lam : R → ℝ) (h : IsCert lam) (hZ : Z.Wf P) (hRD : Z.RouteDominated n fixed) :
    fixed * Z.wStar lam + (1 - Z.muF) * Z.Nw * Z.phiStar lam + (1 - Z.muX) * Z.chiStar lam ≤ Z.G.price lam n := by
  obtain ⟨-, hw0, hphi0, hchi0⟩ := stars_nonneg Z P lam hZ h
  obtain ⟨hlam, -⟩ := h
  obtain ⟨N, Mf, Mx, hN, hMf, hMx, hfix, hsF, hsX, hdom⟩ := hRD
  have hw : ∀ hd, Z.wStar lam ≤ demandPrice lam (Z.demand hd) := fun hd =>
    ciInf_le (Set.finite_range fun hd => demandPrice lam (Z.demand hd)).bddBelow hd
  have hphi : ∀ φ, Z.phiStar lam ≤ demandPrice lam (Z.fold φ) := fun φ =>
    ciInf_le (Set.finite_range fun φ => demandPrice lam (Z.fold φ)).bddBelow φ
  have hchi : ∀ ψ, Z.chiStar lam ≤ demandPrice lam (Z.fill ψ) := fun ψ =>
    ciInf_le (Set.finite_range fun ψ => demandPrice lam (Z.fill ψ)).bddBelow ψ
  calc fixed * Z.wStar lam + (1 - Z.muF) * Z.Nw * Z.phiStar lam + (1 - Z.muX) * Z.chiStar lam
      ≤ (∑ hd, N hd) * Z.wStar lam + (∑ φ, Mf φ) * Z.phiStar lam + (∑ ψ, Mx ψ) * Z.chiStar lam :=
        add_le_add (add_le_add (mul_le_mul_of_nonneg_right hfix hw0) (mul_le_mul_of_nonneg_right hsF hphi0))
          (mul_le_mul_of_nonneg_right hsX hchi0)
    _ = ∑ hd, N hd * Z.wStar lam + ∑ φ, Mf φ * Z.phiStar lam + ∑ ψ, Mx ψ * Z.chiStar lam := by
        simp only [Finset.sum_mul]
    _ ≤ ∑ hd, N hd * demandPrice lam (Z.demand hd) + ∑ φ, Mf φ * demandPrice lam (Z.fold φ) +
          ∑ ψ, Mx ψ * demandPrice lam (Z.fill ψ) :=
        add_le_add (add_le_add
          (Finset.sum_le_sum fun hd _ => mul_le_mul_of_nonneg_left (hw hd) (hN hd))
          (Finset.sum_le_sum fun φ _ => mul_le_mul_of_nonneg_left (hphi φ) (hMf φ)))
          (Finset.sum_le_sum fun ψ _ => mul_le_mul_of_nonneg_left (hchi ψ) (hMx ψ))
    _ = ∑ r, lam r * (∑ hd, N hd * Z.demand hd r + ∑ φ, Mf φ * Z.fold φ r + ∑ ψ, Mx ψ * Z.fill ψ r) := by
        simp only [sum_demandPrice, mul_add, Finset.sum_add_distrib]
    _ ≤ ∑ r, lam r * Z.G.load n r :=
        Finset.sum_le_sum fun r _ => mul_le_mul_of_nonneg_left (hdom r) (hlam r)
    _ = Z.G.price lam n := rfl

/-- Step 4 of Theorem D(a), the identity: `Φ(π)/B(L) = 1 − ε_cert − ε_skip − ε_route − ε_mand − ε_luck − ε_adapt`
for any labelled program `L`, with `Φ(π) = (1 − d̄ − ā)N_w w* + (1 − μ_f)N_w φ* + (1 − μ_x)χ*`. -/
theorem identity (Z : Model K R T L U F X) (P : Labelled K) (lam : R → ℝ) (dbar abar : ℝ) (hZ : Z.Wf P) :
    ((1 - dbar - abar) * Z.Nw * Z.wStar lam + (1 - Z.muF) * Z.Nw * Z.phiStar lam +
        (1 - Z.muX) * Z.chiStar lam) / Z.bnd P =
      1 - Z.epsCert P lam - Z.epsSkip P lam - Z.epsRoute P lam - Z.epsMand P lam - Z.epsLuck P lam dbar -
        Z.epsAdapt P lam abar := by
  obtain ⟨-, -, hNw, hB, -⟩ := hZ
  have hn : Z.G.price lam P.n =
      Z.G.price lam P.nw + Z.G.price lam P.nf + Z.G.price lam P.nx + Z.G.price lam P.ns := by
    change Z.G.price lam (fun k => P.nw k + P.nf k + P.nx k + P.ns k) = _
    rw [price_add, price_add, price_add]
  have hNw' := hNw.ne'
  have hB' := hB.ne'
  simp only [Model.epsCert, Model.epsSkip, Model.epsRoute, Model.epsMand, Model.epsLuck, Model.epsAdapt,
    Model.kappa, hn]
  field_simp
  ring

end Lemmas

end Accounting.SecurityProofs.TS
