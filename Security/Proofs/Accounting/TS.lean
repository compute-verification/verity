import Models.Accounting.Guarantees.TS

/-!
# The tensor–scalar model's proofs (stubs)

Each guarantee of `Models.Accounting.Guarantees.TS` as `Accounting.SecurityProofs.TS.G`, and the lemmas their proofs
need, each a `sorry` until a prover takes it (the gap list: note:20261017T0801Z-draft-lean-stmt). The order is the
dependency order of EPS1-SPEC's proofs (§1.1, §2.2, §2.4, §2.7).
-/

namespace Accounting.SecurityProofs.TS

open Accounting.TS Accounting.Assumptions.TS Accounting.Guarantees.TS Pouw.RelativeSecurity

section Lemmas

variable {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F]
  [Fintype X]

/-- Step 1 of Theorem D(a): a certificate's price is at most the bound, `π·n ≤ B(n)`. -/
theorem price_le_bound [Nonempty R] (G : Machine K R) (lam : R → ℝ) (n : K → ℝ) (h : IsCert lam) :
    G.price lam n ≤ G.bound n := by
  sorry

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

/-- D-ideal's step 2, one bet a round: under (L) at `p`, the ideal game accepts with at least `k` deficient rounds
with probability at most `p^k`. -/
theorem luck_rounds {Ω : Type} [Fintype Ω] {S : ℕ} (g : RoundGame Ω S) (p : ℝ) (hp : 0 ≤ p)
    (hL : RoundGame.LuckAt p g) (k : ℕ) :
    prob (fun x => g.accepts x ∧ k ≤ Accounting.TS.RoundGame.deficientRounds g x) ≤ p ^ k := by
  sorry

/-- A round's share is at most 1, so the deficiency is at most the number of deficient rounds. -/
theorem deficiency_le_rounds {Ω : Type} [Fintype Ω] {S : ℕ} (g : RoundGame Ω S) (p : ℝ)
    (hL : RoundGame.LuckAt p g) (x : Paths Ω S) :
    g.deficiency x ≤ Accounting.TS.RoundGame.deficientRounds g x := by
  sorry

/-- Lemma C's step 1, SAMPLING's budget: when `Λ` bounds the play, the ideal game accepts with deficiency at least
`Λ·t` with probability at most `e^{−t}` (the product over rounds is a supermartingale). -/
theorem budget_tail {Ω : Type} [Fintype Ω] {S : ℕ} (g : RoundGame Ω S) (Λ : ℝ) (hΛ : 0 < Λ)
    (hB : RoundGame.BudgetAt Λ g) (t : ℝ) :
    prob (fun x => g.accepts x ∧ Λ * t ≤ g.deficiency x) ≤ Real.exp (-t) := by
  sorry

/-- The union bound: `Pr[A ∨ B] ≤ Pr[A] + Pr[B]`. -/
theorem prob_or_le {Ω : Type} [Fintype Ω] (A B : Ω → Prop) : prob (fun x => A x ∨ B x) ≤ prob A + prob B := by
  sorry

/-- Monotonicity: an event inside another has no larger probability. -/
theorem prob_mono {Ω : Type} [Fintype Ω] (A B : Ω → Prop) (h : ∀ x, A x → B x) : prob A ≤ prob B := by
  sorry

end Lemmas

/-- Lemma 1. -/
theorem TimeBound : Accounting.Guarantees.TS.TimeBound := by
  sorry

/-- Theorem D(a): `price_le_bound`, `price_of_rd`, `identity` and `TimeBound`. -/
theorem DecompositionD : Accounting.Guarantees.TS.DecompositionD := by
  sorry

/-- Theorem D-ideal (2.7): `luck_rounds`, `deficiency_le_rounds`, `price_of_parts`, `identity`, `TimeBound`. -/
theorem DIdeal : Accounting.Guarantees.TS.DIdeal := by
  sorry

/-- Theorem D-ideal (2.7♭): as `DIdeal`, with the identity at ♭. -/
theorem DIdealFloor : Accounting.Guarantees.TS.DIdealFloor := by
  sorry

/-- Lemma C: `budget_tail`, `price_of_parts`, `identity`, `TimeBound`. -/
theorem CompilerC : Accounting.Guarantees.TS.CompilerC := by
  sorry

/-- Corollary D3: `price_of_rd`, `identity` at ♭, `TimeBound`; the signs from `FloorKnown`. -/
theorem FloorCreditD3 : Accounting.Guarantees.TS.FloorCreditD3 := by
  sorry

/-- Corollary D1, one binding facet. -/
theorem BindingFacetD1 : Accounting.Guarantees.TS.BindingFacetD1 := by
  sorry

/-- Corollary D1, the vector route's threshold. -/
theorem VectorThresholdD1 : Accounting.Guarantees.TS.VectorThresholdD1 := by
  sorry

/-- Corollary D2. -/
theorem RoutesOffD2 : Accounting.Guarantees.TS.RoutesOffD2 := by
  sorry

/-- T1. -/
theorem TensorSlotCount : Accounting.Guarantees.TS.TensorSlotCount := by
  sorry

/-- T1's tightness. -/
theorem TensorSlotTight : Accounting.Guarantees.TS.TensorSlotTight := by
  sorry

/-- Lemma 2's count. -/
theorem OutputCount : Accounting.Guarantees.TS.OutputCount := by
  sorry

end Accounting.SecurityProofs.TS
