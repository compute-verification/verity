import Proofs.Accounting.Headline.Encoding
import Proofs.Accounting.Headline.EpsJl
import Proofs.Accounting.Headline.Luck
import Definitions.Accounting.HeadlineDc.Encoding

/-!
# The data-centre headlines' shared steps

`Whole.headline_of_branches` takes the folded branch's and the unfolded branch's floors at the proof's certificate.
On a model with `μ_f = μ_x = 0` both floors only grow with `w*` and `φ*`, so a lower bound on each is enough:

* `sheetCoefQ`, `sheet_aQ`, `sheetCoefQ_nonneg`: `sheetCoef` and its two facts for a `SheetDataQ` sheet (rational pipe
  widths, the DS sheets' uniform datapath);
* `price_ge_facet`, `price_ge_pair`, `price_ge_triple`: a certificate prices a demand `≥ 0` at least at one facet's
  term, or two's, or three's;
* `wStar_ge`, `wStar_ge_triple`: `λ_{r₀}·w₀ ≤ w*` when every word route's demand on `r₀` is at least `w₀`, and
  `w₀ ≤ w*` when every word route's price on three facets is (DC-90-ri's certificate);
* `phiStar_ge_facet`, `phiStar_ge_pair`, `phiStar_ge_triple`: `φ₀ ≤ φ*` when every fold route's price on one facet,
  two (DS-100's certificate) or three is at least `φ₀`;
* `folded_of_lower_at`, `folded_of_lower`: the folded branch's floor at `p = 7/8` is at least
  `(N_w w (1 − t) + N_w φ)/B_♭` for any `t` from `41·0.6931471808/6400` (`ln(8/7) ≥ 2/15`) to 1, from `identity` at
  `d̄ = ā = 0`, `χ* ≥ 0` and `Headline.epsJl_le`; `folded_of_lower` is `t = 41·0.6931471808/6000` (`ln(8/7) ≥ 1/8`);
* `unfolded_of_lower`: the unfolded branch's floor is at least `(N_w w + N_w φ)/B_♭ · (1 − 41/48000)`.
-/

namespace Accounting.SecurityProofs.TS.HeadlineDc

open Accounting.TS Accounting.TS.Headline Accounting.Assumptions.TS Accounting.SecurityProofs.TS
  Accounting.SecurityProofs.TS.Headline

/-- A `SheetDataQ` sheet's rows as rationals, by class index: `Sheet.machine`'s, at `SheetDataQ.sheet`'s lists. -/
def sheetCoefQ {P : Type} [DecidableEq P] (D : SheetDataQ P) : SheetRow P → ℕ → ℚ
  | .issue, _ => 1
  | .tensor, i => (D.macs.getD i 0 : ℚ) / ((D.peak.getD i 0 : ℚ) / 4)
  | .pipe p, i => if D.pipe.getD i none = some p then 128 / D.width p else 0
  | .smem, i => (D.smem.getD i 0 : ℚ) / 32

theorem sheet_aQ {P : Type} [DecidableEq P] (D : SheetDataQ P) (n : ℕ) (r : SheetRow P) (k : Fin n) :
    (D.sheet n).machine.a r k = (sheetCoefQ D r k.val : ℝ) := by
  cases r with
  | pipe p =>
    show (if D.pipe.getD k.val none = some p then (128 : ℝ) / (D.width p : ℝ) else 0) =
      ((if D.pipe.getD k.val none = some p then (128 : ℚ) / D.width p else 0 : ℚ) : ℝ)
    by_cases h : D.pipe.getD k.val none = some p
    · rw [ite_eq_left h, ite_eq_left h, Rat.cast_div, Rat.cast_ofNat]
    · rw [ite_eq_right h, ite_eq_right h, Rat.cast_zero]
  | issue => simp [Sheet.machine, SheetDataQ.sheet, sheetCoefQ]
  | tensor => simp [Sheet.machine, SheetDataQ.sheet, sheetCoefQ]
  | smem => simp [Sheet.machine, SheetDataQ.sheet, sheetCoefQ]

theorem sheetCoefQ_nonneg {P : Type} [DecidableEq P] (D : SheetDataQ P) (hw : ∀ p, 0 ≤ D.width p)
    (r : SheetRow P) (i : ℕ) : 0 ≤ sheetCoefQ D r i := by
  cases r <;> simp only [sheetCoefQ] <;> (try split_ifs) <;>
    first | positivity | exact div_nonneg (by norm_num) (hw _)

theorem price_ge_facet {R : Type} [Fintype R] (lam d : R → ℝ) (hlam : ∀ r, 0 ≤ lam r) (hd : ∀ r, 0 ≤ d r)
    (r0 : R) : lam r0 * d r0 ≤ demandPrice lam d :=
  Finset.single_le_sum (f := fun r => lam r * d r) (fun r _ => mul_nonneg (hlam r) (hd r)) (Finset.mem_univ r0)

theorem price_ge_pair {R : Type} [Fintype R] [DecidableEq R] (lam d : R → ℝ) (hlam : ∀ r, 0 ≤ lam r)
    (hd : ∀ r, 0 ≤ d r) (a b : R) (hab : a ≠ b) : lam a * d a + lam b * d b ≤ demandPrice lam d := by
  unfold demandPrice
  have e : ∑ r ∈ ({a, b} : Finset R), lam r * d r = lam a * d a + lam b * d b := Finset.sum_pair hab
  rw [← e]
  exact Finset.sum_le_univ_sum_of_nonneg fun r => mul_nonneg (hlam r) (hd r)

theorem price_ge_triple {R : Type} [Fintype R] [DecidableEq R] (lam d : R → ℝ) (hlam : ∀ r, 0 ≤ lam r)
    (hd : ∀ r, 0 ≤ d r) (a b c : R) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
    lam a * d a + lam b * d b + lam c * d c ≤ demandPrice lam d := by
  unfold demandPrice
  have hm : a ∉ ({b, c} : Finset R) := by simp [hab, hac]
  have e2 : ∑ r ∈ ({b, c} : Finset R), lam r * d r = lam b * d b + lam c * d c := Finset.sum_pair hbc
  have e : ∑ r ∈ ({a, b, c} : Finset R), lam r * d r = lam a * d a + lam b * d b + lam c * d c := by
    rw [Finset.sum_insert hm, e2]
    ring
  rw [← e]
  exact Finset.sum_le_univ_sum_of_nonneg fun r => mul_nonneg (hlam r) (hd r)

/-- `w* ≥ λ_{r₀}·w₀` when every word route's demand on facet `r₀` is at least `w₀`. -/
theorem wStar_ge {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F]
    [Fintype X] [Nonempty F] [Nonempty X] (Z : Model K R T L U F X) (Fl : Labelled K) (lam : R → ℝ)
    (h : IsCert lam) (hZ : Z.Wf Fl) (r0 : R) (w0 : ℝ) (hw : ∀ hd, w0 ≤ Z.demand hd r0) :
    lam r0 * w0 ≤ Z.wStar lam := by
  have : Nonempty (Head T L U) := ⟨.vector⟩
  obtain ⟨hd0, -⟩ := stars_nonneg Z Fl lam hZ h
  unfold Model.wStar
  exact le_ciInf fun hd =>
    (mul_le_mul_of_nonneg_left (hw hd) (h.1 r0)).trans (price_ge_facet lam _ h.1 (hd0 hd) r0)

/-- `φ* ≥ φ₀` when every fold route's price on facet `r₀` is at least `φ₀`. -/
theorem phiStar_ge_facet {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U]
    [Fintype F] [Fintype X] [Nonempty F] (Z : Model K R T L U F X) (Fl : Labelled K) (lam : R → ℝ)
    (h : IsCert lam) (hZ : Z.Wf Fl) (r0 : R) (φ0 : ℝ) (hφ : ∀ φ, φ0 ≤ lam r0 * Z.fold φ r0) :
    φ0 ≤ Z.phiStar lam := by
  obtain ⟨-, -, -, -, -, -, -, -, -, hfold, -⟩ := hZ
  unfold Model.phiStar
  exact le_ciInf fun φ => (hφ φ).trans (price_ge_facet lam _ h.1 (hfold φ) r0)

/-- `φ* ≥ φ₀` when every fold route's price on facets `a` and `b` is at least `φ₀`. -/
theorem phiStar_ge_pair {K R T L U F X : Type} [Fintype K] [Fintype R] [DecidableEq R] [Fintype T] [Fintype L]
    [Fintype U] [Fintype F] [Fintype X] [Nonempty F] (Z : Model K R T L U F X) (Fl : Labelled K) (lam : R → ℝ)
    (h : IsCert lam) (hZ : Z.Wf Fl) (a b : R) (hab : a ≠ b) (φ0 : ℝ)
    (hφ : ∀ φ, φ0 ≤ lam a * Z.fold φ a + lam b * Z.fold φ b) : φ0 ≤ Z.phiStar lam := by
  obtain ⟨-, -, -, -, -, -, -, -, -, hfold, -⟩ := hZ
  unfold Model.phiStar
  exact le_ciInf fun φ => (hφ φ).trans (price_ge_pair lam _ h.1 (hfold φ) a b hab)

/-- `w* ≥ w₀` when every word route's price on facets `a`, `b` and `c` is at least `w₀` (DC-90-ri's certificate). -/
theorem wStar_ge_triple {K R T L U F X : Type} [Fintype K] [Fintype R] [DecidableEq R] [Fintype T] [Fintype L]
    [Fintype U] [Fintype F] [Fintype X] [Nonempty F] [Nonempty X] (Z : Model K R T L U F X) (Fl : Labelled K)
    (lam : R → ℝ) (h : IsCert lam) (hZ : Z.Wf Fl) (a b c : R) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) (w0 : ℝ)
    (hw : ∀ hd, w0 ≤ lam a * Z.demand hd a + lam b * Z.demand hd b + lam c * Z.demand hd c) :
    w0 ≤ Z.wStar lam := by
  have : Nonempty (Head T L U) := ⟨.vector⟩
  obtain ⟨hd0, -⟩ := stars_nonneg Z Fl lam hZ h
  unfold Model.wStar
  exact le_ciInf fun hd => (hw hd).trans (price_ge_triple lam _ h.1 (hd0 hd) a b c hab hac hbc)

/-- `φ* ≥ φ₀` when every fold route's price on facets `a`, `b` and `c` is at least `φ₀`. -/
theorem phiStar_ge_triple {K R T L U F X : Type} [Fintype K] [Fintype R] [DecidableEq R] [Fintype T] [Fintype L]
    [Fintype U] [Fintype F] [Fintype X] [Nonempty F] (Z : Model K R T L U F X) (Fl : Labelled K) (lam : R → ℝ)
    (h : IsCert lam) (hZ : Z.Wf Fl) (a b c : R) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) (φ0 : ℝ)
    (hφ : ∀ φ, φ0 ≤ lam a * Z.fold φ a + lam b * Z.fold φ b + lam c * Z.fold φ c) : φ0 ≤ Z.phiStar lam := by
  obtain ⟨-, -, -, -, -, -, -, -, -, hfold, -⟩ := hZ
  unfold Model.phiStar
  exact le_ciInf fun φ => (hφ φ).trans (price_ge_triple lam _ h.1 (hfold φ) a b c hab hac hbc)

/-- The folded branch's floor at `p = 7/8`, `S = 48,000` and `δ′ ≥ 2^-41` from lower bounds `w ≤ w*`, `φ ≤ φ*` on
a model with `μ_f = μ_x = 0`, at any `t ≤ 1` at least `41·0.6931471808/6400`: `ε_jl ≤ κ·t` (`Headline.epsJl_le`),
so the floor is at least `(N_w w (1 − t) + N_w φ)/B_♭`. -/
theorem folded_of_lower_at {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U]
    [Fintype F] [Fintype X] [Nonempty F] [Nonempty X] (Z : Model K R T L U F X) (Fl : Labelled K) (lam : R → ℝ)
    (h : IsCert lam) (hZ : Z.Wf Fl) (hμF : Z.muF = 0) (hμX : Z.muX = 0) (S : ℕ) (hS : (S : ℝ) = 48000)
    (δ' : ℝ) (hδ' : (1 / 2 : ℝ) ^ 41 ≤ δ') (w φ ε : ℝ) (hw : w ≤ Z.wStar lam) (hφ : φ ≤ Z.phiStar lam) (t : ℝ)
    (ht : 41 * (6931471808 / 10000000000) / 6400 ≤ t) (ht1 : t ≤ 1)
    (hε : 1 - ε ≤ (Z.Nw * w * (1 - t) + Z.Nw * φ) / Z.bnd Fl) :
    1 - ε ≤ 1 - Z.epsSkip Fl lam - Z.epsCert Fl lam - Z.epsRoute Fl lam - Z.epsMand Fl lam -
      Z.epsJl Fl lam (7 / 8) δ' S := by
  have hid := identity Z Fl lam 0 0 hZ
  have hZ' := hZ
  obtain ⟨-, -, hNw, hB0, -⟩ := hZ'
  obtain ⟨-, -, -, hχ0⟩ := stars_nonneg Z Fl lam hZ h
  obtain ⟨hκ0, -⟩ := Whole.kappa_f_nonneg Z Fl lam h hZ
  have hjl := (Headline.epsJl_le Z Fl lam hκ0 S hS δ' hδ').trans (mul_le_mul_of_nonneg_left ht hκ0)
  have hLuck : Z.epsLuck Fl lam 0 = 0 := zero_mul _
  have hAdapt : Z.epsAdapt Fl lam 0 = 0 := zero_mul _
  have hκ : Z.kappa Fl lam * t = Z.Nw * Z.wStar lam * t / Z.bnd Fl := by
    unfold Model.kappa
    ring
  have h1 : Z.Nw * w ≤ Z.Nw * Z.wStar lam := mul_le_mul_of_nonneg_left hw hNw.le
  have h2 : Z.Nw * φ ≤ Z.Nw * Z.phiStar lam := mul_le_mul_of_nonneg_left hφ hNw.le
  have h1t : Z.Nw * w * (1 - t) ≤ Z.Nw * Z.wStar lam * (1 - t) := mul_le_mul_of_nonneg_right h1 (by linarith)
  have hnum : Z.Nw * w * (1 - t) + Z.Nw * φ ≤
      ((1 - 0 - 0) * Z.Nw * Z.wStar lam + (1 - Z.muF) * Z.Nw * Z.phiStar lam + (1 - Z.muX) * Z.chiStar lam) -
        Z.Nw * Z.wStar lam * t := by
    rw [hμF, hμX]
    linarith
  have hkey : (Z.Nw * w * (1 - t) + Z.Nw * φ) / Z.bnd Fl ≤
      ((1 - 0 - 0) * Z.Nw * Z.wStar lam + (1 - Z.muF) * Z.Nw * Z.phiStar lam + (1 - Z.muX) * Z.chiStar lam) /
          Z.bnd Fl -
        Z.Nw * Z.wStar lam * t / Z.bnd Fl := by
    rw [← sub_div]
    exact div_le_div_of_nonneg_right hnum hB0.le
  linarith

/-- `folded_of_lower_at` at `t = 41·0.6931471808/6000`, `ln(8/7) ≥ 1/8`'s: the floor is at least
`(N_w w (1 − 41·0.6931471808/6000) + N_w φ)/B_♭`. -/
theorem folded_of_lower {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U]
    [Fintype F] [Fintype X] [Nonempty F] [Nonempty X] (Z : Model K R T L U F X) (Fl : Labelled K) (lam : R → ℝ)
    (h : IsCert lam) (hZ : Z.Wf Fl) (hμF : Z.muF = 0) (hμX : Z.muX = 0) (S : ℕ) (hS : (S : ℝ) = 48000)
    (δ' : ℝ) (hδ' : (1 / 2 : ℝ) ^ 41 ≤ δ') (w φ ε : ℝ) (hw : w ≤ Z.wStar lam) (hφ : φ ≤ Z.phiStar lam)
    (hε : 1 - ε ≤ (Z.Nw * w * (1 - 41 * (6931471808 / 10000000000) / 6000) + Z.Nw * φ) / Z.bnd Fl) :
    1 - ε ≤ 1 - Z.epsSkip Fl lam - Z.epsCert Fl lam - Z.epsRoute Fl lam - Z.epsMand Fl lam -
      Z.epsJl Fl lam (7 / 8) δ' S :=
  folded_of_lower_at Z Fl lam h hZ hμF hμX S hS δ' hδ' w φ ε hw hφ _ (by norm_num) (by norm_num) hε

/-- The unfolded branch's floor from lower bounds `w ≤ w*`, `φ ≤ φ*` on a model with `μ_f = 0`. -/
theorem unfolded_of_lower {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U]
    [Fintype F] [Fintype X] (Z : Model K R T L U F X) (Fl : Labelled K) (lam : R → ℝ) (hZ : Z.Wf Fl)
    (hμF : Z.muF = 0) (w φ ε : ℝ) (hw : w ≤ Z.wStar lam) (hφ : φ ≤ Z.phiStar lam)
    (hε : 1 - ε ≤ (Z.Nw * w + Z.Nw * φ) / Z.bnd Fl * (1 - 41 / 48000)) :
    1 - ε ≤ (Z.Nw * Z.wStar lam + (1 - Z.muF) * Z.Nw * Z.phiStar lam) / Z.bnd Fl * (1 - 41 / 48000) := by
  obtain ⟨-, -, hNw, hB0, -⟩ := hZ
  have hnum : Z.Nw * w + Z.Nw * φ ≤ Z.Nw * Z.wStar lam + (1 - Z.muF) * Z.Nw * Z.phiStar lam := by
    rw [hμF, sub_zero, one_mul]
    exact add_le_add (mul_le_mul_of_nonneg_left hw hNw.le) (mul_le_mul_of_nonneg_left hφ hNw.le)
  exact hε.trans (mul_le_mul_of_nonneg_right (div_le_div_of_nonneg_right hnum hB0.le) (by norm_num))

/-- FILL-OPT on any run of a model with no filler: `χ* ≤ 0` and the run's filler is priced `≥ 0`. -/
theorem fillOpt_of_zero {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U]
    [Fintype F] [Fintype X] [Nonempty X] (Z : Model K R T L U F X) (ha : ∀ r k, 0 ≤ Z.G.a r k)
    (hfill : ∀ ψ r, Z.fill ψ r = 0) (hμX : Z.muX = 0) (A : Anatomy K T L U) (hA : A.Split) : FillOpt Z A := by
  intro lam hlam
  obtain ⟨ψ⟩ := (inferInstance : Nonempty X)
  have hchi : Z.chiStar lam ≤ 0 := by
    refine (iInf_le_price lam Z.fill ψ).trans (le_of_eq ?_)
    unfold demandPrice
    simp [hfill]
  have hprice : 0 ≤ Z.G.price lam A.nfill := by
    unfold FacetCosts.price FacetCosts.load
    exact Finset.sum_nonneg fun r _ => mul_nonneg (hlam.1 r)
      (Finset.sum_nonneg fun k _ => mul_nonneg (ha r k) (hA.2.2.1 k))
  rw [hμX, sub_zero, one_mul]
  exact hchi.trans hprice

end Accounting.SecurityProofs.TS.HeadlineDc
