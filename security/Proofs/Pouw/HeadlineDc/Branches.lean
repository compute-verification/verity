module

public import Definitions.Pouw.Assumptions.TS
public import Definitions.Pouw.HeadlineDc.Encoding
public import Mathlib.CategoryTheory.Category.Basic
public import Proofs.Pouw.Headline.Encoding
public import Proofs.Pouw.Headline.EpsJl
public import Proofs.Pouw.Headline.Luck
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Algebra.Order.Ring.Star
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Data.Rat.Star
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.ENatToNat
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init
import Proofs.Pouw.Headline.Whole
import Proofs.Pouw.TS

/-! The data-centre headlines' shared steps. -/

@[expose] public section

namespace ResourceAccounting.SecurityProofs.TS.HeadlineDc

open ResourceAccounting.TS ResourceAccounting.TS.Headline ResourceAccounting.Assumptions.TS ResourceAccounting.SecurityProofs.TS
  ResourceAccounting.SecurityProofs.TS.Headline

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

theorem wStar_ge {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F]
    [Fintype X] [Nonempty F] [Nonempty X] (Z : Model K R T L U F X) (Fl : Labelled K) (lam : R → ℝ)
    (h : IsCert lam) (hZ : Z.Wf Fl) (r0 : R) (w0 : ℝ) (hw : ∀ hd, w0 ≤ Z.demand hd r0) :
    lam r0 * w0 ≤ Z.wStar lam := by
  have : Nonempty (Head T L U) := ⟨.vector⟩
  obtain ⟨hd0, -⟩ := stars_nonneg Z Fl lam hZ h
  unfold Model.wStar
  exact le_ciInf fun hd =>
    (mul_le_mul_of_nonneg_left (hw hd) (h.1 r0)).trans (price_ge_facet lam _ h.1 (hd0 hd) r0)

theorem phiStar_ge_facet {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U]
    [Fintype F] [Fintype X] [Nonempty F] (Z : Model K R T L U F X) (Fl : Labelled K) (lam : R → ℝ)
    (h : IsCert lam) (hZ : Z.Wf Fl) (r0 : R) (φ0 : ℝ) (hφ : ∀ φ, φ0 ≤ lam r0 * Z.fold φ r0) :
    φ0 ≤ Z.phiStar lam := by
  obtain ⟨-, -, -, -, -, -, -, -, -, hfold, -⟩ := hZ
  unfold Model.phiStar
  exact le_ciInf fun φ => (hφ φ).trans (price_ge_facet lam _ h.1 (hfold φ) r0)

theorem phiStar_ge_pair {K R T L U F X : Type} [Fintype K] [Fintype R] [DecidableEq R] [Fintype T] [Fintype L]
    [Fintype U] [Fintype F] [Fintype X] [Nonempty F] (Z : Model K R T L U F X) (Fl : Labelled K) (lam : R → ℝ)
    (h : IsCert lam) (hZ : Z.Wf Fl) (a b : R) (hab : a ≠ b) (φ0 : ℝ)
    (hφ : ∀ φ, φ0 ≤ lam a * Z.fold φ a + lam b * Z.fold φ b) : φ0 ≤ Z.phiStar lam := by
  obtain ⟨-, -, -, -, -, -, -, -, -, hfold, -⟩ := hZ
  unfold Model.phiStar
  exact le_ciInf fun φ => (hφ φ).trans (price_ge_pair lam _ h.1 (hfold φ) a b hab)

theorem wStar_ge_triple {K R T L U F X : Type} [Fintype K] [Fintype R] [DecidableEq R] [Fintype T] [Fintype L]
    [Fintype U] [Fintype F] [Fintype X] [Nonempty F] [Nonempty X] (Z : Model K R T L U F X) (Fl : Labelled K)
    (lam : R → ℝ) (h : IsCert lam) (hZ : Z.Wf Fl) (a b c : R) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) (w0 : ℝ)
    (hw : ∀ hd, w0 ≤ lam a * Z.demand hd a + lam b * Z.demand hd b + lam c * Z.demand hd c) :
    w0 ≤ Z.wStar lam := by
  have : Nonempty (Head T L U) := ⟨.vector⟩
  obtain ⟨hd0, -⟩ := stars_nonneg Z Fl lam hZ h
  unfold Model.wStar
  exact le_ciInf fun hd => (hw hd).trans (price_ge_triple lam _ h.1 (hd0 hd) a b c hab hac hbc)

theorem phiStar_ge_triple {K R T L U F X : Type} [Fintype K] [Fintype R] [DecidableEq R] [Fintype T] [Fintype L]
    [Fintype U] [Fintype F] [Fintype X] [Nonempty F] (Z : Model K R T L U F X) (Fl : Labelled K) (lam : R → ℝ)
    (h : IsCert lam) (hZ : Z.Wf Fl) (a b c : R) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) (φ0 : ℝ)
    (hφ : ∀ φ, φ0 ≤ lam a * Z.fold φ a + lam b * Z.fold φ b + lam c * Z.fold φ c) : φ0 ≤ Z.phiStar lam := by
  obtain ⟨-, -, -, -, -, -, -, -, -, hfold, -⟩ := hZ
  unfold Model.phiStar
  exact le_ciInf fun φ => (hφ φ).trans (price_ge_triple lam _ h.1 (hfold φ) a b c hab hac hbc)

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

theorem folded_of_lower {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U]
    [Fintype F] [Fintype X] [Nonempty F] [Nonempty X] (Z : Model K R T L U F X) (Fl : Labelled K) (lam : R → ℝ)
    (h : IsCert lam) (hZ : Z.Wf Fl) (hμF : Z.muF = 0) (hμX : Z.muX = 0) (S : ℕ) (hS : (S : ℝ) = 48000)
    (δ' : ℝ) (hδ' : (1 / 2 : ℝ) ^ 41 ≤ δ') (w φ ε : ℝ) (hw : w ≤ Z.wStar lam) (hφ : φ ≤ Z.phiStar lam)
    (hε : 1 - ε ≤ (Z.Nw * w * (1 - 41 * (6931471808 / 10000000000) / 6000) + Z.Nw * φ) / Z.bnd Fl) :
    1 - ε ≤ 1 - Z.epsSkip Fl lam - Z.epsCert Fl lam - Z.epsRoute Fl lam - Z.epsMand Fl lam -
      Z.epsJl Fl lam (7 / 8) δ' S :=
  folded_of_lower_at Z Fl lam h hZ hμF hμX S hS δ' hδ' w φ ε hw hφ _ (by norm_num) (by norm_num) hε

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
    unfold Machine.price Machine.load
    exact Finset.sum_nonneg fun r _ => mul_nonneg (hlam.1 r)
      (Finset.sum_nonneg fun k _ => mul_nonneg (ha r k) (hA.2.2.1 k))
  rw [hμX, sub_zero, one_mul]
  exact hchi.trans hprice

end ResourceAccounting.SecurityProofs.TS.HeadlineDc
