import Proofs.Accounting.Headline.Encoding
import Proofs.Accounting.TS
import Models.Accounting.Guarantees.Headline

/-!
# sm_80's headline at the data sheet

`Sm80SheetHeadline` is `SheetDIdealFloor` at sm_80's data (`Definitions.Accounting.Headline.Sm80`), with the data's
hypotheses discharged and the terms bounded:

* the data, by `decide`: the ALU pipe is the floor program's largest facet (`sm80_max`), so `B_♭` is its ALU count
  `B_♭ ≈ 16.023` (`sm80_bnd`); no tensor class issues on the ALU pipe, so the first writer kind's word route is free
  there and `w* = 0` (`sm80_wStar`), which makes `κ`, and with it `ε_jl`, `0` (`sm80_jl`); the fold route costs `1/8`
  on ALU (`sm80_phiStar`);
* the certificate is `π*`, the ALU facet (`isCert_onFacet`), and FILL-OPT holds since there is no filler
  (`sm80_fillOpt`);
* `identity` at `d̄ = ā = 0` gives `ε_skip + ε_cert + ε_route + ε_mand ≤ 1 − N_w φ*/B_♭ ≤ 1 − 16/B_♭`, below
  `147/100000` by `decide` on the rationals (`sm80_final`).
-/

namespace Accounting.SecurityProofs.TS.Headline

open Accounting.TS Accounting.TS.Headline Accounting.Assumptions.TS Pouw.RelativeSecurity Accounting.SecurityProofs.TS

/-- sm_80's floor program's counts, its four lists in `labelled`'s order. -/
def flatAll80 : Sparse := Sm80.flatWords ++ Sm80.flatFold ++ [] ++ Sm80.flatSkip

/-- sm_80's `B_♭` as a rational: the floor program's ALU count. -/
def Bq80 : ℚ := dot (sheetCoef Sm80.ds80 Sm80.piStar) 35 flatAll80

theorem sm80_max (r : SheetRow Sm80.Pipe) :
    dot (sheetCoef Sm80.ds80 r) 35 flatAll80 ≤ dot (sheetCoef Sm80.ds80 Sm80.piStar) 35 flatAll80 := by
  cases r with
  | issue => decide +kernel
  | tensor => decide +kernel
  | pipe p => cases p <;> decide +kernel
  | smem => decide +kernel

theorem sm80_Bpos : (0 : ℚ) < Bq80 := by decide +kernel

theorem sm80_omega : ∀ i < 23, 0 < (Sm80.d2s.writers.getD i (0, 0)).2 := by
  decide +kernel

/-- The first writer kind's class (BMMA168128) is not on the ALU pipe. -/
theorem sm80_writer0 : sheetCoef Sm80.ds80 Sm80.piStar ((Sm80.d2s.writers.getD 0 (0, 0)).1 % 35) = 0 := by
  decide +kernel

theorem sm80_fold : ∀ i < 1, (1 / 8 : ℚ) ≤ dot (sheetCoef Sm80.ds80 Sm80.piStar) 35 (Sm80.d2s.folds.getD i []) := by
  decide +kernel

theorem sm80_final_q : (1 - 16 / Bq80 : ℚ) ≤ 147 / 100000 := by
  decide +kernel

theorem sm80_hG : ∀ r k, Sm80.model.G.a r k = (sheetCoef Sm80.ds80 r k.val : ℝ) := sheet_a Sm80.ds80 35

theorem sm80_a_nonneg : ∀ r k, 0 ≤ Sm80.model.G.a r k := fun r k => by
  rw [sm80_hG]
  exact Rat.cast_nonneg.mpr (sheetCoef_nonneg _ _ _)

theorem sm80_bnd : Sm80.model.bnd Sm80.flat = (Bq80 : ℝ) := by
  unfold Model.bnd
  rw [show Sm80.flat.n = Sparse.counts flatAll80 from labelled_n 35 Sm80.flatWords Sm80.flatFold [] Sm80.flatSkip]
  exact bound_eq _ (sheetCoef Sm80.ds80) sm80_hG flatAll80 Sm80.piStar sm80_max

theorem sm80_Nw : Sm80.model.Nw = 128 := by
  show ((128 : ℕ) : ℝ) = 128
  norm_num

theorem sm80_q : Sm80.model.q = 1 / 2 := by
  show ((1 / 2 : ℚ) : ℝ) = 1 / 2
  norm_num

theorem sm80_qT : Sm80.model.qT = 1 / 2 := by
  show ((1 / 2 : ℚ) : ℝ) = 1 / 2
  norm_num

theorem sm80_wf : Sm80.model.Wf Sm80.flat := by
  refine ⟨sm80_a_nonneg, ?_, ?_, ?_, fun t => ?_, ?_, ?_, le_rfl, fun u => u.elim, fun φ r => ?_, fun _ _ => le_rfl,
    le_rfl, ?_, le_rfl, ?_⟩
  · show (0 : ℝ) < ((1410 : ℕ) : ℝ)
    norm_num
  · rw [sm80_Nw]
    norm_num
  · rw [sm80_bnd]
    exact_mod_cast sm80_Bpos
  · show (0 : ℝ) < ((Sm80.d2s.writers.getD t.val (0, 0)).2 : ℝ)
    exact_mod_cast sm80_omega t.val t.isLt
  · rw [sm80_q]
    norm_num
  · rw [sm80_qT]
    norm_num
  · show 0 ≤ Sm80.model.G.load (Sparse.counts (Sm80.d2s.folds.getD φ.val [])) r
    exact load_counts_nonneg _ sm80_a_nonneg _ _
  · show (0 : ℝ) ≤ 1
    norm_num
  · show (0 : ℝ) ≤ 1
    norm_num

/-- `w* = 0` at `π*`: writer kind 0's word route has no ALU instruction, and no route is priced below `0`. -/
theorem sm80_wStar : Sm80.model.wStar (onFacet Sm80.piStar) = 0 := by
  refine le_antisymm
    ((iInf_le_price (onFacet Sm80.piStar) Sm80.model.demand (.writer 0)).trans (le_of_eq ?_))
    (stars_nonneg Sm80.model Sm80.flat (onFacet Sm80.piStar) sm80_wf (isCert_onFacet _)).2.1
  rw [demandPrice_onFacet]
  show Sm80.model.G.a Sm80.piStar (idx 35 (Sm80.d2s.writers.getD 0 (0, 0)).1) /
    ((Sm80.d2s.writers.getD 0 (0, 0)).2 : ℝ) = 0
  rw [sm80_hG]
  show ((sheetCoef Sm80.ds80 Sm80.piStar ((Sm80.d2s.writers.getD 0 (0, 0)).1 % 35) : ℚ) : ℝ) /
    ((Sm80.d2s.writers.getD 0 (0, 0)).2 : ℝ) = 0
  rw [sm80_writer0, Rat.cast_zero, zero_div]

/-- `φ* ≥ 1/8` at `π*`: the fold route's ALU count. -/
theorem sm80_phiStar : 1 / 8 ≤ Sm80.model.phiStar (onFacet Sm80.piStar) := by
  unfold Model.phiStar
  refine le_ciInf fun φ => ?_
  rw [demandPrice_onFacet]
  show (1 : ℝ) / 8 ≤ Sm80.model.G.load (Sparse.counts (Sm80.d2s.folds.getD φ.val [])) Sm80.piStar
  rw [load_counts _ (sheetCoef Sm80.ds80) sm80_hG]
  have h : ((1 / 8 : ℚ) : ℝ) ≤
      ((dot (sheetCoef Sm80.ds80 Sm80.piStar) 35 (Sm80.d2s.folds.getD φ.val []) : ℚ) : ℝ) :=
    Rat.cast_le.mpr (sm80_fold φ.val φ.isLt)
  push_cast at h
  linarith

/-- FILL-OPT on any run: the model has no filler, so `χ* = 0` and the run's filler is priced `≥ 0`. -/
theorem sm80_fillOpt {Ω : Type} [Fintype Ω] (Pl : Play Ω Sm80.checks (Fin 35) (Fin 23) Unit Empty)
    (hSplit : ∀ x, (Pl.run x).Split) : ∀ x, FillOpt Sm80.model (Pl.run x) := by
  intro x lam hlam
  have hchi : Sm80.model.chiStar lam ≤ 0 := by
    refine (iInf_le_price lam Sm80.model.fill ()).trans (le_of_eq ?_)
    show ∑ r, lam r * (0 : ℝ) = 0
    simp
  have hprice : 0 ≤ Sm80.model.G.price lam (Pl.run x).nfill := by
    unfold Machine.price Machine.load
    exact Finset.sum_nonneg fun r _ => mul_nonneg (hlam.1 r)
      (Finset.sum_nonneg fun k _ => mul_nonneg (sm80_a_nonneg r k) ((hSplit x).2.2.1 k))
  show (1 - (0 : ℝ)) * Sm80.model.chiStar lam ≤ _
  rw [sub_zero, one_mul]
  exact hchi.trans hprice

/-- `ε_jl = 0` at `π*`, at any `δ′`: `κ = N_w w*/B_♭ = 0`. -/
theorem sm80_jl (δ' : ℝ) : Sm80.model.epsJl Sm80.flat (onFacet Sm80.piStar) (3 / 4) δ' Sm80.checks = 0 := by
  unfold Model.epsJl Model.kappa
  rw [sm80_wStar, mul_zero, zero_div, zero_mul, zero_div]

theorem sm80_final : 1 - 16 / Sm80.model.bnd Sm80.flat ≤ 147 / 100000 := by
  rw [sm80_bnd]
  have h : ((1 - 16 / Bq80 : ℚ) : ℝ) ≤ ((147 / 100000 : ℚ) : ℝ) := Rat.cast_le.mpr sm80_final_q
  push_cast at h
  linarith

/-- The terms at `π*` sum to at most `147/100000`. -/
theorem sm80_eps_le (δ' : ℝ) :
    1 - 147 / 100000 ≤ 1 - Sm80.model.epsSkip Sm80.flat (onFacet Sm80.piStar) -
      Sm80.model.epsCert Sm80.flat (onFacet Sm80.piStar) -
      Sm80.model.epsRoute Sm80.flat (onFacet Sm80.piStar) -
      Sm80.model.epsMand Sm80.flat (onFacet Sm80.piStar) -
      Sm80.model.epsJl Sm80.flat (onFacet Sm80.piStar) (3 / 4) δ' Sm80.checks := by
  have hid := identity Sm80.model Sm80.flat (onFacet Sm80.piStar) 0 0 sm80_wf
  have hBpos : 0 < Sm80.model.bnd Sm80.flat := sm80_wf.2.2.2.1
  have hχ := (stars_nonneg Sm80.model Sm80.flat (onFacet Sm80.piStar) sm80_wf (isCert_onFacet _)).2.2.2
  have hφ := sm80_phiStar
  have hN : 16 ≤ (1 - 0 - 0) * Sm80.model.Nw * Sm80.model.wStar (onFacet Sm80.piStar) +
      (1 - Sm80.model.muF) * Sm80.model.Nw * Sm80.model.phiStar (onFacet Sm80.piStar) +
      (1 - Sm80.model.muX) * Sm80.model.chiStar (onFacet Sm80.piStar) := by
    rw [sm80_Nw, sm80_wStar, show Sm80.model.muF = 0 from rfl, show Sm80.model.muX = 0 from rfl]
    linarith
  have hA := div_le_div_of_nonneg_right hN hBpos.le
  have hLuck : Sm80.model.epsLuck Sm80.flat (onFacet Sm80.piStar) 0 = 0 := zero_mul _
  have hAdapt : Sm80.model.epsAdapt Sm80.flat (onFacet Sm80.piStar) 0 = 0 := zero_mul _
  have hjl := sm80_jl δ'
  have hfin := sm80_final
  linarith

end Accounting.SecurityProofs.TS.Headline

namespace Accounting.SecurityProofs.TS

open Accounting.TS Accounting.TS.Headline Accounting.Assumptions.TS Pouw.RelativeSecurity
  Accounting.SecurityProofs.TS.Headline

/-- sm_80's headline at the data sheet: `SheetDIdealFloor` at `π*`, `p = ¾`, `S = 48,000` and `δ = 2^-40`, (I8) from
its pieces (`ProofsSoundOfPinned`), FILL-OPT from the data, and the terms bounded by `sm80_eps_le`. -/
theorem Sm80SheetHeadline : Accounting.Guarantees.TS.Sm80SheetHeadline := by
  intro Ω _ Pl V nπ pf δbind δzk hδ hSplit h1 h2 h3 h4 h5 h6 h7 hF hL hOf hB hZK hPin hRF
  have hPS := ProofsSoundOfPinned Ω Sm80.checks nπ Pl pf Sm80.model.Nw δbind δzk hOf hB hZK hPin hRF
  have hδlo : (1 / 2 : ℝ) ^ 41 ≤ (1 / 2) ^ 40 - (Sm80.checks : ℝ) * (δbind + nπ * δzk) := by
    have e : ((1 : ℝ) / 2) ^ 40 = 2 * (1 / 2) ^ 41 := by norm_num
    linarith
  have hδpos : 0 < (1 / 2 : ℝ) ^ 40 - (Sm80.checks : ℝ) * (δbind + nπ * δzk) :=
    lt_of_lt_of_le (by positivity) hδlo
  have hmain := SheetDIdealFloor Sm80.sheet Sm80.model Sm80.flat Ω Sm80.checks Pl (onFacet Sm80.piStar)
    (3 / 4) ((1 / 2) ^ 40) δbind δzk nπ rfl (isCert_onFacet _) sm80_wf (by decide) (by norm_num) (by norm_num)
    hδpos hSplit (le_of_eq sm80_q) (le_of_eq sm80_qT) le_rfl h1 h2 h3 h4 h5 h6 h7 hF (sm80_fillOpt Pl hSplit) hL
    hPS
  refine le_trans (prob_mono ?_) hmain
  rintro x ⟨ha, hlt⟩
  exact ⟨ha, lt_of_lt_of_le hlt (sm80_eps_le _)⟩

end Accounting.SecurityProofs.TS
