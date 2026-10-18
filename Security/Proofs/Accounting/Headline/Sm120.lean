import Proofs.Accounting.Headline.Encoding
import Proofs.Accounting.TS
import Models.Accounting.Guarantees.Headline

/-!
# sm_120's headline at the data sheet

`Sm120SheetHeadline` is `SheetDIdealFloor` at sm_120's data (`Definitions.Accounting.Headline.Sm120`), with the data's
hypotheses discharged and the terms bounded:

* the data, by `decide`: issue is the floor program's largest facet (`sm120_max`), so `B_♭` is its issue count
  `B_♭ ≈ 17.098` (`sm120_bnd`); every writer kind's `ω ≤ 128` and IMMA16832's is `128`, so `w* = 1/128`
  (`sm120_wStar`); each fold route costs at least `1/8` on issue (`sm120_phiStar`);
* the certificate is `π*`, the issue facet (`isCert_onFacet`), and FILL-OPT holds since there is no filler
  (`sm120_fillOpt`);
* `identity` at `d̄ = ā = 0` gives `ε_skip + ε_cert + ε_route + ε_mand ≤ 1 − (N_w w* + N_w φ*)/B_♭ ≤ 1 − 17/B_♭`, and
  `ε_jl ≤ (1/B_♭)·41 ln 2/(48,000 · ¼)` from `δ′ ≥ 2^-41`, `ln 2 < 0.6931471808` and `ln(4/3) ≥ ¼` (`sm120_jl`);
* their sum is below `117/20000`, by `decide` on the rationals (`sm120_final`).
-/

namespace Accounting.SecurityProofs.TS.Headline

open Accounting.TS Accounting.TS.Headline Accounting.Assumptions.TS Pouw.RelativeSecurity Accounting.SecurityProofs.TS

/-- The floor program's counts, its four lists in `labelled`'s order. -/
def flatAll : Sparse := Sm120.flatWords ++ Sm120.flatFold ++ [] ++ Sm120.flatSkip

/-- `B_♭` as a rational: the floor program's issue count. -/
def Bq : ℚ := dot (sheetCoef Sm120.ds120 .issue) 122 flatAll

theorem sm120_max (r : SheetRow Sm120.Pipe) :
    dot (sheetCoef Sm120.ds120 r) 122 flatAll ≤ dot (sheetCoef Sm120.ds120 .issue) 122 flatAll := by
  cases r with
  | issue => exact le_rfl
  | tensor => decide +kernel
  | pipe p => cases p <;> decide +kernel
  | smem => decide +kernel

theorem sm120_Bpos : (0 : ℚ) < Bq := by decide +kernel

theorem sm120_omega :
    ∀ i < 17, 0 < (Sm120.h1fx1.writers.getD i (0, 0)).2 ∧ (Sm120.h1fx1.writers.getD i (0, 0)).2 ≤ 128 := by
  decide +kernel

theorem sm120_fold :
    ∀ i < 2, (1 / 8 : ℚ) ≤ dot (sheetCoef Sm120.ds120 .issue) 122 (Sm120.h1fx1.folds.getD i []) := by
  decide +kernel

theorem sm120_final_q :
    (1 - 17 / Bq + 1 / Bq * (41 * (6931471808 / 10000000000) / 12000) : ℚ) ≤ 117 / 20000 := by
  decide +kernel

theorem sm120_hG : ∀ r k, Sm120.model.G.a r k = (sheetCoef Sm120.ds120 r k.val : ℝ) := sheet_a Sm120.ds120 122

theorem sm120_a_nonneg : ∀ r k, 0 ≤ Sm120.model.G.a r k := fun r k => by
  rw [sm120_hG]
  exact Rat.cast_nonneg.mpr (sheetCoef_nonneg _ _ _)

theorem sm120_bnd : Sm120.model.bnd Sm120.flat = (Bq : ℝ) := by
  unfold Model.bnd
  rw [show Sm120.flat.n = Sparse.counts flatAll from labelled_n 122 Sm120.flatWords Sm120.flatFold [] Sm120.flatSkip]
  exact bound_eq _ (sheetCoef Sm120.ds120) sm120_hG flatAll .issue sm120_max

theorem sm120_Nw : Sm120.model.Nw = 128 := by
  show ((128 : ℕ) : ℝ) = 128
  norm_num

theorem sm120_q : Sm120.model.q = 1 / 2 := by
  show ((1 / 2 : ℚ) : ℝ) = 1 / 2
  norm_num

theorem sm120_qT : Sm120.model.qT = 1 / 2 := by
  show ((1 / 2 : ℚ) : ℝ) = 1 / 2
  norm_num

theorem sm120_wf : Sm120.model.Wf Sm120.flat := by
  refine ⟨sm120_a_nonneg, ?_, ?_, ?_, fun t => ?_, ?_, ?_, le_rfl, fun u => u.elim, fun φ r => ?_, fun _ _ => le_rfl,
    le_rfl, ?_, le_rfl, ?_⟩
  · show (0 : ℝ) < ((2430 : ℕ) : ℝ)
    norm_num
  · rw [sm120_Nw]
    norm_num
  · rw [sm120_bnd]
    exact_mod_cast sm120_Bpos
  · show (0 : ℝ) < ((Sm120.h1fx1.writers.getD t.val (0, 0)).2 : ℝ)
    exact_mod_cast (sm120_omega t.val t.isLt).1
  · rw [sm120_q]
    norm_num
  · rw [sm120_qT]
    norm_num
  · show 0 ≤ Sm120.model.G.load (Sparse.counts (Sm120.h1fx1.folds.getD φ.val [])) r
    exact load_counts_nonneg _ sm120_a_nonneg _ _
  · show (0 : ℝ) ≤ 1
    norm_num
  · show (0 : ℝ) ≤ 1
    norm_num

theorem sm120_vecMin : 1 ≤ Sm120.model.vecMin .issue := by
  haveI : Nonempty ↥Sm120.model.vec :=
    ⟨⟨idx 122 13, List.mem_toFinset.mpr (List.mem_map.mpr ⟨13, by decide, rfl⟩)⟩⟩
  unfold Model.vecMin
  refine le_ciInf fun k => ?_
  rw [sm120_hG]
  simp [sheetCoef]

/-- `w* = 1/128` at `π*`: IMMA16832 (writer kind 6) is `1/128` on issue, every other word route at least that. -/
theorem sm120_wStar : Sm120.model.wStar (onFacet Sm120.piStar) = 1 / 128 := by
  refine le_antisymm
    ((iInf_le_price (onFacet Sm120.piStar) Sm120.model.demand (.writer 6)).trans (le_of_eq ?_)) ?_
  · rw [demandPrice_onFacet]
    show (1 : ℝ) / ((128 : ℕ) : ℝ) = 1 / 128
    norm_num
  · haveI : Nonempty (Head (Fin 17) Unit Empty) := ⟨.vector⟩
    unfold Model.wStar
    refine le_ciInf fun h => ?_
    rw [demandPrice_onFacet]
    cases h with
    | writer t =>
      show (1 : ℝ) / 128 ≤ 1 / ((Sm120.h1fx1.writers.getD t.val (0, 0)).2 : ℝ)
      exact one_div_le_one_div_of_le (by exact_mod_cast (sm120_omega t.val t.isLt).1)
        (by exact_mod_cast (sm120_omega t.val t.isLt).2)
    | vector =>
      show (1 : ℝ) / 128 ≤ Sm120.model.q / 32 * Sm120.model.vecMin .issue
      rw [sm120_q]
      calc (1 : ℝ) / 128 ≤ 1 / 2 / 32 * 1 := by norm_num
        _ ≤ 1 / 2 / 32 * Sm120.model.vecMin .issue := mul_le_mul_of_nonneg_left sm120_vecMin (by norm_num)
    | table l =>
      show (1 : ℝ) / 128 ≤ max (Sm120.model.qT / 32 * Sm120.model.vecMin .issue)
        (Sm120.model.bT * Sm120.model.G.a .issue (Sm120.model.bytes l))
      refine le_trans ?_ (le_max_left _ _)
      rw [sm120_qT]
      calc (1 : ℝ) / 128 ≤ 1 / 2 / 32 * 1 := by norm_num
        _ ≤ 1 / 2 / 32 * Sm120.model.vecMin .issue := mul_le_mul_of_nonneg_left sm120_vecMin (by norm_num)
    | side u => exact u.elim

/-- `φ* ≥ 1/8` at `π*`: each fold route's issue count. -/
theorem sm120_phiStar : 1 / 8 ≤ Sm120.model.phiStar (onFacet Sm120.piStar) := by
  unfold Model.phiStar
  refine le_ciInf fun φ => ?_
  rw [demandPrice_onFacet]
  show (1 : ℝ) / 8 ≤ Sm120.model.G.load (Sparse.counts (Sm120.h1fx1.folds.getD φ.val [])) .issue
  rw [load_counts _ (sheetCoef Sm120.ds120) sm120_hG]
  exact_mod_cast sm120_fold φ.val φ.isLt

/-- FILL-OPT on any run: the model has no filler, so `χ* = 0` and the run's filler is priced `≥ 0`. -/
theorem sm120_fillOpt {Ω : Type} [Fintype Ω] (Pl : Play Ω Sm120.checks (Fin 122) (Fin 17) Unit Empty)
    (hSplit : ∀ x, (Pl.run x).Split) : ∀ x, FillOpt Sm120.model (Pl.run x) := by
  intro x lam hlam
  have hchi : Sm120.model.chiStar lam ≤ 0 := by
    refine (iInf_le_price lam Sm120.model.fill ()).trans (le_of_eq ?_)
    show ∑ r, lam r * (0 : ℝ) = 0
    simp
  have hprice : 0 ≤ Sm120.model.G.price lam (Pl.run x).nfill := by
    unfold Machine.price Machine.load
    exact Finset.sum_nonneg fun r _ => mul_nonneg (hlam.1 r)
      (Finset.sum_nonneg fun k _ => mul_nonneg (sm120_a_nonneg r k) ((hSplit x).2.2.1 k))
  show (1 - (0 : ℝ)) * Sm120.model.chiStar lam ≤ _
  rw [sub_zero, one_mul]
  exact hchi.trans hprice

/-- `ε_jl ≤ (1/B_♭)·41·0.6931471808/12000` at `δ′ ≥ 2^-41`. -/
theorem sm120_jl (δ' : ℝ) (hδ' : (1 / 2 : ℝ) ^ 41 ≤ δ') :
    Sm120.model.epsJl Sm120.flat (onFacet Sm120.piStar) (3 / 4) δ' Sm120.checks ≤
      1 / Sm120.model.bnd Sm120.flat * (41 * (6931471808 / 10000000000) / 12000) := by
  have hBpos : 0 < Sm120.model.bnd Sm120.flat := sm120_wf.2.2.2.1
  have hκ : Sm120.model.kappa Sm120.flat (onFacet Sm120.piStar) = 1 / Sm120.model.bnd Sm120.flat := by
    unfold Model.kappa
    rw [sm120_Nw, sm120_wStar, show (128 : ℝ) * (1 / 128) = 1 by norm_num]
  have hlog2 : Real.log 2 < 6931471808 / 10000000000 := lt_of_lt_of_eq Real.log_two_lt_d9 (by norm_num)
  have hδpos : 0 < δ' := lt_of_lt_of_le (by positivity) hδ'
  have hinv : 1 / δ' ≤ (2 : ℝ) ^ 41 := by
    have h := one_div_le_one_div_of_le (by positivity : (0 : ℝ) < (1 / 2) ^ 41) hδ'
    rwa [show (1 : ℝ) / (1 / 2) ^ 41 = 2 ^ 41 by norm_num] at h
  have hlogδ : Real.log (1 / δ') ≤ 41 * (6931471808 / 10000000000) := by
    calc Real.log (1 / δ') ≤ Real.log ((2 : ℝ) ^ 41) := Real.log_le_log (one_div_pos.mpr hδpos) hinv
      _ = 41 * Real.log 2 := by rw [Real.log_pow]; norm_num
      _ ≤ 41 * (6931471808 / 10000000000) := by linarith
  have hlogp : (1 : ℝ) / 4 ≤ Real.log (1 / (3 / 4)) := by
    calc (1 : ℝ) / 4 = 1 - (1 / (3 / 4))⁻¹ := by norm_num
      _ ≤ Real.log (1 / (3 / 4)) := Real.one_sub_inv_le_log_of_pos (by norm_num)
  have hq : Real.log (1 / δ') / (Real.log (1 / (3 / 4)) * 48000) ≤ 41 * (6931471808 / 10000000000) / 12000 :=
    div_le_div₀ (by norm_num) hlogδ (by norm_num) (by linarith)
  have hS : ((Sm120.checks : ℕ) : ℝ) = 48000 := by
    show ((48000 : ℕ) : ℝ) = 48000
    norm_num
  unfold Model.epsJl
  rw [hκ, hS]
  exact (mul_div_assoc _ _ _).trans_le (mul_le_mul_of_nonneg_left hq (one_div_pos.mpr hBpos).le)

theorem sm120_final :
    1 - 17 / Sm120.model.bnd Sm120.flat +
        1 / Sm120.model.bnd Sm120.flat * (41 * (6931471808 / 10000000000) / 12000) ≤ 117 / 20000 := by
  rw [sm120_bnd]
  exact_mod_cast sm120_final_q

/-- The terms at `π*` and `δ′ ≥ 2^-41` sum to at most `117/20000`. -/
theorem sm120_eps_le (δ' : ℝ) (hδ' : (1 / 2 : ℝ) ^ 41 ≤ δ') :
    1 - 117 / 20000 ≤ 1 - Sm120.model.epsSkip Sm120.flat (onFacet Sm120.piStar) -
      Sm120.model.epsCert Sm120.flat (onFacet Sm120.piStar) -
      Sm120.model.epsRoute Sm120.flat (onFacet Sm120.piStar) -
      Sm120.model.epsMand Sm120.flat (onFacet Sm120.piStar) -
      Sm120.model.epsJl Sm120.flat (onFacet Sm120.piStar) (3 / 4) δ' Sm120.checks := by
  have hid := identity Sm120.model Sm120.flat (onFacet Sm120.piStar) 0 0 sm120_wf
  have hBpos : 0 < Sm120.model.bnd Sm120.flat := sm120_wf.2.2.2.1
  have hχ := (stars_nonneg Sm120.model Sm120.flat (onFacet Sm120.piStar) sm120_wf (isCert_onFacet _)).2.2.2
  have hφ := sm120_phiStar
  have hN : 17 ≤ (1 - 0 - 0) * Sm120.model.Nw * Sm120.model.wStar (onFacet Sm120.piStar) +
      (1 - Sm120.model.muF) * Sm120.model.Nw * Sm120.model.phiStar (onFacet Sm120.piStar) +
      (1 - Sm120.model.muX) * Sm120.model.chiStar (onFacet Sm120.piStar) := by
    rw [sm120_Nw, sm120_wStar, show Sm120.model.muF = 0 from rfl, show Sm120.model.muX = 0 from rfl]
    linarith
  have hA := div_le_div_of_nonneg_right hN hBpos.le
  have hLuck : Sm120.model.epsLuck Sm120.flat (onFacet Sm120.piStar) 0 = 0 := zero_mul _
  have hAdapt : Sm120.model.epsAdapt Sm120.flat (onFacet Sm120.piStar) 0 = 0 := zero_mul _
  have hjl := sm120_jl δ' hδ'
  have hfin := sm120_final
  linarith

end Accounting.SecurityProofs.TS.Headline

namespace Accounting.SecurityProofs.TS

open Accounting.TS Accounting.TS.Headline Accounting.Assumptions.TS Pouw.RelativeSecurity
  Accounting.SecurityProofs.TS.Headline

/-- sm_120's headline at the data sheet: `SheetDIdealFloor` at `π*`, `p = ¾`, `S = 48,000` and `δ = 2^-40`, (I8) from
its pieces (`ProofsSoundOfPinned`), FILL-OPT from the data, and the terms bounded by `sm120_eps_le`. -/
theorem Sm120SheetHeadline : Accounting.Guarantees.TS.Sm120SheetHeadline := by
  intro Ω _ Pl V nπ pf δbind δzk hδ hSplit h1 h2 h3 h4 h5 h6 h7 hF hL hOf hB hZK hPin hRF
  have hPS := ProofsSoundOfPinned Ω Sm120.checks nπ Pl pf Sm120.model.Nw δbind δzk hOf hB hZK hPin hRF
  have hδlo : (1 / 2 : ℝ) ^ 41 ≤ (1 / 2) ^ 40 - (Sm120.checks : ℝ) * (δbind + nπ * δzk) := by
    have e : ((1 : ℝ) / 2) ^ 40 = 2 * (1 / 2) ^ 41 := by norm_num
    linarith
  have hδpos : 0 < (1 / 2 : ℝ) ^ 40 - (Sm120.checks : ℝ) * (δbind + nπ * δzk) :=
    lt_of_lt_of_le (by positivity) hδlo
  have hmain := SheetDIdealFloor Sm120.sheet Sm120.model Sm120.flat Ω Sm120.checks Pl (onFacet Sm120.piStar)
    (3 / 4) ((1 / 2) ^ 40) δbind δzk nπ rfl (isCert_onFacet _) sm120_wf (by decide) (by norm_num) (by norm_num)
    hδpos hSplit (le_of_eq sm120_q) (le_of_eq sm120_qT) le_rfl h1 h2 h3 h4 h5 h6 h7 hF (sm120_fillOpt Pl hSplit) hL
    hPS
  refine le_trans (prob_mono ?_) hmain
  rintro x ⟨ha, hlt⟩
  exact ⟨ha, lt_of_lt_of_le hlt (sm120_eps_le _ hδlo)⟩

end Accounting.SecurityProofs.TS
