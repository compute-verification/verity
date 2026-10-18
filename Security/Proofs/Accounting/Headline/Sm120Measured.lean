import Proofs.Accounting.Headline.Sm120

/-!
# sm_120's headline on its measured vector

`Sm120MeasuredHeadline` is `DIdealFloor` at sm_120's measured data (`Definitions.Accounting.Headline.Sm120Measured`),
with the data's hypotheses discharged and the terms bounded as at the data sheet:

* the data, by `decide`: π* (JI3s_0) is the floor program's largest facet of VF3r-16's 109 (`sm120m_max`), so `B_♭`
  is its load there, `≈ 19.829` (`sm120m_bnd`); no writer kind's word is cheaper there than IMMA16832's
  (`sm120m_writers`), nor a vector or table word at `9/8` (`sm120m_vec`), so `w*` is IMMA16832's `≈ 0.03371`
  (`sm120m_wStar`); each fold route costs at least the lane fold's `≈ 0.1204` there (`sm120m_phiStar`);
* (I5) `SideUnits` and FILL-OPT hold at any run: the model has no side unit and no filler;
* `identity` at `d̄ = ā = 0` gives `ε_skip + ε_cert + ε_route + ε_mand ≤ 1 − (N_w w* + N_w φ*)/B_♭`, `epsJl_le`
  bounds `ε_jl` by `κ·41 ln 2/(48,000 · ⅛)` at `p = 7/8`, and their sum is below `591/100000`, by `decide` on the
  rationals (`sm120m_final`);
* (I7) at `7/8` is the receipt split's (`luck_of_receipt_half`).
-/

namespace Accounting.SecurityProofs.TS.Headline

open Accounting.TS Accounting.TS.Headline Accounting.Assumptions.TS Pouw.RelativeSecurity Accounting.SecurityProofs.TS

/-- `ε_jl ≤ κ·41·0.6931471808/6000` at `p = 7/8`, `48,000` rounds and `δ′ ≥ 2^-41`, from `ln 2 < 0.6931471808` and
`ln(8/7) ≥ ⅛`. -/
theorem epsJl_le {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F]
    [Fintype X] (Z : Model K R T L U F X) (Fl : Labelled K) (lam : R → ℝ) (hκ : 0 ≤ Z.kappa Fl lam) (S : ℕ)
    (hS : (S : ℝ) = 48000) (δ' : ℝ) (hδ' : (1 / 2 : ℝ) ^ 41 ≤ δ') :
    Z.epsJl Fl lam (7 / 8) δ' S ≤ Z.kappa Fl lam * (41 * (6931471808 / 10000000000) / 6000) := by
  have hlog2 : Real.log 2 < 6931471808 / 10000000000 := lt_of_lt_of_eq Real.log_two_lt_d9 (by norm_num)
  have hδpos : 0 < δ' := lt_of_lt_of_le (by positivity) hδ'
  have hinv : 1 / δ' ≤ (2 : ℝ) ^ 41 := by
    have h := one_div_le_one_div_of_le (by positivity : (0 : ℝ) < (1 / 2) ^ 41) hδ'
    rwa [show (1 : ℝ) / (1 / 2) ^ 41 = 2 ^ 41 by norm_num] at h
  have hlogδ : Real.log (1 / δ') ≤ 41 * (6931471808 / 10000000000) := by
    calc Real.log (1 / δ') ≤ Real.log ((2 : ℝ) ^ 41) := Real.log_le_log (one_div_pos.mpr hδpos) hinv
      _ = 41 * Real.log 2 := by rw [Real.log_pow]; norm_num
      _ ≤ 41 * (6931471808 / 10000000000) := by linarith
  have hlogp : (1 : ℝ) / 8 ≤ Real.log (1 / (7 / 8)) := by
    calc (1 : ℝ) / 8 = 1 - (1 / (7 / 8))⁻¹ := by norm_num
      _ ≤ Real.log (1 / (7 / 8)) := Real.one_sub_inv_le_log_of_pos (by norm_num)
  have hq : Real.log (1 / δ') / (Real.log (1 / (7 / 8)) * 48000) ≤ 41 * (6931471808 / 10000000000) / 6000 :=
    div_le_div₀ (by norm_num) hlogδ (by norm_num) (by linarith)
  unfold Model.epsJl
  rw [hS]
  exact (mul_div_assoc _ _ _).trans_le (mul_le_mul_of_nonneg_left hq hκ)

/-- VF3r-16's row `r` as rationals, by class index. -/
def vfCoef (r i : ℕ) : ℚ := Sparse.coeff (Sm120Measured.vf3r16.getD r []) i

/-- The measured machine's row `r`. -/
def vfRow (r : Fin 109) (i : ℕ) : ℚ := vfCoef r.val i

/-- The measured floor program's counts, its four lists in `labelled`'s order. -/
def flatAllM : Sparse := Sm120.flatWords ++ Sm120.flatFold ++ [] ++ Sm120Measured.flatSkip

/-- `B_♭` on VF3r-16 as a rational: the floor program's load on π*. -/
def BqM : ℚ := dot (vfCoef Sm120Measured.piStar.val) 122 flatAllM

/-- `w*` at π* as a rational: the writer kind `wStarKind`'s coefficient there over its `ω`. -/
def wqM : ℚ :=
  vfCoef Sm120Measured.piStar.val ((Sm120Measured.h1fx1.writers.getD Sm120Measured.wStarKind.val (0, 0)).1 % 122) /
    (Sm120Measured.h1fx1.writers.getD Sm120Measured.wStarKind.val (0, 0)).2

/-- The least vector-class coefficient at π* that keeps the scalar routes at `9/8` from undercutting `w*`. -/
def vqM : ℚ := 256 / 9 * wqM

/-- `φ*` at π* as a rational: the lane fold's load there. -/
def phiqM : ℚ := dot (vfCoef Sm120Measured.piStar.val) 122 (Sm120Measured.h1fx1.folds.getD 0 [])

theorem sm120m_max : ∀ i < 109, dot (vfCoef i) 122 flatAllM ≤ BqM := by decide +kernel

theorem sm120m_Bpos : (0 : ℚ) < BqM := by decide +kernel

theorem sm120m_wpos : (0 : ℚ) < wqM := by decide +kernel

theorem sm120m_writers : ∀ i < 15, 0 < (Sm120Measured.h1fx1.writers.getD i (0, 0)).2 ∧
    wqM * (Sm120Measured.h1fx1.writers.getD i (0, 0)).2 ≤
      vfCoef Sm120Measured.piStar.val ((Sm120Measured.h1fx1.writers.getD i (0, 0)).1 % 122) := by
  decide +kernel

theorem sm120m_vec : ∀ j ∈ Sm120Measured.h1fx1.vec, vqM ≤ vfCoef Sm120Measured.piStar.val (j % 122) := by
  decide +kernel

theorem sm120m_wv : wqM ≤ 9 / 8 / 32 * vqM := by decide +kernel

theorem sm120m_fold :
    ∀ i < 2, phiqM ≤ dot (vfCoef Sm120Measured.piStar.val) 122 (Sm120Measured.h1fx1.folds.getD i []) := by
  decide +kernel

theorem sm120m_final_q :
    (1 - (128 * wqM + 128 * phiqM) / BqM + 128 * wqM / BqM * (41 * (6931471808 / 10000000000) / 6000) : ℚ) ≤
      591 / 100000 := by
  decide +kernel

theorem sm120m_hG : ∀ r k, Sm120Measured.model.G.a r k = (vfRow r k.val : ℝ) := fun _ _ => rfl

theorem sm120m_a_nonneg : ∀ r k, 0 ≤ Sm120Measured.model.G.a r k := fun r k => by
  rw [sm120m_hG]
  exact Rat.cast_nonneg.mpr (coeff_nonneg _ _)

theorem sm120m_bnd : Sm120Measured.model.bnd Sm120Measured.flat = (BqM : ℝ) := by
  unfold Model.bnd
  rw [show Sm120Measured.flat.n = Sparse.counts flatAllM from
    labelled_n 122 Sm120.flatWords Sm120.flatFold [] Sm120Measured.flatSkip]
  exact bound_eq _ vfRow sm120m_hG flatAllM Sm120Measured.piStar fun r => sm120m_max r.val r.isLt

theorem sm120m_Nw : Sm120Measured.model.Nw = 128 := by
  show ((128 : ℕ) : ℝ) = 128
  norm_num

theorem sm120m_q : Sm120Measured.model.q = 9 / 8 := by
  show ((9 / 8 : ℚ) : ℝ) = 9 / 8
  norm_num

theorem sm120m_qT : Sm120Measured.model.qT = 9 / 8 := by
  show ((9 / 8 : ℚ) : ℝ) = 9 / 8
  norm_num

theorem sm120m_wf : Sm120Measured.model.Wf Sm120Measured.flat := by
  refine ⟨sm120m_a_nonneg, ?_, ?_, ?_, fun t => ?_, ?_, ?_, le_rfl, fun u => u.elim, fun φ r => ?_, fun _ _ => le_rfl,
    le_rfl, ?_, le_rfl, ?_⟩
  · show (0 : ℝ) < ((2430 : ℕ) : ℝ)
    norm_num
  · rw [sm120m_Nw]
    norm_num
  · rw [sm120m_bnd]
    exact_mod_cast sm120m_Bpos
  · show (0 : ℝ) < ((Sm120Measured.h1fx1.writers.getD t.val (0, 0)).2 : ℝ)
    exact_mod_cast (sm120m_writers t.val t.isLt).1
  · rw [sm120m_q]
    norm_num
  · rw [sm120m_qT]
    norm_num
  · show 0 ≤ Sm120Measured.model.G.load (Sparse.counts (Sm120Measured.h1fx1.folds.getD φ.val [])) r
    exact load_counts_nonneg _ sm120m_a_nonneg _ _
  · show (0 : ℝ) ≤ 1
    norm_num
  · show (0 : ℝ) ≤ 1
    norm_num

theorem sm120m_vecMin : (vqM : ℝ) ≤ Sm120Measured.model.vecMin Sm120Measured.piStar := by
  have : Nonempty ↥Sm120Measured.model.vec :=
    ⟨⟨idx 122 13, List.mem_toFinset.mpr (List.mem_map.mpr ⟨13, by decide, rfl⟩)⟩⟩
  unfold Model.vecMin
  refine le_ciInf fun k => ?_
  obtain ⟨j, hj, hk⟩ := List.mem_map.mp (List.mem_toFinset.mp k.2)
  rw [sm120m_hG, ← hk]
  exact Rat.cast_le.mpr (sm120m_vec j hj)

/-- `w*` at `π*` is the writer kind `wStarKind`'s (IMMA16832's) word: every other writer kind's is at least that
(`sm120m_writers`), and so are the vector and table routes' at `9/8` (`sm120m_vec`). -/
theorem sm120m_wStar : Sm120Measured.model.wStar (onFacet Sm120Measured.piStar) = (wqM : ℝ) := by
  have hvec : (wqM : ℝ) ≤ 9 / 8 / 32 * Sm120Measured.model.vecMin Sm120Measured.piStar := by
    have h0 : (wqM : ℝ) ≤ ((9 / 8 / 32 * vqM : ℚ) : ℝ) := Rat.cast_le.mpr sm120m_wv
    have h : (wqM : ℝ) ≤ 9 / 8 / 32 * (vqM : ℝ) := by
      push_cast at h0
      linarith
    exact h.trans (mul_le_mul_of_nonneg_left sm120m_vecMin (by norm_num))
  refine le_antisymm
    ((iInf_le_price (onFacet Sm120Measured.piStar) Sm120Measured.model.demand
      (.writer Sm120Measured.wStarKind)).trans (le_of_eq ?_)) ?_
  · rw [demandPrice_onFacet]
    show (vfCoef Sm120Measured.piStar.val
        ((Sm120Measured.h1fx1.writers.getD Sm120Measured.wStarKind.val (0, 0)).1 % 122) : ℝ) /
        ((Sm120Measured.h1fx1.writers.getD Sm120Measured.wStarKind.val (0, 0)).2 : ℝ) = (wqM : ℝ)
    first
      | rw [wqM, Rat.cast_div, Rat.cast_natCast]
      | simp only [wqM, Rat.cast_div, Rat.cast_natCast]
      | (unfold wqM; push_cast; ring)
  · have : Nonempty (Head (Fin 15) Unit Empty) := ⟨.vector⟩
    unfold Model.wStar
    refine le_ciInf fun h => ?_
    rw [demandPrice_onFacet]
    cases h with
    | writer t =>
      have hw := sm120m_writers t.val t.isLt
      show (wqM : ℝ) ≤ (vfCoef Sm120Measured.piStar.val
          ((Sm120Measured.h1fx1.writers.getD t.val (0, 0)).1 % 122) : ℝ) /
          ((Sm120Measured.h1fx1.writers.getD t.val (0, 0)).2 : ℝ)
      rw [le_div_iff₀ (by exact_mod_cast hw.1)]
      have h : ((wqM * ((Sm120Measured.h1fx1.writers.getD t.val (0, 0)).2 : ℚ) : ℚ) : ℝ) ≤
          ((vfCoef Sm120Measured.piStar.val ((Sm120Measured.h1fx1.writers.getD t.val (0, 0)).1 % 122) : ℚ) : ℝ) :=
        Rat.cast_le.mpr hw.2
      push_cast at h
      linarith
    | vector =>
      show (wqM : ℝ) ≤ Sm120Measured.model.q / 32 * Sm120Measured.model.vecMin Sm120Measured.piStar
      rw [sm120m_q]
      exact hvec
    | table l =>
      show (wqM : ℝ) ≤ max (Sm120Measured.model.qT / 32 * Sm120Measured.model.vecMin Sm120Measured.piStar)
        (Sm120Measured.model.bT * Sm120Measured.model.G.a Sm120Measured.piStar (Sm120Measured.model.bytes l))
      refine le_trans ?_ (le_max_left _ _)
      rw [sm120m_qT]
      exact hvec
    | side u => exact u.elim

/-- `φ*` at `π*` is at least the lane fold's load there. -/
theorem sm120m_phiStar : (phiqM : ℝ) ≤ Sm120Measured.model.phiStar (onFacet Sm120Measured.piStar) := by
  unfold Model.phiStar
  refine le_ciInf fun φ => ?_
  rw [demandPrice_onFacet]
  show (phiqM : ℝ) ≤
    Sm120Measured.model.G.load (Sparse.counts (Sm120Measured.h1fx1.folds.getD φ.val [])) Sm120Measured.piStar
  rw [load_counts _ vfRow sm120m_hG]
  exact Rat.cast_le.mpr (sm120m_fold φ.val φ.isLt)

/-- FILL-OPT on any run: the model has no filler, so `χ* = 0` and the run's filler is priced `≥ 0`. -/
theorem sm120m_fillOpt {Ω : Type} [Fintype Ω] (Pl : Play Ω Sm120.checks (Fin 122) (Fin 15) Unit Empty)
    (hSplit : ∀ x, (Pl.run x).Split) : ∀ x, FillOpt Sm120Measured.model (Pl.run x) := by
  intro x lam hlam
  have hchi : Sm120Measured.model.chiStar lam ≤ 0 := by
    refine (iInf_le_price lam Sm120Measured.model.fill ()).trans (le_of_eq ?_)
    show ∑ r, lam r * (0 : ℝ) = 0
    simp
  have hprice : 0 ≤ Sm120Measured.model.G.price lam (Pl.run x).nfill := by
    unfold Machine.price Machine.load
    exact Finset.sum_nonneg fun r _ => mul_nonneg (hlam.1 r)
      (Finset.sum_nonneg fun k _ => mul_nonneg (sm120m_a_nonneg r k) ((hSplit x).2.2.1 k))
  show (1 - (0 : ℝ)) * Sm120Measured.model.chiStar lam ≤ _
  rw [sub_zero, one_mul]
  exact hchi.trans hprice

theorem sm120m_final :
    1 - (128 * (wqM : ℝ) + 128 * (phiqM : ℝ)) / Sm120Measured.model.bnd Sm120Measured.flat +
        128 * (wqM : ℝ) / Sm120Measured.model.bnd Sm120Measured.flat * (41 * (6931471808 / 10000000000) / 6000) ≤
      591 / 100000 := by
  rw [sm120m_bnd]
  have h : ((1 - (128 * wqM + 128 * phiqM) / BqM + 128 * wqM / BqM * (41 * (6931471808 / 10000000000) / 6000) : ℚ) :
      ℝ) ≤ ((591 / 100000 : ℚ) : ℝ) := Rat.cast_le.mpr sm120m_final_q
  push_cast at h
  linarith

/-- The terms at `π*`, `p = 7/8` and `δ′ ≥ 2^-41` sum to at most `591/100000`. -/
theorem sm120m_eps_le (δ' : ℝ) (hδ' : (1 / 2 : ℝ) ^ 41 ≤ δ') :
    1 - 591 / 100000 ≤ 1 - Sm120Measured.model.epsSkip Sm120Measured.flat (onFacet Sm120Measured.piStar) -
      Sm120Measured.model.epsCert Sm120Measured.flat (onFacet Sm120Measured.piStar) -
      Sm120Measured.model.epsRoute Sm120Measured.flat (onFacet Sm120Measured.piStar) -
      Sm120Measured.model.epsMand Sm120Measured.flat (onFacet Sm120Measured.piStar) -
      Sm120Measured.model.epsJl Sm120Measured.flat (onFacet Sm120Measured.piStar) (7 / 8) δ' Sm120.checks := by
  have hid := identity Sm120Measured.model Sm120Measured.flat (onFacet Sm120Measured.piStar) 0 0 sm120m_wf
  have hBpos : 0 < Sm120Measured.model.bnd Sm120Measured.flat := sm120m_wf.2.2.2.1
  have hχ := (stars_nonneg Sm120Measured.model Sm120Measured.flat (onFacet Sm120Measured.piStar) sm120m_wf
    (isCert_onFacet _)).2.2.2
  have hφ := sm120m_phiStar
  have hN : 128 * (wqM : ℝ) + 128 * (phiqM : ℝ) ≤
      (1 - 0 - 0) * Sm120Measured.model.Nw * Sm120Measured.model.wStar (onFacet Sm120Measured.piStar) +
      (1 - Sm120Measured.model.muF) * Sm120Measured.model.Nw *
        Sm120Measured.model.phiStar (onFacet Sm120Measured.piStar) +
      (1 - Sm120Measured.model.muX) * Sm120Measured.model.chiStar (onFacet Sm120Measured.piStar) := by
    rw [sm120m_Nw, sm120m_wStar, show Sm120Measured.model.muF = 0 from rfl,
      show Sm120Measured.model.muX = 0 from rfl]
    linarith
  have hA := div_le_div_of_nonneg_right hN hBpos.le
  have hLuck : Sm120Measured.model.epsLuck Sm120Measured.flat (onFacet Sm120Measured.piStar) 0 = 0 := zero_mul _
  have hAdapt : Sm120Measured.model.epsAdapt Sm120Measured.flat (onFacet Sm120Measured.piStar) 0 = 0 := zero_mul _
  have hκ : Sm120Measured.model.kappa Sm120Measured.flat (onFacet Sm120Measured.piStar) =
      128 * (wqM : ℝ) / Sm120Measured.model.bnd Sm120Measured.flat := by
    unfold Model.kappa
    rw [sm120m_Nw, sm120m_wStar]
  have hκ0 : 0 ≤ Sm120Measured.model.kappa Sm120Measured.flat (onFacet Sm120Measured.piStar) := by
    rw [hκ]
    exact div_nonneg (mul_nonneg (by norm_num) (by exact_mod_cast sm120m_wpos.le)) hBpos.le
  have hS : ((Sm120.checks : ℕ) : ℝ) = 48000 := by
    show ((48000 : ℕ) : ℝ) = 48000
    norm_num
  have hjl := epsJl_le Sm120Measured.model Sm120Measured.flat (onFacet Sm120Measured.piStar) hκ0 Sm120.checks hS
    δ' hδ'
  rw [hκ] at hjl
  have hfin := sm120m_final
  linarith

end Accounting.SecurityProofs.TS.Headline

namespace Accounting.SecurityProofs.TS

open Accounting.TS Accounting.TS.Headline Accounting.Assumptions.TS Pouw.RelativeSecurity
  Accounting.SecurityProofs.TS.Headline

/-- sm_120's headline on VF3r-16: `DIdealFloor` at `π*`, `p = 7/8`, `S = 48,000` and `δ = 2^-40`, (I7) from the
receipt split (`luck_of_receipt_half`), (I8) from its pieces (`ProofsSoundOfPinned`), (I5) `SideUnits` and FILL-OPT
from the data, and the terms bounded by `sm120m_eps_le`. -/
theorem Sm120MeasuredHeadline : Accounting.Guarantees.TS.Sm120MeasuredHeadline := by
  intro α ι _ _ _ R _ Pl G V nπ pf δbind δzk hδ hSplit h1 hW hSR hN hF hG hPinG hCube hRL hOf hB hZK hPin hRF
  have hL := luck_of_receipt_half R Pl.game G hG hPinG hCube hRL
  have hPS := ProofsSoundOfPinned (α × Finset ι) Sm120.checks nπ Pl pf Sm120Measured.model.Nw δbind δzk hOf hB hZK
    hPin hRF
  have hδlo : (1 / 2 : ℝ) ^ 41 ≤ (1 / 2) ^ 40 - (Sm120.checks : ℝ) * (δbind + nπ * δzk) := by
    have e : ((1 : ℝ) / 2) ^ 40 = 2 * (1 / 2) ^ 41 := by norm_num
    linarith
  have hδpos : 0 < (1 / 2 : ℝ) ^ 40 - (Sm120.checks : ℝ) * (δbind + nπ * δzk) :=
    lt_of_lt_of_le (by positivity) hδlo
  have hmain := DIdealFloor Sm120Measured.model Sm120Measured.flat (α × Finset ι) Sm120.checks Pl
    (onFacet Sm120Measured.piStar) (7 / 8) ((1 / 2) ^ 40) δbind δzk nπ (isCert_onFacet _) sm120m_wf (by decide)
    (by norm_num) (by norm_num) hδpos hSplit h1 hW hSR (fun _ u => u.elim) hN hF (sm120m_fillOpt Pl hSplit) hL hPS
  refine le_trans (prob_mono ?_) hmain
  rintro x ⟨ha, hlt⟩
  exact ⟨ha, lt_of_lt_of_le hlt (sm120m_eps_le _ hδlo)⟩

end Accounting.SecurityProofs.TS
