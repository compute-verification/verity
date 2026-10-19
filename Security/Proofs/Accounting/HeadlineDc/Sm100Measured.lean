import Proofs.Accounting.HeadlineDc.Sm100

/-!
# The B200's headline on its measured vector

`Sm100MeasuredHeadline` is `Whole.dIdealFloor_whole` at DC-100-ri and `dither-2s-licpw-redux-ldtm0-l32`
(`Definitions.Accounting.HeadlineDc.Sm100Measured`), at the data's certificate `π* = Sm100Measured.lamStar`, the issue
row (`sm100m_lamStar`), `p = 7/8`, `p₀ = ½` and `Λ = max(κ/ln(8/7), (κ + f)/ln 2)`:

* the data, by `decide`: issue is ♭'s largest facet of DC-100-ri's 12 (`sm100m_max`), so `B_♭` is its issue load
  `≈ 1089.745` (`sm100m_bnd`); every writer kind's word costs at least `1/32768` there (`sm100m_writers`), and so do the
  vector and table words at `½` (`sm100m_vec`), so `w* ≥ 1/32768` (`sm100m_wStar`); each fold route costs at least the
  lane fold's `135/4096` there (`sm100m_fold`, `sm100m_phiStar`);
* A6 (`SheetWords`) gives (I2)/(I4) `WordsWritten` and, at `q = q_T = ½` and `b_T = 0`, (I3) `ScalarRoute`
  (`scalarRoute_half`); (I5) `SideUnits` and FILL-OPT hold at any run, since the model has no side unit and no filler;
* the folded branch from those bounds (`folded_of_lower`): below `888/100000` by `decide` (`sm100m_folded_q`);
* the unfolded branch (`unfolded_of_lower`): below `888/100000` by `decide` (`sm100m_unfolded_q`), and it binds
  (0.88723%);
* (I7) at `7/8` is the receipt split's (`luck_of_receipt_half`).
-/

namespace Accounting.SecurityProofs.TS.HeadlineDc

open Accounting.TS Accounting.TS.Headline Accounting.Assumptions.TS Pouw.RelativeSecurity Accounting.SecurityProofs.TS
  Accounting.SecurityProofs.TS.Headline

/-- DC-100-ri's row `r` as rationals, by class index. -/
def vfCoef100 (r i : ℕ) : ℚ := Sparse.coeff (Sm100Measured.dc100ri.getD r []) i

/-- The measured machine's row `r`. -/
def vfRow100 (r : Fin 12) (i : ℕ) : ℚ := vfCoef100 r.val i

/-- ♭'s counts on DC-100-ri, its four lists in `labelled`'s order. -/
def flat100M : Sparse := Sm100.flatWords ++ Sm100.flatFold ++ Sm100.flatFill ++ Sm100Measured.flatSkip

/-- `B_♭` on DC-100-ri as a rational: ♭'s load on issue. -/
def B100M : ℚ := dot (vfCoef100 0) 26 flat100M

/-- The lane fold's load on issue as a rational. -/
def phi100M : ℚ := dot (vfCoef100 0) 26 (Sm100Measured.d2s.folds.getD 0 [])

theorem sm100m_max : ∀ i < 12, dot (vfCoef100 i) 26 flat100M ≤ B100M := by decide +kernel

theorem sm100m_Bpos : (0 : ℚ) < B100M := by decide +kernel

theorem sm100m_writers : ∀ i < 13, 0 < (Sm100Measured.d2s.writers.getD i (0, 0)).2 ∧
    1 / 32768 * ((Sm100Measured.d2s.writers.getD i (0, 0)).2 : ℚ) ≤
      vfCoef100 0 ((Sm100Measured.d2s.writers.getD i (0, 0)).1 % 26) := by
  decide +kernel

theorem sm100m_vec : ∀ j ∈ Sm100Measured.d2s.vec, 1 ≤ vfCoef100 0 (j % 26) := by decide +kernel

theorem sm100m_fold :
    ∀ i < 2, phi100M ≤ dot (vfCoef100 0) 26 (Sm100Measured.d2s.folds.getD i []) := by
  decide +kernel

theorem sm100m_folded_q :
    (1 - 888 / 100000 : ℚ) ≤ (32768 * (1 / 32768) * (1 - 41 * (6931471808 / 10000000000) / 6000) +
      32768 * phi100M) / B100M := by
  decide +kernel

theorem sm100m_unfolded_q :
    (1 - 888 / 100000 : ℚ) ≤ (32768 * (1 / 32768) + 32768 * phi100M) / B100M * (1 - 41 / 48000) := by
  decide +kernel

theorem sm100m_one : Sparse.coeff [(0, 1, 1)] 0 = 1 := by decide +kernel

theorem sm100m_hG : ∀ r k, Sm100Measured.model.G.a r k = (vfRow100 r k.val : ℝ) := fun _ _ => rfl

theorem sm100m_a_nonneg : ∀ r k, 0 ≤ Sm100Measured.model.G.a r k := fun r k => by
  rw [sm100m_hG]
  exact Rat.cast_nonneg.mpr (coeff_nonneg _ _)

theorem sm100m_bnd : Sm100Measured.model.bnd Sm100Measured.flat = (B100M : ℝ) := by
  unfold Model.bnd
  rw [show Sm100Measured.flat.n = Sparse.counts flat100M from
    labelled_n 26 Sm100.flatWords Sm100.flatFold Sm100.flatFill Sm100Measured.flatSkip]
  exact bound_eq _ vfRow100 sm100m_hG flat100M 0 fun r => sm100m_max r.val r.isLt

theorem sm100m_Nw : Sm100Measured.model.Nw = 32768 := by
  show ((32768 : ℕ) : ℝ) = 32768
  norm_num

theorem sm100m_q : Sm100Measured.model.q = 1 / 2 := by
  show ((1 / 2 : ℚ) : ℝ) = 1 / 2
  norm_num

theorem sm100m_qT : Sm100Measured.model.qT = 1 / 2 := by
  show ((1 / 2 : ℚ) : ℝ) = 1 / 2
  norm_num

/-- The data's certificate is the issue row's. -/
theorem sm100m_lamStar : Sm100Measured.lamStar = onFacet 0 := by
  funext r
  show ((Sparse.coeff [(0, 1, 1)] r.val : ℚ) : ℝ) = if r = 0 then 1 else 0
  by_cases h : r = 0
  · rw [if_pos h, h]
    exact_mod_cast sm100m_one
  · rw [if_neg h]
    have hv : r.val ≠ 0 := fun hv => h (Fin.ext hv)
    have hc : Sparse.coeff [(0, 1, 1)] r.val = 0 := by
      simp [Sparse.coeff, Sparse.ratio, Ne.symm hv]
    rw [hc, Rat.cast_zero]

theorem sm100m_isCert : IsCert Sm100Measured.lamStar := by
  rw [sm100m_lamStar]
  exact isCert_onFacet _

theorem sm100m_wf : Sm100Measured.model.Wf Sm100Measured.flat := by
  refine ⟨sm100m_a_nonneg, ?_, ?_, ?_, fun t => ?_, ?_, ?_, le_rfl, fun u => u.elim, fun φ r => ?_, fun _ _ => le_rfl,
    le_rfl, ?_, le_rfl, ?_⟩
  · show (0 : ℝ) < ((1965 : ℕ) : ℝ)
    norm_num
  · rw [sm100m_Nw]
    norm_num
  · rw [sm100m_bnd]
    exact_mod_cast sm100m_Bpos
  · show (0 : ℝ) < ((Sm100Measured.d2s.writers.getD t.val (0, 0)).2 : ℝ)
    exact_mod_cast (sm100m_writers t.val t.isLt).1
  · rw [sm100m_q]
    norm_num
  · rw [sm100m_qT]
    norm_num
  · show 0 ≤ Sm100Measured.model.G.load (Sparse.counts (Sm100Measured.d2s.folds.getD φ.val [])) r
    exact load_counts_nonneg _ sm100m_a_nonneg _ _
  · show (0 : ℝ) ≤ 1
    norm_num
  · show (0 : ℝ) ≤ 1
    norm_num

theorem sm100m_vecMin : 1 ≤ Sm100Measured.model.vecMin 0 := by
  have : Nonempty ↥Sm100Measured.model.vec :=
    ⟨⟨idx 26 1, List.mem_toFinset.mpr (List.mem_map.mpr ⟨1, by decide, rfl⟩)⟩⟩
  unfold Model.vecMin
  refine le_ciInf fun k => ?_
  obtain ⟨j, hj, hk⟩ := List.mem_map.mp (List.mem_toFinset.mp k.2)
  rw [sm100m_hG, ← hk]
  exact_mod_cast sm100m_vec j hj

/-- Every word route costs at least `1/32768` on issue. -/
theorem sm100m_demand_issue : ∀ hd, (1 : ℝ) / 32768 ≤ Sm100Measured.model.demand hd 0 := by
  intro h
  cases h with
  | writer t =>
    have hw := sm100m_writers t.val t.isLt
    show (1 : ℝ) / 32768 ≤ (vfCoef100 (0 : Fin 12).val
        ((Sm100Measured.d2s.writers.getD t.val (0, 0)).1 % 26) : ℝ) /
        ((Sm100Measured.d2s.writers.getD t.val (0, 0)).2 : ℝ)
    rw [le_div_iff₀ (by exact_mod_cast hw.1)]
    have h : ((1 / 32768 * ((Sm100Measured.d2s.writers.getD t.val (0, 0)).2 : ℚ) : ℚ) : ℝ) ≤
        ((vfCoef100 (0 : Fin 12).val ((Sm100Measured.d2s.writers.getD t.val (0, 0)).1 % 26) : ℚ) : ℝ) :=
      Rat.cast_le.mpr hw.2
    push_cast at h
    linarith
  | vector =>
    show (1 : ℝ) / 32768 ≤ Sm100Measured.model.q / 32 * Sm100Measured.model.vecMin 0
    rw [sm100m_q]
    calc (1 : ℝ) / 32768 ≤ 1 / 2 / 32 * 1 := by norm_num
      _ ≤ 1 / 2 / 32 * Sm100Measured.model.vecMin 0 := mul_le_mul_of_nonneg_left sm100m_vecMin (by norm_num)
  | table l =>
    show (1 : ℝ) / 32768 ≤ max (Sm100Measured.model.qT / 32 * Sm100Measured.model.vecMin 0)
      (Sm100Measured.model.bT * Sm100Measured.model.G.a 0 (Sm100Measured.model.bytes l))
    refine le_trans ?_ (le_max_left _ _)
    rw [sm100m_qT]
    calc (1 : ℝ) / 32768 ≤ 1 / 2 / 32 * 1 := by norm_num
      _ ≤ 1 / 2 / 32 * Sm100Measured.model.vecMin 0 := mul_le_mul_of_nonneg_left sm100m_vecMin (by norm_num)
  | side u => exact u.elim

/-- `w* ≥ 1/32768` at π*. -/
theorem sm100m_wStar : (1 : ℝ) * (1 / 32768) ≤ Sm100Measured.model.wStar Sm100Measured.lamStar := by
  have h := wStar_ge Sm100Measured.model Sm100Measured.flat Sm100Measured.lamStar sm100m_isCert sm100m_wf 0 _
    sm100m_demand_issue
  rwa [sm100m_lamStar, show onFacet (0 : Fin 12) 0 = 1 from if_pos rfl] at h

/-- `φ* ≥` the lane fold's load on issue. -/
theorem sm100m_phiStar : (phi100M : ℝ) ≤ Sm100Measured.model.phiStar Sm100Measured.lamStar := by
  refine phiStar_ge_facet Sm100Measured.model Sm100Measured.flat Sm100Measured.lamStar sm100m_isCert sm100m_wf 0 _
    fun φ => ?_
  rw [sm100m_lamStar, show onFacet (0 : Fin 12) 0 = 1 from if_pos rfl, one_mul]
  show (phi100M : ℝ) ≤
    Sm100Measured.model.G.load (Sparse.counts (Sm100Measured.d2s.folds.getD φ.val [])) 0
  rw [load_counts _ vfRow100 sm100m_hG]
  exact_mod_cast sm100m_fold φ.val φ.isLt

theorem sm100m_fillOpt {Ω : Type} [Fintype Ω] (Pl : Play Ω Sm100.checks (Fin 26) (Fin 13) Unit Empty)
    (hSplit : ∀ x, (Pl.run x).Split) : ∀ x, FillOpt Sm100Measured.model (Pl.run x) := fun x =>
  fillOpt_of_zero Sm100Measured.model sm100m_a_nonneg (fun _ _ => rfl) rfl (Pl.run x) (hSplit x)

/-- The folded branch at π*, `p = 7/8` and `δ′ ≥ 2^-41` is at least `1 − 888/100000`. -/
theorem sm100m_folded (δ' : ℝ) (hδ' : (1 / 2 : ℝ) ^ 41 ≤ δ') :
    1 - 888 / 100000 ≤ 1 - Sm100Measured.model.epsSkip Sm100Measured.flat Sm100Measured.lamStar -
      Sm100Measured.model.epsCert Sm100Measured.flat Sm100Measured.lamStar -
      Sm100Measured.model.epsRoute Sm100Measured.flat Sm100Measured.lamStar -
      Sm100Measured.model.epsMand Sm100Measured.flat Sm100Measured.lamStar -
      Sm100Measured.model.epsJl Sm100Measured.flat Sm100Measured.lamStar (7 / 8) δ' Sm100.checks := by
  refine folded_of_lower Sm100Measured.model Sm100Measured.flat Sm100Measured.lamStar sm100m_isCert sm100m_wf rfl
    rfl Sm100.checks sm100_S δ' hδ' _ _ _ sm100m_wStar sm100m_phiStar ?_
  rw [sm100m_Nw, sm100m_bnd]
  exact_mod_cast sm100m_folded_q

/-- The unfolded branch's floor at π* is at least `1 − 888/100000`. -/
theorem sm100m_unfolded :
    1 - 888 / 100000 ≤ (Sm100Measured.model.Nw * Sm100Measured.model.wStar Sm100Measured.lamStar +
      (1 - Sm100Measured.model.muF) * Sm100Measured.model.Nw *
        Sm100Measured.model.phiStar Sm100Measured.lamStar) /
        Sm100Measured.model.bnd Sm100Measured.flat * (1 - 41 / 48000) := by
  refine unfolded_of_lower Sm100Measured.model Sm100Measured.flat Sm100Measured.lamStar sm100m_wf rfl _ _ _
    sm100m_wStar sm100m_phiStar ?_
  rw [sm100m_Nw, sm100m_bnd]
  exact_mod_cast sm100m_unfolded_q

end Accounting.SecurityProofs.TS.HeadlineDc

namespace Accounting.SecurityProofs.TS

open Accounting.TS Accounting.TS.Headline Accounting.Assumptions.TS Pouw.RelativeSecurity
  Accounting.SecurityProofs.TS.Headline Accounting.SecurityProofs.TS.HeadlineDc

/-- The B200's headline on DC-100-ri: `Whole.dIdealFloor_whole` at π* (`Sm100Measured.lamStar`, the issue row),
`p = 7/8`, `p₀ = ½`, `Λ = max(κ/ln(8/7), (κ + f)/ln 2)`, `S = 48,000` and `δ = 2^-40`, (I2)/(I4) and (I3) from A6
(`scalarRoute_half`), (I7) from the receipt split (`luck_of_receipt_half`), (I8) from its pieces
(`ProofsSoundOfPinned`), (I5) `SideUnits` and FILL-OPT from the data, and the terms bounded by
`Whole.headline_of_branches`: the folded branch by `sm100m_folded`, the unfolded one by `sm100m_unfolded`. -/
theorem Sm100MeasuredHeadline : Accounting.Guarantees.TS.Sm100MeasuredHeadline := by
  intro Ω _ Pl G Uf V nπ pf δbind δzk hδ hSplit h1 h6 hN hF hG hPinG hOp hRL hWL hOf hB hZK hPin hRF
  have hL := luck_of_receipt_half Pl.game G hG hPinG hOp hRL
  have hPS := ProofsSoundOfPinned Ω Sm100.checks nπ Pl pf Sm100Measured.model.Nw δbind δzk hOf hB hZK
    hPin hRF
  have hδlo : (1 / 2 : ℝ) ^ 41 ≤ (1 / 2) ^ 40 - (Sm100.checks : ℝ) * (δbind + nπ * δzk) := by
    have e : ((1 : ℝ) / 2) ^ 40 = 2 * (1 / 2) ^ 41 := by norm_num
    linarith
  have hδpos : 0 < (1 / 2 : ℝ) ^ 40 - (Sm100.checks : ℝ) * (δbind + nπ * δzk) :=
    lt_of_lt_of_le (by positivity) hδlo
  obtain ⟨hκ0, -⟩ := Whole.kappa_f_nonneg Sm100Measured.model Sm100Measured.flat Sm100Measured.lamStar
    sm100m_isCert sm100m_wf
  obtain ⟨hΛ, hΛ1, hΛ2⟩ := Whole.max_budget
    (f := (1 - Sm100Measured.model.muF) * Sm100Measured.model.Nw *
      Sm100Measured.model.phiStar Sm100Measured.lamStar / Sm100Measured.model.bnd Sm100Measured.flat) hκ0
    (Real.log_pos (by norm_num) : 0 < Real.log (1 / (7 / 8 : ℝ)))
    (Real.log_pos (by norm_num) : 0 < Real.log (1 / (1 / 2 : ℝ)))
  have hmain := Whole.dIdealFloor_whole Sm100Measured.model Sm100Measured.flat Pl G Uf
    Sm100Measured.lamStar (7 / 8) (1 / 2) _ ((1 / 2) ^ 40) δbind δzk nπ sm100m_isCert sm100m_wf
    (by decide) (by norm_num) (by norm_num) hΛ hΛ1 hΛ2 hδpos hSplit h1 (fun x => (h6 x).1)
    (fun x => scalarRoute_half Sm100Measured.model _ (hSplit x) (h6 x) (le_of_eq sm100m_q) (le_of_eq sm100m_qT)
      le_rfl)
    (fun _ u => u.elim) hN hF (sm100m_fillOpt Pl hSplit) hG hPinG hL hWL hPS
  refine le_trans (prob_mono ?_) hmain
  rintro x ⟨ha, hlt⟩
  exact ⟨ha, lt_of_lt_of_le hlt (Whole.headline_of_branches Sm100Measured.model Sm100Measured.flat
    Sm100Measured.lamStar sm100m_isCert sm100m_wf Sm100.checks sm100_S _ hδlo _
    (sm100m_folded _ hδlo) sm100m_unfolded)⟩

end Accounting.SecurityProofs.TS
