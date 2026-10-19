import Proofs.Accounting.HeadlineDc.Branches
import Models.Accounting.Guarantees.HeadlineDc

/-!
# The H200's headline at the data sheet

`Sm90SheetHeadline` is `Whole.sheetDIdealFloor_whole` at DS-90 and `dither-2s-licpw-redux-p1-l32`
(`Definitions.Accounting.HeadlineDc.Sm90`), at the data's certificate `π* = Sm90.lamStar` (DS_issue
`1/2`, DS_alu `1/2`), `p = 7/8`, `p₀ = ½` and `Λ = max(κ/ln(8/7), (κ + f)/ln 2)`:

* the data, by `decide`: issue is ♭'s largest facet (`sm90_max`), so `B_♭` is its issue load (`sm90_bnd`);
  every writer kind's `ω ≤ 4096` (`sm90_omega`), so each word route costs at least `1/4096` on issue and
  `w* ≥ λ_issue/4096` (`sm90_wStar`); each of the four fold routes (the design's fold, cp-seg4, cp-seg4u and the
  uniform one) costs at least the cheapest one's price on the two facets (`sm90_fold`, `sm90_phiStar`);
* the folded branch from those bounds (`folded_of_lower`): below `933/100000` by `decide` (`sm90_folded_q`);
* the unfolded branch (`unfolded_of_lower`): below `933/100000` by `decide` (`sm90_unfolded_q`);
* FILL-OPT holds since there is no filler (`fillOpt_of_zero`), and (I7) at `7/8` is the receipt split's
  (`luck_of_receipt_half`).
-/

namespace Accounting.SecurityProofs.TS.HeadlineDc

open Accounting.TS Accounting.TS.Headline Accounting.Assumptions.TS Pouw.RelativeSecurity Accounting.SecurityProofs.TS
  Accounting.SecurityProofs.TS.Headline

/-- ♭'s counts on DS-90, its four lists in `labelled`'s order. -/
def flat90 : Sparse := Sm90.flatWords ++ Sm90.flatFold ++ Sm90.flatFill ++ Sm90.flatSkip

/-- `B_♭` on DS-90 as a rational: ♭'s load on issue. -/
def B90 : ℚ := dot (sheetCoef Sm90.ds90 .issue) 29 flat90

/-- π*'s weights as rationals: DS_issue and DS_alu. -/
def li90 : ℚ := 1 / 2

def la90 : ℚ := 1 / 2

/-- Fold route `i`'s price at π* as a rational, on DS_issue and DS_alu. -/
def price90 (i : ℕ) : ℚ :=
  li90 * dot (sheetCoef Sm90.ds90 .issue) 29 (Sm90.d2s.folds.getD i []) +
    la90 * dot (sheetCoef Sm90.ds90 (.pipe .alu)) 29 (Sm90.d2s.folds.getD i [])

/-- The cheapest fold route's price at π*. -/
def phi90 : ℚ := min (min (price90 0) (price90 1)) (min (price90 2) (price90 3))

theorem sm90_max (r : SheetRow Sm90.Pipe) : dot (sheetCoef Sm90.ds90 r) 29 flat90 ≤ B90 := by
  cases r with
  | issue => decide +kernel
  | tensor => decide +kernel
  | pipe p => cases p <;> decide +kernel
  | smem => decide +kernel

theorem sm90_Bpos : (0 : ℚ) < B90 := by decide +kernel

theorem sm90_omega :
    ∀ i < 24, 0 < (Sm90.d2s.writers.getD i (0, 0)).2 ∧ (Sm90.d2s.writers.getD i (0, 0)).2 ≤ 4096 := by
  decide +kernel

theorem sm90_fold : ∀ i < 4, phi90 ≤
    li90 * dot (sheetCoef Sm90.ds90 .issue) 29 (Sm90.d2s.folds.getD i []) +
      la90 * dot (sheetCoef Sm90.ds90 (.pipe .alu)) 29 (Sm90.d2s.folds.getD i []) := by
  decide +kernel

theorem sm90_lam_q : 0 ≤ li90 ∧ 0 ≤ la90 ∧ li90 + la90 = 1 := by decide +kernel

theorem sm90_folded_q :
    (1 - 933 / 100000 : ℚ) * B90 ≤ 4096 * (li90 * (1 / 4096)) * (1 - 41 * (6931471808 / 10000000000) / 6000) +
      4096 * phi90 := by
  decide +kernel

theorem sm90_unfolded_q :
    (1 - 933 / 100000 : ℚ) * B90 ≤ (4096 * (li90 * (1 / 4096)) + 4096 * phi90) * (1 - 41 / 48000) := by
  decide +kernel

theorem sm90_hG : ∀ r k, Sm90.model.G.a r k = (sheetCoef Sm90.ds90 r k.val : ℝ) := sheet_a Sm90.ds90 29

theorem sm90_a_nonneg : ∀ r k, 0 ≤ Sm90.model.G.a r k := fun r k => by
  rw [sm90_hG]
  exact Rat.cast_nonneg.mpr (sheetCoef_nonneg _ _ _)

theorem sm90_bnd : Sm90.model.bnd Sm90.flat = (B90 : ℝ) := by
  unfold Model.bnd
  rw [show Sm90.flat.n = Sparse.counts flat90 from
    labelled_n 29 Sm90.flatWords Sm90.flatFold Sm90.flatFill Sm90.flatSkip]
  exact bound_eq _ (sheetCoef Sm90.ds90) sm90_hG flat90 .issue sm90_max

theorem sm90_Nw : Sm90.model.Nw = 4096 := by
  show ((4096 : ℕ) : ℝ) = 4096
  norm_num

theorem sm90_q : Sm90.model.q = 1 / 2 := by
  show ((1 / 2 : ℚ) : ℝ) = 1 / 2
  norm_num

theorem sm90_qT : Sm90.model.qT = 1 / 2 := by
  show ((1 / 2 : ℚ) : ℝ) = 1 / 2
  norm_num

theorem sm90_S : ((Sm90.checks : ℕ) : ℝ) = 48000 := by
  show ((48000 : ℕ) : ℝ) = 48000
  norm_num

theorem sm90_li : Sm90.lamStar .issue = (li90 : ℝ) := by
  show (1 / 2 : ℝ) = ((1 / 2 : ℚ) : ℝ)
  norm_num

theorem sm90_la : Sm90.lamStar (.pipe .alu) = (la90 : ℝ) := by
  show (1 / 2 : ℝ) = ((1 / 2 : ℚ) : ℝ)
  norm_num

theorem sm90_lamStar_zero : ∀ r : SheetRow Sm90.Pipe, r ≠ .issue ∧ r ≠ .pipe .alu → Sm90.lamStar r = 0
  | .issue, h => absurd rfl h.1
  | .tensor, _ => rfl
  | .pipe .alu, h => absurd rfl h.2
  | .pipe .imad, _ => rfl
  | .pipe .shfl, _ => rfl
  | .smem, _ => rfl

/-- π* is a certificate: its two weights are `≥ 0` and sum to 1. -/
theorem sm90_isCert : IsCert Sm90.lamStar := by
  obtain ⟨hli, hla, hsum⟩ := sm90_lam_q
  refine ⟨fun r => ?_, ?_⟩
  · by_cases h1 : r = .issue
    · rw [h1, sm90_li]
      exact Rat.cast_nonneg.mpr hli
    by_cases h2 : r = .pipe .alu
    · rw [h2, sm90_la]
      exact Rat.cast_nonneg.mpr hla
    exact le_of_eq (sm90_lamStar_zero r ⟨h1, h2⟩).symm
  · rw [Fintype.sum_eq_add SheetRow.issue (SheetRow.pipe Sm90.Pipe.alu) (by decide) sm90_lamStar_zero,
      sm90_li, sm90_la]
    exact_mod_cast hsum

theorem sm90_wf : Sm90.model.Wf Sm90.flat := by
  refine ⟨sm90_a_nonneg, ?_, ?_, ?_, fun t => ?_, ?_, ?_, le_rfl, fun u => u.elim, fun φ r => ?_, fun _ _ => le_rfl,
    le_rfl, ?_, le_rfl, ?_⟩
  · show (0 : ℝ) < ((1980 : ℕ) : ℝ)
    norm_num
  · rw [sm90_Nw]
    norm_num
  · rw [sm90_bnd]
    exact_mod_cast sm90_Bpos
  · show (0 : ℝ) < ((Sm90.d2s.writers.getD t.val (0, 0)).2 : ℝ)
    exact_mod_cast (sm90_omega t.val t.isLt).1
  · rw [sm90_q]
    norm_num
  · rw [sm90_qT]
    norm_num
  · show 0 ≤ Sm90.model.G.load (Sparse.counts (Sm90.d2s.folds.getD φ.val [])) r
    exact load_counts_nonneg _ sm90_a_nonneg _ _
  · show (0 : ℝ) ≤ 1
    norm_num
  · show (0 : ℝ) ≤ 1
    norm_num

theorem sm90_vecMin : 1 ≤ Sm90.model.vecMin .issue := by
  have : Nonempty ↥Sm90.model.vec :=
    ⟨⟨idx 29 3, List.mem_toFinset.mpr (List.mem_map.mpr ⟨3, by decide, rfl⟩)⟩⟩
  unfold Model.vecMin
  refine le_ciInf fun k => ?_
  rw [sm90_hG]
  simp [sheetCoef]

/-- Every word route costs at least `1/4096` on issue: a writer kind's instruction `1/ω` with `ω ≤ 4096`, a vector
or table word `½/32` an instruction. -/
theorem sm90_demand_issue : ∀ hd, (1 : ℝ) / 4096 ≤ Sm90.model.demand hd .issue := by
  intro h
  cases h with
  | writer t =>
    show (1 : ℝ) / 4096 ≤ 1 / ((Sm90.d2s.writers.getD t.val (0, 0)).2 : ℝ)
    exact one_div_le_one_div_of_le (by exact_mod_cast (sm90_omega t.val t.isLt).1)
      (by exact_mod_cast (sm90_omega t.val t.isLt).2)
  | vector =>
    show (1 : ℝ) / 4096 ≤ Sm90.model.q / 32 * Sm90.model.vecMin .issue
    rw [sm90_q]
    calc (1 : ℝ) / 4096 ≤ 1 / 2 / 32 * 1 := by norm_num
      _ ≤ 1 / 2 / 32 * Sm90.model.vecMin .issue := mul_le_mul_of_nonneg_left sm90_vecMin (by norm_num)
  | table l =>
    show (1 : ℝ) / 4096 ≤ max (Sm90.model.qT / 32 * Sm90.model.vecMin .issue)
      (Sm90.model.bT * Sm90.model.G.a .issue (Sm90.model.bytes l))
    refine le_trans ?_ (le_max_left _ _)
    rw [sm90_qT]
    calc (1 : ℝ) / 4096 ≤ 1 / 2 / 32 * 1 := by norm_num
      _ ≤ 1 / 2 / 32 * Sm90.model.vecMin .issue := mul_le_mul_of_nonneg_left sm90_vecMin (by norm_num)
  | side u => exact u.elim

/-- `w* ≥ λ_issue/4096` at π*. -/
theorem sm90_wStar : (li90 : ℝ) * (1 / 4096) ≤ Sm90.model.wStar Sm90.lamStar := by
  have h := wStar_ge Sm90.model Sm90.flat Sm90.lamStar sm90_isCert sm90_wf .issue _ sm90_demand_issue
  rwa [sm90_li] at h

/-- `φ* ≥` the cheapest fold route's price at π*. -/
theorem sm90_phiStar : (phi90 : ℝ) ≤ Sm90.model.phiStar Sm90.lamStar := by
  refine phiStar_ge_pair Sm90.model Sm90.flat Sm90.lamStar sm90_isCert sm90_wf .issue (.pipe .alu)
    (by decide) _ fun φ => ?_
  show (phi90 : ℝ) ≤
    Sm90.lamStar .issue * Sm90.model.G.load (Sparse.counts (Sm90.d2s.folds.getD φ.val [])) .issue +
      Sm90.lamStar (.pipe .alu) * Sm90.model.G.load (Sparse.counts (Sm90.d2s.folds.getD φ.val [])) (.pipe .alu)
  rw [load_counts _ (sheetCoef Sm90.ds90) sm90_hG, load_counts _ (sheetCoef Sm90.ds90) sm90_hG, sm90_li,
    sm90_la]
  have h := (Rat.cast_le (K := ℝ)).mpr (sm90_fold φ.val φ.isLt)
  push_cast at h
  linarith

theorem sm90_fillOpt {Ω : Type} [Fintype Ω] (Pl : Play Ω Sm90.checks (Fin 29) (Fin 24) Unit Empty)
    (hSplit : ∀ x, (Pl.run x).Split) : ∀ x, FillOpt Sm90.model (Pl.run x) := fun x =>
  fillOpt_of_zero Sm90.model sm90_a_nonneg (fun _ _ => rfl) rfl (Pl.run x) (hSplit x)

/-- The folded branch at π*, `p = 7/8` and `δ′ ≥ 2^-41` is at least `1 − 933/100000`. -/
theorem sm90_folded (δ' : ℝ) (hδ' : (1 / 2 : ℝ) ^ 41 ≤ δ') :
    1 - 933 / 100000 ≤ 1 - Sm90.model.epsSkip Sm90.flat Sm90.lamStar -
      Sm90.model.epsCert Sm90.flat Sm90.lamStar - Sm90.model.epsRoute Sm90.flat Sm90.lamStar -
      Sm90.model.epsMand Sm90.flat Sm90.lamStar -
      Sm90.model.epsJl Sm90.flat Sm90.lamStar (7 / 8) δ' Sm90.checks := by
  refine folded_of_lower Sm90.model Sm90.flat Sm90.lamStar sm90_isCert sm90_wf rfl rfl Sm90.checks sm90_S
    δ' hδ' _ _ _ sm90_wStar sm90_phiStar ?_
  have hB : (0 : ℝ) < (B90 : ℝ) := by exact_mod_cast sm90_Bpos
  have h := (Rat.cast_le (K := ℝ)).mpr sm90_folded_q
  push_cast at h
  rw [sm90_Nw, sm90_bnd, le_div_iff₀ hB]
  linarith

/-- The unfolded branch's floor at π* is at least `1 − 933/100000`. -/
theorem sm90_unfolded :
    1 - 933 / 100000 ≤ (Sm90.model.Nw * Sm90.model.wStar Sm90.lamStar +
      (1 - Sm90.model.muF) * Sm90.model.Nw * Sm90.model.phiStar Sm90.lamStar) /
        Sm90.model.bnd Sm90.flat * (1 - 41 / 48000) := by
  refine unfolded_of_lower Sm90.model Sm90.flat Sm90.lamStar sm90_wf rfl _ _ _ sm90_wStar sm90_phiStar ?_
  have hB : (0 : ℝ) < (B90 : ℝ) := by exact_mod_cast sm90_Bpos
  have h := (Rat.cast_le (K := ℝ)).mpr sm90_unfolded_q
  push_cast at h
  rw [sm90_Nw, sm90_bnd, div_mul_eq_mul_div, le_div_iff₀ hB]
  linarith

end Accounting.SecurityProofs.TS.HeadlineDc

namespace Accounting.SecurityProofs.TS

open Accounting.TS Accounting.TS.Headline Accounting.Assumptions.TS Pouw.RelativeSecurity
  Accounting.SecurityProofs.TS.Headline Accounting.SecurityProofs.TS.HeadlineDc

/-- The H200's headline at the data sheet: `Whole.sheetDIdealFloor_whole` at π* (`Sm90.lamStar`), `p = 7/8`,
`p₀ = ½`, `Λ = max(κ/ln(8/7), (κ + f)/ln 2)`, `S = 48,000` and `δ = 2^-40`, (I7) from the receipt split
(`luck_of_receipt_half`), (I8) from its pieces (`ProofsSoundOfPinned`), FILL-OPT from the data, and the terms bounded
by `Whole.headline_of_branches`: the folded branch by `sm90_folded`, the unfolded one by `sm90_unfolded`. -/
theorem Sm90SheetHeadline : Accounting.Guarantees.TS.Sm90SheetHeadline := by
  intro Ω _ Pl G Uf V nπ pf δbind δzk hδ hSplit h1 h2 h3 h4 h5 h6 h7 hF hG hPinG hOp hRL hWL hOf hB hZK hPin hRF
  have hL := luck_of_receipt_half Pl.game G hG hPinG hOp hRL
  have hPS :=
    ProofsSoundOfPinned Ω Sm90.checks nπ Pl pf Sm90.model.Nw δbind δzk hOf hB hZK hPin hRF
  have hδlo : (1 / 2 : ℝ) ^ 41 ≤ (1 / 2) ^ 40 - (Sm90.checks : ℝ) * (δbind + nπ * δzk) := by
    have e : ((1 : ℝ) / 2) ^ 40 = 2 * (1 / 2) ^ 41 := by norm_num
    linarith
  have hδpos : 0 < (1 / 2 : ℝ) ^ 40 - (Sm90.checks : ℝ) * (δbind + nπ * δzk) :=
    lt_of_lt_of_le (by positivity) hδlo
  obtain ⟨hκ0, -⟩ := Whole.kappa_f_nonneg Sm90.model Sm90.flat Sm90.lamStar sm90_isCert sm90_wf
  obtain ⟨hΛ, hΛ1, hΛ2⟩ := Whole.max_budget
    (f := (1 - Sm90.model.muF) * Sm90.model.Nw * Sm90.model.phiStar Sm90.lamStar / Sm90.model.bnd Sm90.flat)
    hκ0 (Real.log_pos (by norm_num) : 0 < Real.log (1 / (7 / 8 : ℝ)))
    (Real.log_pos (by norm_num) : 0 < Real.log (1 / (1 / 2 : ℝ)))
  have hmain := Whole.sheetDIdealFloor_whole Sm90.sheet Sm90.model Sm90.flat Pl G Uf
    Sm90.lamStar (7 / 8) (1 / 2) _ ((1 / 2) ^ 40) δbind δzk nπ rfl sm90_isCert sm90_wf
    (by decide) (by norm_num) (by norm_num) hΛ hΛ1 hΛ2 hδpos hSplit (le_of_eq sm90_q) (le_of_eq sm90_qT) le_rfl
    h1 h2 h3 h4 h5 h6 h7 hF (sm90_fillOpt Pl hSplit) hG hPinG hL hWL hPS
  refine le_trans (prob_mono ?_) hmain
  rintro x ⟨ha, hlt⟩
  exact ⟨ha, lt_of_lt_of_le hlt (Whole.headline_of_branches Sm90.model Sm90.flat Sm90.lamStar
    sm90_isCert sm90_wf Sm90.checks sm90_S _ hδlo _ (sm90_folded _ hδlo) sm90_unfolded)⟩

end Accounting.SecurityProofs.TS
