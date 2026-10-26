import Proofs.Accounting.HeadlineDc.Branches
import Definitions.Accounting.Guarantees.HeadlineDc

/-!
# The B200's headline at the data sheet

`Sm100SheetHeadline` is `Whole.sheetDIdealFloor_whole` at DS-100 and `dither-2s-licpw-redux-ldtm0-l32`
(`Definitions.Accounting.HeadlineDc.Sm100`), at the data's certificate `π* = Sm100.lamStar` (DS_issue
`666666667/10^9`, DS_alu `333333333/10^9`), `p = 7/8`, `p₀ = ½` and `Λ = max(κ/ln(8/7), (κ + f)/ln 2)`:

* the data, by `decide`: issue is ♭'s largest facet (`sm100_max`), so `B_♭` is its issue load (`sm100_bnd`);
  every writer kind's `ω ≤ 32768` (`sm100_omega`), so each word route costs at least `1/32768` on issue and
  `w* ≥ λ_issue/32768` (`sm100_wStar`); each of the 14 fold routes costs at least the cheapest one's price on the
  two facets (`sm100_fold`, `sm100_phiStar`);
* the folded branch from those bounds (`folded_of_lower`): below `783/100000` by `decide` (`sm100_folded_q`);
* the unfolded branch (`unfolded_of_lower`): below `783/100000` by `decide` (`sm100_unfolded_q`);
* FILL-OPT holds since there is no filler (`fillOpt_of_zero`), and (I7) at `7/8` is the receipt split's
  (`luck_of_receipt_half`).
-/

namespace Accounting.SecurityProofs.TS.HeadlineDc

open Accounting.TS Accounting.TS.Headline Accounting.Assumptions.TS Pouw.RelativeSecurity Accounting.SecurityProofs.TS
  Accounting.SecurityProofs.TS.Headline

/-- ♭'s counts on DS-100, its four lists in `labelled`'s order. -/
def flat100 : Sparse := Sm100.flatWords ++ Sm100.flatFold ++ Sm100.flatFill ++ Sm100.flatSkip

/-- `B_♭` on DS-100 as a rational: ♭'s load on issue. -/
def B100 : ℚ := dot (sheetCoefQ Sm100.ds100 .issue) 52 flat100

/-- π*'s weights as rationals: DS_issue and DS_alu. -/
def li100 : ℚ := 666666667 / 1000000000

def la100 : ℚ := 333333333 / 1000000000

/-- Fold route `i`'s price at π* as a rational, on DS_issue and DS_alu. -/
def price100 (i : ℕ) : ℚ :=
  li100 * dot (sheetCoefQ Sm100.ds100 .issue) 52 (Sm100.d2s.folds.getD i []) +
    la100 * dot (sheetCoefQ Sm100.ds100 (.pipe .alu)) 52 (Sm100.d2s.folds.getD i [])

/-- The cheapest fold route's price at π*. -/
def phi100 : ℚ := ((List.range 14).map price100).foldr min (price100 0)

theorem sm100_max (r : SheetRow Sm100.Pipe) : dot (sheetCoefQ Sm100.ds100 r) 52 flat100 ≤ B100 := by
  cases r with
  | issue => decide +kernel
  | tensor => decide +kernel
  | pipe p => cases p <;> decide +kernel
  | smem => decide +kernel

theorem sm100_Bpos : (0 : ℚ) < B100 := by decide +kernel

theorem sm100_omega :
    ∀ i < 27, 0 < (Sm100.d2s.writers.getD i (0, 0)).2 ∧ (Sm100.d2s.writers.getD i (0, 0)).2 ≤ 32768 := by
  decide +kernel

theorem sm100_fold : ∀ i < 14, phi100 ≤
    li100 * dot (sheetCoefQ Sm100.ds100 .issue) 52 (Sm100.d2s.folds.getD i []) +
      la100 * dot (sheetCoefQ Sm100.ds100 (.pipe .alu)) 52 (Sm100.d2s.folds.getD i []) := by
  decide +kernel

theorem sm100_lam_q : 0 ≤ li100 ∧ 0 ≤ la100 ∧ li100 + la100 = 1 := by decide +kernel

theorem sm100_folded_q :
    (1 - 783 / 100000 : ℚ) * B100 ≤ 32768 * (li100 * (1 / 32768)) * (1 - 41 * (6931471808 / 10000000000) / 6000) +
      32768 * phi100 := by
  decide +kernel

theorem sm100_unfolded_q :
    (1 - 783 / 100000 : ℚ) * B100 ≤ (32768 * (li100 * (1 / 32768)) + 32768 * phi100) * (1 - 41 / 48000) := by
  decide +kernel

theorem sm100_hG : ∀ r k, Sm100.model.G.a r k = (sheetCoefQ Sm100.ds100 r k.val : ℝ) := sheet_aQ Sm100.ds100 52

theorem sm100_a_nonneg : ∀ r k, 0 ≤ Sm100.model.G.a r k := fun r k => by
  rw [sm100_hG]
  exact Rat.cast_nonneg.mpr (sheetCoefQ_nonneg _ (fun p => by cases p <;> decide +kernel) _ _)

theorem sm100_bnd : Sm100.model.bnd Sm100.flat = (B100 : ℝ) := by
  unfold Model.bnd
  rw [show Sm100.flat.n = Sparse.counts flat100 from
    labelled_n 52 Sm100.flatWords Sm100.flatFold Sm100.flatFill Sm100.flatSkip]
  exact bound_eq _ (sheetCoefQ Sm100.ds100) sm100_hG flat100 .issue sm100_max

theorem sm100_Nw : Sm100.model.Nw = 32768 := by
  show ((32768 : ℕ) : ℝ) = 32768
  norm_num

theorem sm100_q : Sm100.model.q = 1 / 2 := by
  show ((1 / 2 : ℚ) : ℝ) = 1 / 2
  norm_num

theorem sm100_qT : Sm100.model.qT = 1 / 2 := by
  show ((1 / 2 : ℚ) : ℝ) = 1 / 2
  norm_num

theorem sm100_S : ((Sm100.checks : ℕ) : ℝ) = 48000 := by
  show ((48000 : ℕ) : ℝ) = 48000
  norm_num

theorem sm100_li : Sm100.lamStar .issue = (li100 : ℝ) := by
  show (666666667 / 1000000000 : ℝ) = ((666666667 / 1000000000 : ℚ) : ℝ)
  norm_num

theorem sm100_la : Sm100.lamStar (.pipe .alu) = (la100 : ℝ) := by
  show (333333333 / 1000000000 : ℝ) = ((333333333 / 1000000000 : ℚ) : ℝ)
  norm_num

theorem sm100_lamStar_zero : ∀ r : SheetRow Sm100.Pipe, r ≠ .issue ∧ r ≠ .pipe .alu → Sm100.lamStar r = 0
  | .issue, h => absurd rfl h.1
  | .tensor, _ => rfl
  | .pipe .alu, h => absurd rfl h.2
  | .pipe .imad, _ => rfl
  | .pipe .shfl, _ => rfl
  | .pipe .uniform, _ => rfl
  | .smem, _ => rfl

/-- π* is a certificate: its two weights are `≥ 0` and sum to 1. -/
theorem sm100_isCert : IsCert Sm100.lamStar := by
  obtain ⟨hli, hla, hsum⟩ := sm100_lam_q
  refine ⟨fun r => ?_, ?_⟩
  · by_cases h1 : r = .issue
    · rw [h1, sm100_li]
      exact Rat.cast_nonneg.mpr hli
    by_cases h2 : r = .pipe .alu
    · rw [h2, sm100_la]
      exact Rat.cast_nonneg.mpr hla
    exact le_of_eq (sm100_lamStar_zero r ⟨h1, h2⟩).symm
  · rw [Fintype.sum_eq_add SheetRow.issue (SheetRow.pipe Sm100.Pipe.alu) (by decide) sm100_lamStar_zero,
      sm100_li, sm100_la]
    exact_mod_cast hsum

theorem sm100_wf : Sm100.model.Wf Sm100.flat := by
  refine ⟨sm100_a_nonneg, ?_, ?_, ?_, fun t => ?_, ?_, ?_, le_rfl, fun u => u.elim, fun φ r => ?_, fun _ _ => le_rfl,
    le_rfl, ?_, le_rfl, ?_⟩
  · show (0 : ℝ) < ((1965 : ℕ) : ℝ)
    norm_num
  · rw [sm100_Nw]
    norm_num
  · rw [sm100_bnd]
    exact_mod_cast sm100_Bpos
  · show (0 : ℝ) < ((Sm100.d2s.writers.getD t.val (0, 0)).2 : ℝ)
    exact_mod_cast (sm100_omega t.val t.isLt).1
  · rw [sm100_q]
    norm_num
  · rw [sm100_qT]
    norm_num
  · show 0 ≤ Sm100.model.G.load (Sparse.counts (Sm100.d2s.folds.getD φ.val [])) r
    exact load_counts_nonneg _ sm100_a_nonneg _ _
  · show (0 : ℝ) ≤ 1
    norm_num
  · show (0 : ℝ) ≤ 1
    norm_num

theorem sm100_vecMin : 1 ≤ Sm100.model.vecMin .issue := by
  have : Nonempty ↥Sm100.model.vec :=
    ⟨⟨idx 52 7, List.mem_toFinset.mpr (List.mem_map.mpr ⟨7, by decide, rfl⟩)⟩⟩
  unfold Model.vecMin
  refine le_ciInf fun k => ?_
  rw [sm100_hG]
  simp [sheetCoefQ]

/-- Every word route costs at least `1/32768` on issue: a writer kind's instruction `1/ω` with `ω ≤ 32768`, a vector
or table word `½/32` an instruction. -/
theorem sm100_demand_issue : ∀ hd, (1 : ℝ) / 32768 ≤ Sm100.model.demand hd .issue := by
  intro h
  cases h with
  | writer t =>
    show (1 : ℝ) / 32768 ≤ 1 / ((Sm100.d2s.writers.getD t.val (0, 0)).2 : ℝ)
    exact one_div_le_one_div_of_le (by exact_mod_cast (sm100_omega t.val t.isLt).1)
      (by exact_mod_cast (sm100_omega t.val t.isLt).2)
  | vector =>
    show (1 : ℝ) / 32768 ≤ Sm100.model.q / 32 * Sm100.model.vecMin .issue
    rw [sm100_q]
    calc (1 : ℝ) / 32768 ≤ 1 / 2 / 32 * 1 := by norm_num
      _ ≤ 1 / 2 / 32 * Sm100.model.vecMin .issue := mul_le_mul_of_nonneg_left sm100_vecMin (by norm_num)
  | table l =>
    show (1 : ℝ) / 32768 ≤ max (Sm100.model.qT / 32 * Sm100.model.vecMin .issue)
      (Sm100.model.bT * Sm100.model.G.a .issue (Sm100.model.bytes l))
    refine le_trans ?_ (le_max_left _ _)
    rw [sm100_qT]
    calc (1 : ℝ) / 32768 ≤ 1 / 2 / 32 * 1 := by norm_num
      _ ≤ 1 / 2 / 32 * Sm100.model.vecMin .issue := mul_le_mul_of_nonneg_left sm100_vecMin (by norm_num)
  | side u => exact u.elim

/-- `w* ≥ λ_issue/32768` at π*. -/
theorem sm100_wStar : (li100 : ℝ) * (1 / 32768) ≤ Sm100.model.wStar Sm100.lamStar := by
  have h := wStar_ge Sm100.model Sm100.flat Sm100.lamStar sm100_isCert sm100_wf .issue _ sm100_demand_issue
  rwa [sm100_li] at h

/-- `φ* ≥` the cheapest fold route's price at π*. -/
theorem sm100_phiStar : (phi100 : ℝ) ≤ Sm100.model.phiStar Sm100.lamStar := by
  refine phiStar_ge_pair Sm100.model Sm100.flat Sm100.lamStar sm100_isCert sm100_wf .issue (.pipe .alu)
    (by decide) _ fun φ => ?_
  show (phi100 : ℝ) ≤
    Sm100.lamStar .issue * Sm100.model.G.load (Sparse.counts (Sm100.d2s.folds.getD φ.val [])) .issue +
      Sm100.lamStar (.pipe .alu) * Sm100.model.G.load (Sparse.counts (Sm100.d2s.folds.getD φ.val [])) (.pipe .alu)
  rw [load_counts _ (sheetCoefQ Sm100.ds100) sm100_hG, load_counts _ (sheetCoefQ Sm100.ds100) sm100_hG, sm100_li,
    sm100_la]
  have h := (Rat.cast_le (K := ℝ)).mpr (sm100_fold φ.val φ.isLt)
  push_cast at h
  linarith

theorem sm100_fillOpt {Ω : Type} [Fintype Ω] (Pl : Play Ω Sm100.checks (Fin 52) (Fin 27) Unit Empty)
    (hSplit : ∀ x, (Pl.run x).Split) : ∀ x, FillOpt Sm100.model (Pl.run x) := fun x =>
  fillOpt_of_zero Sm100.model sm100_a_nonneg (fun _ _ => rfl) rfl (Pl.run x) (hSplit x)

/-- The folded branch at π*, `p = 7/8` and `δ′ ≥ 2^-41` is at least `1 − 783/100000`. -/
theorem sm100_folded (δ' : ℝ) (hδ' : (1 / 2 : ℝ) ^ 41 ≤ δ') :
    1 - 783 / 100000 ≤ 1 - Sm100.model.epsSkip Sm100.flat Sm100.lamStar -
      Sm100.model.epsCert Sm100.flat Sm100.lamStar - Sm100.model.epsRoute Sm100.flat Sm100.lamStar -
      Sm100.model.epsMand Sm100.flat Sm100.lamStar -
      Sm100.model.epsJl Sm100.flat Sm100.lamStar (7 / 8) δ' Sm100.checks := by
  refine folded_of_lower Sm100.model Sm100.flat Sm100.lamStar sm100_isCert sm100_wf rfl rfl Sm100.checks sm100_S
    δ' hδ' _ _ _ sm100_wStar sm100_phiStar ?_
  have hB : (0 : ℝ) < (B100 : ℝ) := by exact_mod_cast sm100_Bpos
  have h := (Rat.cast_le (K := ℝ)).mpr sm100_folded_q
  push_cast at h
  rw [sm100_Nw, sm100_bnd, le_div_iff₀ hB]
  linarith

/-- The unfolded branch's floor at π* is at least `1 − 783/100000`. -/
theorem sm100_unfolded :
    1 - 783 / 100000 ≤ (Sm100.model.Nw * Sm100.model.wStar Sm100.lamStar +
      (1 - Sm100.model.muF) * Sm100.model.Nw * Sm100.model.phiStar Sm100.lamStar) /
        Sm100.model.bnd Sm100.flat * (1 - 41 / 48000) := by
  refine unfolded_of_lower Sm100.model Sm100.flat Sm100.lamStar sm100_wf rfl _ _ _ sm100_wStar sm100_phiStar ?_
  have hB : (0 : ℝ) < (B100 : ℝ) := by exact_mod_cast sm100_Bpos
  have h := (Rat.cast_le (K := ℝ)).mpr sm100_unfolded_q
  push_cast at h
  rw [sm100_Nw, sm100_bnd, div_mul_eq_mul_div, le_div_iff₀ hB]
  linarith

end Accounting.SecurityProofs.TS.HeadlineDc

namespace Accounting.SecurityProofs.TS

open Accounting.TS Accounting.TS.Headline Accounting.Assumptions.TS Pouw.RelativeSecurity
  Accounting.SecurityProofs.TS.Headline Accounting.SecurityProofs.TS.HeadlineDc

/-- The B200's headline at the data sheet: `Whole.sheetDIdealFloor_whole` at π* (`Sm100.lamStar`, DS_issue ⅔ +
DS_alu ⅓), `p = 7/8`, `p₀ = ½`, `Λ = max(κ/ln(8/7), (κ + f)/ln 2)`, `S = 48,000` and `δ = 2^-40`, (I7) from the
receipt split (`luck_of_receipt_half`), (I8) from its pieces (`ProofsSoundOfPinned`, binding at `√δ_fork` by
`Whole.moved_le_sqrt`), FILL-OPT from the data, and the terms bounded by `Whole.headline_of_branches`: the folded
branch by `sm100_folded`, the unfolded one by `sm100_unfolded`. -/
theorem Sm100SheetHeadline : Accounting.Guarantees.TS.Sm100SheetHeadline := by
  intro Ω _ Pl G Uf V nπ pf δfork δzk hδ hSplit h1 h2 h3 h4 h5 h6 h7 hF hG hPinG hOp hRL hWL hOf hB hZK hPin hRF
  set δbind := Real.sqrt δfork
  have hL := luck_of_receipt_half Pl.game G hG hPinG hOp hRL
  have hPS :=
    ProofsSoundOfPinned Ω Sm100.checks nπ Pl pf Sm100.model.Nw δbind δzk hOf
    (Whole.moved_le_sqrt pf δfork hB) hZK hPin hRF
  have hδlo : (1 / 2 : ℝ) ^ 41 ≤ (1 / 2) ^ 40 - (Sm100.checks : ℝ) * (δbind + nπ * δzk) := by
    have e : ((1 : ℝ) / 2) ^ 40 = 2 * (1 / 2) ^ 41 := by norm_num
    linarith
  have hδpos : 0 < (1 / 2 : ℝ) ^ 40 - (Sm100.checks : ℝ) * (δbind + nπ * δzk) :=
    lt_of_lt_of_le (by positivity) hδlo
  obtain ⟨hκ0, -⟩ := Whole.kappa_f_nonneg Sm100.model Sm100.flat Sm100.lamStar sm100_isCert sm100_wf
  obtain ⟨hΛ, hΛ1, hΛ2⟩ := Whole.max_budget
    (f := (1 - Sm100.model.muF) * Sm100.model.Nw * Sm100.model.phiStar Sm100.lamStar / Sm100.model.bnd Sm100.flat)
    hκ0 (Real.log_pos (by norm_num) : 0 < Real.log (1 / (7 / 8 : ℝ)))
    (Real.log_pos (by norm_num) : 0 < Real.log (1 / (1 / 2 : ℝ)))
  have hmain := Whole.sheetDIdealFloor_whole Sm100.sheet Sm100.model Sm100.flat Pl G Uf
    Sm100.lamStar (7 / 8) (1 / 2) _ ((1 / 2) ^ 40) δbind δzk nπ rfl sm100_isCert sm100_wf
    (by decide) (by norm_num) (by norm_num) hΛ hΛ1 hΛ2 hδpos hSplit (le_of_eq sm100_q) (le_of_eq sm100_qT) le_rfl
    h1 h2 h3 h4 h5 h6 h7 hF (sm100_fillOpt Pl hSplit) hG hPinG hL hWL hPS
  refine le_trans (prob_mono ?_) hmain
  rintro x ⟨ha, hlt⟩
  exact ⟨ha, lt_of_lt_of_le hlt (Whole.headline_of_branches Sm100.model Sm100.flat Sm100.lamStar
    sm100_isCert sm100_wf Sm100.checks sm100_S _ hδlo _ (sm100_folded _ hδlo) sm100_unfolded)⟩

end Accounting.SecurityProofs.TS
