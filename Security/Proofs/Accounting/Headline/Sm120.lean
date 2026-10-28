import Proofs.Accounting.Headline.Encoding
import Proofs.Accounting.Headline.Luck
import Proofs.Accounting.Headline.Whole
import Definitions.Accounting.Guarantees.Headline

/-!
# sm_120's headline at the data sheet

`Sm120SheetHeadline` is `Whole.sheetDIdealFloor_whole` (`SheetDIdealFloor` with fold skippers charged) at sm_120's
data (`Definitions.Accounting.Headline.Sm120`), `p = 7/8`, `p₀ = ½` and `Λ = max(κ/ln(8/7), (κ + f)/ln 2)`, with the
data's hypotheses discharged and the terms bounded:

* the data, by `decide`: issue is the floor program's largest facet (`sm120_max`), so `B_♭` is its issue count
  `B_♭ ≈ 17.098` (`sm120_bnd`); every writer kind's `ω ≤ 128` and IMMA16832's is `128`, off shared memory, so
  `w* = c/128` (`sm120_wStar`), `c = 3564/3761` π*'s issue weight; each fold route costs at least `1/8` on issue, so
  `φ* ≥ c/8` (`sm120_phiStar`);
* the certificate is `π*`, `c` on issue and `197/3761` on shared memory (`isCert_piStar`; `demandPrice_piStar` prices
  a demand at it), and FILL-OPT holds since there is no filler (`sm120_fillOpt`);
* `identity` at `d̄ = ā = 0` gives `ε_skip + ε_cert + ε_route + ε_mand ≤ 1 − (N_w w* + N_w φ*)/B_♭ ≤ 1 − 17c/B_♭`, and
  `ε_jl ≤ (c/B_♭)·41 ln 2/(48,000 · ⅛)` at `p = 7/8` from `δ′ ≥ 2^-41`, `ln 2 < 0.6931471808` and `ln(8/7) ≥ ⅛`
  (`sm120_jl`);
* their sum, the folded branch, is below `5806/100000`, by `decide` on the rationals (`sm120_final`);
* the unfolded branch is at most `1 − (N_w w* + N_w φ*)/B_♭ · (1 − 41/48000) ≤ 1 − 17c/B_♭ · (1 − 41/48000)`
  (`Whole.unfolded_floor`), below `5860/100000` by `decide` (`sm120_unfolded`), and it binds;
* (I7) at `7/8` is the receipt split's (`luck_of_receipt_half`).
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
    (1 - 60588 / 3761 / Bq + 3564 / 3761 / Bq * (41 * (6931471808 / 10000000000) / 6000) : ℚ) ≤
      5806 / 100000 := by
  decide +kernel

theorem sm120_unfolded_q : (1 - 60588 / 3761 / Bq * (1 - 41 / 48000) : ℚ) ≤ 5860 / 100000 := by
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
  have : Nonempty ↥Sm120.model.vec :=
    ⟨⟨idx 122 13, List.mem_toFinset.mpr (List.mem_map.mpr ⟨13, by decide, rfl⟩)⟩⟩
  unfold Model.vecMin
  refine le_ciInf fun k => ?_
  rw [sm120_hG]
  simp [sheetCoef]

/-- π* prices a demand at `3564/3761` of its issue entry and `197/3761` of its shared-memory entry. -/
theorem demandPrice_piStar (d : SheetRow Sm120.Pipe → ℝ) :
    demandPrice Sm120.piStar d = 3564 / 3761 * d .issue + 197 / 3761 * d .smem := by
  have h : demandPrice Sm120.piStar d = 3564 / 3761 * demandPrice (onFacet SheetRow.issue) d +
      197 / 3761 * demandPrice (onFacet SheetRow.smem) d := by
    simp only [demandPrice, Sm120.piStar, add_mul, Finset.sum_add_distrib, mul_assoc, Finset.mul_sum]
  rw [h, demandPrice_onFacet, demandPrice_onFacet]

/-- π* is a certificate: both weights are `≥ 0` and they sum to one. -/
theorem isCert_piStar : IsCert Sm120.piStar := by
  obtain ⟨hi0, hi1⟩ := isCert_onFacet (SheetRow.issue : SheetRow Sm120.Pipe)
  obtain ⟨hs0, hs1⟩ := isCert_onFacet (SheetRow.smem : SheetRow Sm120.Pipe)
  refine ⟨fun r => ?_, ?_⟩
  · show 0 ≤ 3564 / 3761 * onFacet SheetRow.issue r + 197 / 3761 * onFacet SheetRow.smem r
    exact add_nonneg (mul_nonneg (by norm_num) (hi0 r)) (mul_nonneg (by norm_num) (hs0 r))
  · show ∑ r, (3564 / 3761 * onFacet SheetRow.issue r + 197 / 3761 * onFacet SheetRow.smem r) = 1
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum, hi1, hs1]
    norm_num

/-- IMMA16832 (writer kind 6) holds no shared memory. -/
theorem sm120_writer6_smem : Sm120.model.demand (.writer 6) .smem = 0 := by
  show Sm120.model.G.a .smem (Sm120.model.cls 6) / Sm120.model.omega 6 = 0
  rw [sm120_hG]
  have h : sheetCoef Sm120.ds120 .smem (Sm120.model.cls 6).val = 0 := by decide +kernel
  rw [h, Rat.cast_zero, zero_div]

/-- Every word route's demand is at least `1/128` on issue. -/
theorem sm120_demand_issue (h : Head (Fin 17) Unit Empty) : 1 / 128 ≤ Sm120.model.demand h .issue := by
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

/-- `w* = c/128` at `π*`, `c = 3564/3761`: IMMA16832 (writer kind 6) is `1/128` on issue and `0` on shared memory, and
every word route at least `1/128` on issue and `0` on shared memory. -/
theorem sm120_wStar : Sm120.model.wStar Sm120.piStar = 3564 / 3761 / 128 := by
  have hd := (stars_nonneg Sm120.model Sm120.flat Sm120.piStar sm120_wf isCert_piStar).1
  refine le_antisymm
    ((iInf_le_price Sm120.piStar Sm120.model.demand (.writer 6)).trans (le_of_eq ?_)) ?_
  · rw [demandPrice_piStar, sm120_writer6_smem]
    show 3564 / 3761 * ((1 : ℝ) / ((128 : ℕ) : ℝ)) + 197 / 3761 * 0 = 3564 / 3761 / 128
    norm_num
  · have : Nonempty (Head (Fin 17) Unit Empty) := ⟨.vector⟩
    unfold Model.wStar
    refine le_ciInf fun h => ?_
    rw [demandPrice_piStar]
    have h1 := sm120_demand_issue h
    have h2 := hd h .smem
    linarith

/-- `φ* ≥ c/8` at `π*`: each fold route's issue count is at least `1/8`, its shared-memory load `≥ 0`. -/
theorem sm120_phiStar : 3564 / 3761 / 8 ≤ Sm120.model.phiStar Sm120.piStar := by
  unfold Model.phiStar
  refine le_ciInf fun φ => ?_
  rw [demandPrice_piStar]
  have hi : (1 : ℝ) / 8 ≤ Sm120.model.fold φ .issue := by
    show (1 : ℝ) / 8 ≤ Sm120.model.G.load (Sparse.counts (Sm120.h1fx1.folds.getD φ.val [])) .issue
    rw [load_counts _ (sheetCoef Sm120.ds120) sm120_hG]
    have h : ((1 / 8 : ℚ) : ℝ) ≤
        ((dot (sheetCoef Sm120.ds120 .issue) 122 (Sm120.h1fx1.folds.getD φ.val []) : ℚ) : ℝ) :=
      Rat.cast_le.mpr (sm120_fold φ.val φ.isLt)
    push_cast at h
    linarith
  have hs : 0 ≤ Sm120.model.fold φ .smem := by
    show 0 ≤ Sm120.model.G.load (Sparse.counts (Sm120.h1fx1.folds.getD φ.val [])) .smem
    exact load_counts_nonneg _ sm120_a_nonneg _ _
  linarith

/-- FILL-OPT on any run: the model has no filler, so `χ* = 0` and the run's filler is priced `≥ 0`. -/
theorem sm120_fillOpt {Ω : Type} [Fintype Ω] (Pl : Play Ω Sm120.checks (Fin 122) (Fin 17) Unit Empty)
    (hSplit : ∀ x, (Pl.run x).Split) : ∀ x, FillOpt Sm120.model (Pl.run x) := by
  intro x lam hlam
  have hchi : Sm120.model.chiStar lam ≤ 0 := by
    refine (iInf_le_price lam Sm120.model.fill ()).trans (le_of_eq ?_)
    show ∑ r, lam r * (0 : ℝ) = 0
    simp
  have hprice : 0 ≤ Sm120.model.G.price lam (Pl.run x).nfill := by
    unfold FacetCosts.price FacetCosts.load
    exact Finset.sum_nonneg fun r _ => mul_nonneg (hlam.1 r)
      (Finset.sum_nonneg fun k _ => mul_nonneg (sm120_a_nonneg r k) ((hSplit x).2.2.1 k))
  show (1 - (0 : ℝ)) * Sm120.model.chiStar lam ≤ _
  rw [sub_zero, one_mul]
  exact hchi.trans hprice

/-- `ε_jl ≤ (c/B_♭)·41·0.6931471808/6000` at `p = 7/8` and `δ′ ≥ 2^-41`, `c = 3564/3761`. -/
theorem sm120_jl (δ' : ℝ) (hδ' : (1 / 2 : ℝ) ^ 41 ≤ δ') :
    Sm120.model.epsJl Sm120.flat Sm120.piStar (7 / 8) δ' Sm120.checks ≤
      3564 / 3761 / Sm120.model.bnd Sm120.flat * (41 * (6931471808 / 10000000000) / 6000) := by
  have hBpos : 0 < Sm120.model.bnd Sm120.flat := sm120_wf.2.2.2.1
  have hκ : Sm120.model.kappa Sm120.flat Sm120.piStar = 3564 / 3761 / Sm120.model.bnd Sm120.flat := by
    unfold Model.kappa
    rw [sm120_Nw, sm120_wStar, show (128 : ℝ) * (3564 / 3761 / 128) = 3564 / 3761 by norm_num]
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
  have hS : ((Sm120.checks : ℕ) : ℝ) = 48000 := by
    show ((48000 : ℕ) : ℝ) = 48000
    norm_num
  unfold Model.epsJl
  rw [hκ, hS]
  exact (mul_div_assoc _ _ _).trans_le (mul_le_mul_of_nonneg_left hq (div_pos (by norm_num) hBpos).le)

theorem sm120_final :
    1 - 60588 / 3761 / Sm120.model.bnd Sm120.flat +
        3564 / 3761 / Sm120.model.bnd Sm120.flat * (41 * (6931471808 / 10000000000) / 6000) ≤
      5806 / 100000 := by
  rw [sm120_bnd]
  have h : ((1 - 60588 / 3761 / Bq + 3564 / 3761 / Bq * (41 * (6931471808 / 10000000000) / 6000) : ℚ) : ℝ) ≤
      ((5806 / 100000 : ℚ) : ℝ) :=
    Rat.cast_le.mpr sm120_final_q
  push_cast at h
  linarith

/-- The terms at `π*`, `p = 7/8` and `δ′ ≥ 2^-41` sum to at most `5806/100000`. -/
theorem sm120_eps_le (δ' : ℝ) (hδ' : (1 / 2 : ℝ) ^ 41 ≤ δ') :
    1 - 5806 / 100000 ≤ 1 - Sm120.model.epsSkip Sm120.flat Sm120.piStar -
      Sm120.model.epsCert Sm120.flat Sm120.piStar -
      Sm120.model.epsRoute Sm120.flat Sm120.piStar -
      Sm120.model.epsMand Sm120.flat Sm120.piStar -
      Sm120.model.epsJl Sm120.flat Sm120.piStar (7 / 8) δ' Sm120.checks := by
  have hid := identity Sm120.model Sm120.flat Sm120.piStar 0 0 sm120_wf
  have hBpos : 0 < Sm120.model.bnd Sm120.flat := sm120_wf.2.2.2.1
  have hχ := (stars_nonneg Sm120.model Sm120.flat Sm120.piStar sm120_wf isCert_piStar).2.2.2
  have hφ := sm120_phiStar
  have hN : 60588 / 3761 ≤ (1 - 0 - 0) * Sm120.model.Nw * Sm120.model.wStar Sm120.piStar +
      (1 - Sm120.model.muF) * Sm120.model.Nw * Sm120.model.phiStar Sm120.piStar +
      (1 - Sm120.model.muX) * Sm120.model.chiStar Sm120.piStar := by
    rw [sm120_Nw, sm120_wStar, show Sm120.model.muF = 0 from rfl, show Sm120.model.muX = 0 from rfl]
    linarith
  have hA := div_le_div_of_nonneg_right hN hBpos.le
  have hLuck : Sm120.model.epsLuck Sm120.flat Sm120.piStar 0 = 0 := zero_mul _
  have hAdapt : Sm120.model.epsAdapt Sm120.flat Sm120.piStar 0 = 0 := zero_mul _
  have hjl := sm120_jl δ' hδ'
  have hfin := sm120_final
  linarith

/-- The unfolded branch's floor at `π*`: `(N_w w* + N_w φ*)/B_♭ · (1 − 41/48000) ≥ 1 − 5860/100000`, since
`N_w w* + N_w φ* ≥ c + 16c = 60588/3761`, `c = 3564/3761`. -/
theorem sm120_unfolded :
    1 - 5860 / 100000 ≤ (Sm120.model.Nw * Sm120.model.wStar Sm120.piStar +
      (1 - Sm120.model.muF) * Sm120.model.Nw * Sm120.model.phiStar Sm120.piStar) /
        Sm120.model.bnd Sm120.flat * (1 - 41 / 48000) := by
  have hBpos : (0 : ℝ) < Bq := by exact_mod_cast sm120_Bpos
  have hY : 60588 / 3761 / (Bq : ℝ) ≤ (Sm120.model.Nw * Sm120.model.wStar Sm120.piStar +
      (1 - Sm120.model.muF) * Sm120.model.Nw * Sm120.model.phiStar Sm120.piStar) /
        Sm120.model.bnd Sm120.flat := by
    rw [sm120_bnd]
    refine div_le_div_of_nonneg_right ?_ hBpos.le
    rw [sm120_Nw, sm120_wStar, show Sm120.model.muF = 0 from rfl]
    have hφ := sm120_phiStar
    linarith
  have hq : ((1 - 60588 / 3761 / Bq * (1 - 41 / 48000) : ℚ) : ℝ) ≤ ((5860 / 100000 : ℚ) : ℝ) :=
    Rat.cast_le.mpr sm120_unfolded_q
  push_cast at hq
  have hm := mul_le_mul_of_nonneg_right hY (by norm_num : (0 : ℝ) ≤ 1 - 41 / 48000)
  linarith

theorem sm120_S : ((Sm120.checks : ℕ) : ℝ) = 48000 := by
  show ((48000 : ℕ) : ℝ) = 48000
  norm_num

end Accounting.SecurityProofs.TS.Headline

namespace Accounting.SecurityProofs.TS

open Accounting.TS Accounting.TS.Headline Accounting.Assumptions.TS Pouw.RelativeSecurity
  Accounting.SecurityProofs.TS.Headline

/-- sm_120's headline at the data sheet with binding at `δ_bind` (`ProofLayer.BindingAt`), the form (I8)'s pieces
take: `Whole.sheetDIdealFloor_whole` at `π*` (`isCert_piStar`), `p = 7/8`, `p₀ = ½`, `Λ = max(κ/ln(8/7), (κ + f)/ln 2)`,
`S = 48,000` and `δ = 2^-40`, (I7) from the receipt split (`luck_of_receipt_half`), (I8) from its pieces
(`ProofsSoundOfPinned`), FILL-OPT from the data, and the terms bounded by `Whole.headline_of_branches`: the folded
branch by `sm120_eps_le`, the unfolded one by `sm120_unfolded`. -/
theorem sm120SheetHeadline_bind {Ω : Type} [Fintype Ω] (Pl : Play Ω Sm120.checks (Fin 122) (Fin 17) Unit Empty)
    (G : ReceiptGame Ω Sm120.checks) (Uf : Unfolded Ω Sm120.checks) {V : Type} {nπ : ℕ}
    (pf : ProofLayer Ω Sm120.checks nπ V) (δbind δzk : ℝ)
    (hδ : (Sm120.checks : ℝ) * (δbind + nπ * δzk) ≤ (1 / 2) ^ 41) (hSplit : ∀ x, (Pl.run x).Split)
    (h1 : ∀ x, SheetClock Sm120.sheet (Pl.run x).run) (h2 : ∀ x, SheetDispatch (Pl.run x).run)
    (h3 : ∀ x, SheetTensor Sm120.sheet (Pl.run x).run) (h4 : ∀ x, SheetPipe Sm120.sheet (Pl.run x).run)
    (h5 : ∀ x, SheetSmem Sm120.sheet (Pl.run x).run) (h6 : ∀ x, SheetWords Sm120.model (Pl.run x))
    (h7 : ∀ x, SheetSemantics Sm120.model (Pl.run x))
    (hF : ∀ x, FoldOptFoldedAt Sm120.model (Pl.run x) ((1 - Uf.share x / Sm120.checks) * Sm120.model.Nw)
      Sm120.piStar)
    (hG : Pl.game = G.game) (hPinG : G.Pinned) (hOp : OperandHMu (3 / 4) G) (hRL : ReceiptLuckAt (1 / 2) G)
    (hWL : WholeLuckAt (1 / 2) G Uf) (hOf : pf.Of Pl) (hB : pf.BindingAt δbind) (hZK : ProofSoundAt pf δzk)
    (hPin : ProofsPin pf) (hRF : RightFixed Pl pf Sm120.model.Nw) :
    prob (fun x => Pl.accepts x ∧ (Pl.run x).run.time / Sm120.model.credit Sm120.flat < 1 - 5860 / 100000) ≤
      (1 / 2) ^ 40 := by
  have hL := luck_of_receipt_half Pl.game G hG hPinG hOp hRL
  have hPS := ProofsSoundOfPinned Ω Sm120.checks nπ Pl pf Sm120.model.Nw δbind δzk hOf hB hZK hPin hRF
  have hδlo : (1 / 2 : ℝ) ^ 41 ≤ (1 / 2) ^ 40 - (Sm120.checks : ℝ) * (δbind + nπ * δzk) := by
    have e : ((1 : ℝ) / 2) ^ 40 = 2 * (1 / 2) ^ 41 := by norm_num
    linarith
  have hδpos : 0 < (1 / 2 : ℝ) ^ 40 - (Sm120.checks : ℝ) * (δbind + nπ * δzk) :=
    lt_of_lt_of_le (by positivity) hδlo
  obtain ⟨hκ0, -⟩ :=
    Whole.kappa_f_nonneg Sm120.model Sm120.flat Sm120.piStar isCert_piStar sm120_wf
  obtain ⟨hΛ, hΛ1, hΛ2⟩ := Whole.max_budget
    (f := (1 - Sm120.model.muF) * Sm120.model.Nw * Sm120.model.phiStar Sm120.piStar /
      Sm120.model.bnd Sm120.flat) hκ0
    (Real.log_pos (by norm_num) : 0 < Real.log (1 / (7 / 8 : ℝ)))
    (Real.log_pos (by norm_num) : 0 < Real.log (1 / (1 / 2 : ℝ)))
  have hmain := Whole.sheetDIdealFloor_whole Sm120.sheet Sm120.model Sm120.flat Pl G Uf
    Sm120.piStar (7 / 8) (1 / 2) _ ((1 / 2) ^ 40) δbind δzk nπ rfl isCert_piStar sm120_wf
    (by decide) (by norm_num) (by norm_num) hΛ hΛ1 hΛ2 hδpos hSplit (le_of_eq sm120_q) (le_of_eq sm120_qT) le_rfl
    h1 h2 h3 h4 h5 h6 h7 hF (sm120_fillOpt Pl hSplit) hG hPinG hL hWL hPS
  refine le_trans (prob_mono ?_) hmain
  rintro x ⟨ha, hlt⟩
  exact ⟨ha, lt_of_lt_of_le hlt (Whole.headline_of_branches Sm120.model Sm120.flat Sm120.piStar
    isCert_piStar sm120_wf Sm120.checks sm120_S _ hδlo _ (le_trans (by norm_num) (sm120_eps_le _ hδlo))
    sm120_unfolded)⟩

/-- sm_120's headline at the data sheet: `sm120SheetHeadline_bind` at binding `√δ_fork`, from the fork form by
`Whole.moved_le_sqrt` (`BindingOfFork`). -/
theorem Sm120SheetHeadline : Accounting.Guarantees.TS.Sm120SheetHeadline := by
  intro Ω _ Pl G Uf V nπ pf δfork δzk hδ hSplit h1 h2 h3 h4 h5 h6 h7 hF hG hPinG hOp hRL hWL hOf hB hZK hPin hRF
  exact sm120SheetHeadline_bind Pl G Uf pf (Real.sqrt δfork) δzk hδ hSplit h1 h2 h3 h4 h5 h6 h7 hF hG hPinG hOp hRL
    hWL hOf (Whole.moved_le_sqrt pf δfork hB) hZK hPin hRF

end Accounting.SecurityProofs.TS
