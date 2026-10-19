import Proofs.Accounting.HeadlineDc.Sm90

/-!
# The H200's headline on its measured vector

`Sm90MeasuredHeadline` is `Whole.dIdealFloor_whole` at DC-90-ri and `dither-2s-licpw-redux-p1-l32`
(`Definitions.Accounting.HeadlineDc.Sm90Measured`), at the data's certificate `π* = Sm90Measured.lamStar` on three
rows (issue, JI90_0, JI90_1; `sm90m_isCert`), `p = 7/8`, `p₀ = ½` and `Λ = max(κ/ln(8/7), (κ + f)/ln 2)`:

* the data, by `decide`: JPA90_0 is ♭'s largest facet of DC-90-ri's 15 (`sm90m_max`), so `B_♭` is its load
  `≈ 199.305` (`sm90m_bnd`); on π*'s three rows each writer kind's word costs at least honest's IGMMA's,
  `w90M ≈ 0.0152938` (`sm90m_writers`), and so do the vector and table words at `½`, priced at each row's ALU
  coefficient, which is that row's least over the vector classes (`sm90m_vec`, `sm90m_wv`), so `w* ≥ w90M`
  (`sm90m_wStar`); each fold route costs at least the lane fold's `phi90M ≈ 0.0322603` (`sm90m_fold`,
  `sm90m_phiStar`);
* A6 (`SheetWords`) gives (I2)/(I4) `WordsWritten` and, at `q = q_T = ½` and `b_T = 0`, (I3) `ScalarRoute`
  (`scalarRoute_half`); (I5) `SideUnits` and FILL-OPT hold at any run, since the model has no side unit and no filler;
* the folded branch from those bounds (`folded_of_lower`): below `2419/100000` by `decide` (`sm90m_folded_q`), and it
  binds (2.41850%);
* the unfolded branch (`unfolded_of_lower`): below `2419/100000` by `decide` (`sm90m_unfolded_q`);
* (I7) at `7/8` is the receipt split's (`luck_of_receipt_half`).
-/

namespace Accounting.SecurityProofs.TS.HeadlineDc

open Accounting.TS Accounting.TS.Headline Accounting.Assumptions.TS Pouw.RelativeSecurity Accounting.SecurityProofs.TS
  Accounting.SecurityProofs.TS.Headline

/-- DC-90-ri's row `r` as rationals, by class index. -/
def vfCoef90 (r i : ℕ) : ℚ := Sparse.coeff (Sm90Measured.dc90ri.getD r []) i

/-- The measured machine's row `r`. -/
def vfRow90 (r : Fin 15) (i : ℕ) : ℚ := vfCoef90 r.val i

/-- π*'s weights, as `Sm90Measured.lamStar` lists them. -/
def lamv90 : Sparse := [(0, 224871, 62500000), (5, 483349359, 500000000), (6, 14851673, 500000000)]

/-- π*'s weight on row `r` as a rational. -/
def lam90 (r : Fin 15) : ℚ := Sparse.coeff lamv90 r.val

/-- ♭'s counts on DC-90-ri, its four lists in `labelled`'s order. -/
def flat90M : Sparse := Sm90.flatWords ++ Sm90.flatFold ++ Sm90.flatFill ++ Sm90Measured.flatSkip

/-- `B_♭` on DC-90-ri as a rational: ♭'s load on JPA90_0. -/
def B90M : ℚ := dot (vfCoef90 10) 28 flat90M

/-- Honest's IGMMA word priced at π* on its three rows. -/
def w90M : ℚ :=
  lam90 0 * (vfRow90 0 14 / 4096) + lam90 5 * (vfRow90 5 14 / 4096) + lam90 6 * (vfRow90 6 14 / 4096)

/-- The lane fold priced at π* on its three rows. -/
def phi90M : ℚ :=
  lam90 0 * dot (vfRow90 0) 28 (Sm90Measured.d2s.folds.getD 0 []) +
    lam90 5 * dot (vfRow90 5) 28 (Sm90Measured.d2s.folds.getD 0 []) +
    lam90 6 * dot (vfRow90 6) 28 (Sm90Measured.d2s.folds.getD 0 [])

theorem sm90m_lamSum : dot (fun _ => 1) 15 lamv90 = 1 := by decide +kernel

theorem sm90m_max : ∀ i < 15, dot (vfCoef90 i) 28 flat90M ≤ B90M := by decide +kernel

theorem sm90m_Bpos : (0 : ℚ) < B90M := by decide +kernel

theorem sm90m_writers : ∀ i < 2, 0 < (Sm90Measured.d2s.writers.getD i (0, 0)).2 ∧
    w90M ≤ lam90 0 * (vfRow90 0 ((Sm90Measured.d2s.writers.getD i (0, 0)).1 % 28) /
        ((Sm90Measured.d2s.writers.getD i (0, 0)).2 : ℚ)) +
      lam90 5 * (vfRow90 5 ((Sm90Measured.d2s.writers.getD i (0, 0)).1 % 28) /
        ((Sm90Measured.d2s.writers.getD i (0, 0)).2 : ℚ)) +
      lam90 6 * (vfRow90 6 ((Sm90Measured.d2s.writers.getD i (0, 0)).1 % 28) /
        ((Sm90Measured.d2s.writers.getD i (0, 0)).2 : ℚ)) := by
  decide +kernel

theorem sm90m_vec : ∀ j ∈ Sm90Measured.d2s.vec, vfRow90 0 0 ≤ vfRow90 0 (j % 28) ∧
    vfRow90 5 0 ≤ vfRow90 5 (j % 28) ∧ vfRow90 6 0 ≤ vfRow90 6 (j % 28) := by
  decide +kernel

theorem sm90m_wv :
    w90M ≤ lam90 0 * (1 / 2 / 32 * vfRow90 0 0) + lam90 5 * (1 / 2 / 32 * vfRow90 5 0) +
      lam90 6 * (1 / 2 / 32 * vfRow90 6 0) := by
  decide +kernel

theorem sm90m_fold : ∀ i < 2, phi90M ≤
    lam90 0 * dot (vfRow90 0) 28 (Sm90Measured.d2s.folds.getD i []) +
      lam90 5 * dot (vfRow90 5) 28 (Sm90Measured.d2s.folds.getD i []) +
      lam90 6 * dot (vfRow90 6) 28 (Sm90Measured.d2s.folds.getD i []) := by
  decide +kernel

theorem sm90m_folded_q :
    (1 - 2419 / 100000 : ℚ) * B90M ≤ 4096 * w90M * (1 - 41 * (6931471808 / 10000000000) / 6000) +
      4096 * phi90M := by
  decide +kernel

theorem sm90m_unfolded_q :
    (1 - 2419 / 100000 : ℚ) * B90M ≤ (4096 * w90M + 4096 * phi90M) * (1 - 41 / 48000) := by
  decide +kernel

theorem sm90m_hG : ∀ r k, Sm90Measured.model.G.a r k = (vfRow90 r k.val : ℝ) := fun _ _ => rfl

theorem sm90m_a_nonneg : ∀ r k, 0 ≤ Sm90Measured.model.G.a r k := fun r k => by
  rw [sm90m_hG]
  exact Rat.cast_nonneg.mpr (coeff_nonneg _ _)

theorem sm90m_bnd : Sm90Measured.model.bnd Sm90Measured.flat = (B90M : ℝ) := by
  unfold Model.bnd
  rw [show Sm90Measured.flat.n = Sparse.counts flat90M from
    labelled_n 28 Sm90.flatWords Sm90.flatFold Sm90.flatFill Sm90Measured.flatSkip]
  exact bound_eq _ vfRow90 sm90m_hG flat90M 10 fun r => sm90m_max r.val r.isLt

theorem sm90m_Nw : Sm90Measured.model.Nw = 4096 := by
  show ((4096 : ℕ) : ℝ) = 4096
  norm_num

theorem sm90m_q : Sm90Measured.model.q = 1 / 2 := by
  show ((1 / 2 : ℚ) : ℝ) = 1 / 2
  norm_num

theorem sm90m_qT : Sm90Measured.model.qT = 1 / 2 := by
  show ((1 / 2 : ℚ) : ℝ) = 1 / 2
  norm_num

theorem sm90m_lam_nonneg (r : Fin 15) : (0 : ℝ) ≤ (lam90 r : ℝ) := Rat.cast_nonneg.mpr (coeff_nonneg _ _)

/-- The data's certificate is one: its weights are `≥ 0` and sum to 1. -/
theorem sm90m_isCert : IsCert Sm90Measured.lamStar := by
  refine ⟨fun r => sm90m_lam_nonneg r, ?_⟩
  have h := sum_mul_coeff 15 (fun _ => 1) lamv90
  rw [sm90m_lamSum] at h
  simp only [one_mul] at h
  show ∑ r : Fin 15, ((Sparse.coeff lamv90 r.val : ℚ) : ℝ) = 1
  exact_mod_cast h

theorem sm90m_wf : Sm90Measured.model.Wf Sm90Measured.flat := by
  refine ⟨sm90m_a_nonneg, ?_, ?_, ?_, fun t => ?_, ?_, ?_, le_rfl, fun u => u.elim, fun φ r => ?_, fun _ _ => le_rfl,
    le_rfl, ?_, le_rfl, ?_⟩
  · show (0 : ℝ) < ((1980 : ℕ) : ℝ)
    norm_num
  · rw [sm90m_Nw]
    norm_num
  · rw [sm90m_bnd]
    exact_mod_cast sm90m_Bpos
  · show (0 : ℝ) < ((Sm90Measured.d2s.writers.getD t.val (0, 0)).2 : ℝ)
    exact_mod_cast (sm90m_writers t.val t.isLt).1
  · rw [sm90m_q]
    norm_num
  · rw [sm90m_qT]
    norm_num
  · show 0 ≤ Sm90Measured.model.G.load (Sparse.counts (Sm90Measured.d2s.folds.getD φ.val [])) r
    exact load_counts_nonneg _ sm90m_a_nonneg _ _
  · show (0 : ℝ) ≤ 1
    norm_num
  · show (0 : ℝ) ≤ 1
    norm_num

/-- A row's least vector-class coefficient is at least its ALU one, when every vector class costs that much there. -/
theorem sm90m_vecMin (r : Fin 15) (h : ∀ j ∈ Sm90Measured.d2s.vec, vfRow90 r 0 ≤ vfRow90 r (j % 28)) :
    (vfRow90 r 0 : ℝ) ≤ Sm90Measured.model.vecMin r := by
  have : Nonempty ↥Sm90Measured.model.vec :=
    ⟨⟨idx 28 3, List.mem_toFinset.mpr (List.mem_map.mpr ⟨3, by decide, rfl⟩)⟩⟩
  unfold Model.vecMin
  refine le_ciInf fun k => ?_
  obtain ⟨j, hj, hk⟩ := List.mem_map.mp (List.mem_toFinset.mp k.2)
  rw [sm90m_hG, ← hk]
  exact Rat.cast_le.mpr (h j hj)

/-- `w* ≥ w90M` at π*: every writer kind's word costs at least honest's IGMMA's on π*'s rows, and so do the vector
and table words at `½`. -/
theorem sm90m_wStar : (w90M : ℝ) ≤ Sm90Measured.model.wStar Sm90Measured.lamStar := by
  have hvec : (w90M : ℝ) ≤ (lam90 0 : ℝ) * (1 / 2 / 32 * Sm90Measured.model.vecMin 0) +
      (lam90 5 : ℝ) * (1 / 2 / 32 * Sm90Measured.model.vecMin 5) +
      (lam90 6 : ℝ) * (1 / 2 / 32 * Sm90Measured.model.vecMin 6) := by
    have hq : ((w90M : ℚ) : ℝ) ≤ ((lam90 0 * (1 / 2 / 32 * vfRow90 0 0) + lam90 5 * (1 / 2 / 32 * vfRow90 5 0) +
        lam90 6 * (1 / 2 / 32 * vfRow90 6 0) : ℚ) : ℝ) := Rat.cast_le.mpr sm90m_wv
    push_cast at hq
    have e0 := mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
      (sm90m_vecMin 0 fun j hj => (sm90m_vec j hj).1) (by norm_num : (0 : ℝ) ≤ 1 / 2 / 32)) (sm90m_lam_nonneg 0)
    have e5 := mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
      (sm90m_vecMin 5 fun j hj => (sm90m_vec j hj).2.1) (by norm_num : (0 : ℝ) ≤ 1 / 2 / 32))
      (sm90m_lam_nonneg 5)
    have e6 := mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
      (sm90m_vecMin 6 fun j hj => (sm90m_vec j hj).2.2) (by norm_num : (0 : ℝ) ≤ 1 / 2 / 32))
      (sm90m_lam_nonneg 6)
    linarith
  refine wStar_ge_triple Sm90Measured.model Sm90Measured.flat Sm90Measured.lamStar sm90m_isCert sm90m_wf 0 5 6
    (by decide) (by decide) (by decide) _ fun h => ?_
  cases h with
  | writer t =>
    have hw := sm90m_writers t.val t.isLt
    show (w90M : ℝ) ≤
      (lam90 0 : ℝ) * ((vfRow90 0 ((Sm90Measured.d2s.writers.getD t.val (0, 0)).1 % 28) : ℝ) /
        ((Sm90Measured.d2s.writers.getD t.val (0, 0)).2 : ℝ)) +
      (lam90 5 : ℝ) * ((vfRow90 5 ((Sm90Measured.d2s.writers.getD t.val (0, 0)).1 % 28) : ℝ) /
        ((Sm90Measured.d2s.writers.getD t.val (0, 0)).2 : ℝ)) +
      (lam90 6 : ℝ) * ((vfRow90 6 ((Sm90Measured.d2s.writers.getD t.val (0, 0)).1 % 28) : ℝ) /
        ((Sm90Measured.d2s.writers.getD t.val (0, 0)).2 : ℝ))
    exact_mod_cast hw.2
  | vector =>
    show (w90M : ℝ) ≤ (lam90 0 : ℝ) * (Sm90Measured.model.q / 32 * Sm90Measured.model.vecMin 0) +
      (lam90 5 : ℝ) * (Sm90Measured.model.q / 32 * Sm90Measured.model.vecMin 5) +
      (lam90 6 : ℝ) * (Sm90Measured.model.q / 32 * Sm90Measured.model.vecMin 6)
    rw [sm90m_q]
    exact hvec
  | table l =>
    show (w90M : ℝ) ≤
      (lam90 0 : ℝ) * max (Sm90Measured.model.qT / 32 * Sm90Measured.model.vecMin 0)
        (Sm90Measured.model.bT * Sm90Measured.model.G.a 0 (Sm90Measured.model.bytes l)) +
      (lam90 5 : ℝ) * max (Sm90Measured.model.qT / 32 * Sm90Measured.model.vecMin 5)
        (Sm90Measured.model.bT * Sm90Measured.model.G.a 5 (Sm90Measured.model.bytes l)) +
      (lam90 6 : ℝ) * max (Sm90Measured.model.qT / 32 * Sm90Measured.model.vecMin 6)
        (Sm90Measured.model.bT * Sm90Measured.model.G.a 6 (Sm90Measured.model.bytes l))
    refine le_trans ?_ (add_le_add (add_le_add
      (mul_le_mul_of_nonneg_left (le_max_left _ _) (sm90m_lam_nonneg 0))
      (mul_le_mul_of_nonneg_left (le_max_left _ _) (sm90m_lam_nonneg 5)))
      (mul_le_mul_of_nonneg_left (le_max_left _ _) (sm90m_lam_nonneg 6)))
    rw [sm90m_qT]
    exact hvec
  | side u => exact u.elim

/-- `φ* ≥` the lane fold's price at π*. -/
theorem sm90m_phiStar : (phi90M : ℝ) ≤ Sm90Measured.model.phiStar Sm90Measured.lamStar := by
  refine phiStar_ge_triple Sm90Measured.model Sm90Measured.flat Sm90Measured.lamStar sm90m_isCert sm90m_wf 0 5 6
    (by decide) (by decide) (by decide) _ fun φ => ?_
  show (phi90M : ℝ) ≤
    (lam90 0 : ℝ) * Sm90Measured.model.G.load (Sparse.counts (Sm90Measured.d2s.folds.getD φ.val [])) 0 +
      (lam90 5 : ℝ) * Sm90Measured.model.G.load (Sparse.counts (Sm90Measured.d2s.folds.getD φ.val [])) 5 +
      (lam90 6 : ℝ) * Sm90Measured.model.G.load (Sparse.counts (Sm90Measured.d2s.folds.getD φ.val [])) 6
  rw [load_counts _ vfRow90 sm90m_hG, load_counts _ vfRow90 sm90m_hG, load_counts _ vfRow90 sm90m_hG]
  have h := (Rat.cast_le (K := ℝ)).mpr (sm90m_fold φ.val φ.isLt)
  push_cast at h
  linarith

theorem sm90m_fillOpt {Ω : Type} [Fintype Ω] (Pl : Play Ω Sm90.checks (Fin 28) (Fin 2) Unit Empty)
    (hSplit : ∀ x, (Pl.run x).Split) : ∀ x, FillOpt Sm90Measured.model (Pl.run x) := fun x =>
  fillOpt_of_zero Sm90Measured.model sm90m_a_nonneg (fun _ _ => rfl) rfl (Pl.run x) (hSplit x)

/-- The folded branch at π*, `p = 7/8` and `δ′ ≥ 2^-41` is at least `1 − 2419/100000`. -/
theorem sm90m_folded (δ' : ℝ) (hδ' : (1 / 2 : ℝ) ^ 41 ≤ δ') :
    1 - 2419 / 100000 ≤ 1 - Sm90Measured.model.epsSkip Sm90Measured.flat Sm90Measured.lamStar -
      Sm90Measured.model.epsCert Sm90Measured.flat Sm90Measured.lamStar -
      Sm90Measured.model.epsRoute Sm90Measured.flat Sm90Measured.lamStar -
      Sm90Measured.model.epsMand Sm90Measured.flat Sm90Measured.lamStar -
      Sm90Measured.model.epsJl Sm90Measured.flat Sm90Measured.lamStar (7 / 8) δ' Sm90.checks := by
  refine folded_of_lower Sm90Measured.model Sm90Measured.flat Sm90Measured.lamStar sm90m_isCert sm90m_wf rfl
    rfl Sm90.checks sm90_S δ' hδ' _ _ _ sm90m_wStar sm90m_phiStar ?_
  have hB : (0 : ℝ) < (B90M : ℝ) := by exact_mod_cast sm90m_Bpos
  have h := (Rat.cast_le (K := ℝ)).mpr sm90m_folded_q
  push_cast at h
  rw [sm90m_Nw, sm90m_bnd, le_div_iff₀ hB]
  linarith

/-- The unfolded branch's floor at π* is at least `1 − 2419/100000`. -/
theorem sm90m_unfolded :
    1 - 2419 / 100000 ≤ (Sm90Measured.model.Nw * Sm90Measured.model.wStar Sm90Measured.lamStar +
      (1 - Sm90Measured.model.muF) * Sm90Measured.model.Nw *
        Sm90Measured.model.phiStar Sm90Measured.lamStar) /
        Sm90Measured.model.bnd Sm90Measured.flat * (1 - 41 / 48000) := by
  refine unfolded_of_lower Sm90Measured.model Sm90Measured.flat Sm90Measured.lamStar sm90m_wf rfl _ _ _
    sm90m_wStar sm90m_phiStar ?_
  have hB : (0 : ℝ) < (B90M : ℝ) := by exact_mod_cast sm90m_Bpos
  have h := (Rat.cast_le (K := ℝ)).mpr sm90m_unfolded_q
  push_cast at h
  rw [sm90m_Nw, sm90m_bnd, div_mul_eq_mul_div, le_div_iff₀ hB]
  linarith

end Accounting.SecurityProofs.TS.HeadlineDc

namespace Accounting.SecurityProofs.TS

open Accounting.TS Accounting.TS.Headline Accounting.Assumptions.TS Pouw.RelativeSecurity
  Accounting.SecurityProofs.TS.Headline Accounting.SecurityProofs.TS.HeadlineDc

/-- The H200's headline on DC-90-ri: `Whole.dIdealFloor_whole` at π* (`Sm90Measured.lamStar`, on issue, JI90_0 and
JI90_1), `p = 7/8`, `p₀ = ½`, `Λ = max(κ/ln(8/7), (κ + f)/ln 2)`, `S = 48,000` and `δ = 2^-40`, (I2)/(I4) and (I3)
from A6 (`scalarRoute_half`), (I7) from the receipt split (`luck_of_receipt_half`), (I8) from its pieces
(`ProofsSoundOfPinned`), (I5) `SideUnits` and FILL-OPT from the data, and the terms bounded by
`Whole.headline_of_branches`: the folded branch by `sm90m_folded`, the unfolded one by `sm90m_unfolded`. -/
theorem Sm90MeasuredHeadline : Accounting.Guarantees.TS.Sm90MeasuredHeadline := by
  intro Ω _ Pl G Uf V nπ pf δbind δzk hδ hSplit h1 h6 hN hF hG hPinG hOp hRL hWL hOf hB hZK hPin hRF
  have hL := luck_of_receipt_half Pl.game G hG hPinG hOp hRL
  have hPS := ProofsSoundOfPinned Ω Sm90.checks nπ Pl pf Sm90Measured.model.Nw δbind δzk hOf hB hZK hPin hRF
  have hδlo : (1 / 2 : ℝ) ^ 41 ≤ (1 / 2) ^ 40 - (Sm90.checks : ℝ) * (δbind + nπ * δzk) := by
    have e : ((1 : ℝ) / 2) ^ 40 = 2 * (1 / 2) ^ 41 := by norm_num
    linarith
  have hδpos : 0 < (1 / 2 : ℝ) ^ 40 - (Sm90.checks : ℝ) * (δbind + nπ * δzk) :=
    lt_of_lt_of_le (by positivity) hδlo
  obtain ⟨hκ0, -⟩ := Whole.kappa_f_nonneg Sm90Measured.model Sm90Measured.flat Sm90Measured.lamStar
    sm90m_isCert sm90m_wf
  obtain ⟨hΛ, hΛ1, hΛ2⟩ := Whole.max_budget
    (f := (1 - Sm90Measured.model.muF) * Sm90Measured.model.Nw *
      Sm90Measured.model.phiStar Sm90Measured.lamStar / Sm90Measured.model.bnd Sm90Measured.flat) hκ0
    (Real.log_pos (by norm_num) : 0 < Real.log (1 / (7 / 8 : ℝ)))
    (Real.log_pos (by norm_num) : 0 < Real.log (1 / (1 / 2 : ℝ)))
  have hmain := Whole.dIdealFloor_whole Sm90Measured.model Sm90Measured.flat Pl G Uf
    Sm90Measured.lamStar (7 / 8) (1 / 2) _ ((1 / 2) ^ 40) δbind δzk nπ sm90m_isCert sm90m_wf
    (by decide) (by norm_num) (by norm_num) hΛ hΛ1 hΛ2 hδpos hSplit h1 (fun x => (h6 x).1)
    (fun x => scalarRoute_half Sm90Measured.model _ (hSplit x) (h6 x) (le_of_eq sm90m_q) (le_of_eq sm90m_qT)
      le_rfl)
    (fun _ u => u.elim) hN hF (sm90m_fillOpt Pl hSplit) hG hPinG hL hWL hPS
  refine le_trans (prob_mono ?_) hmain
  rintro x ⟨ha, hlt⟩
  exact ⟨ha, lt_of_lt_of_le hlt (Whole.headline_of_branches Sm90Measured.model Sm90Measured.flat
    Sm90Measured.lamStar sm90m_isCert sm90m_wf Sm90.checks sm90_S _ hδlo _
    (sm90m_folded _ hδlo) sm90m_unfolded)⟩

end Accounting.SecurityProofs.TS
