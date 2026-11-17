module

public import Proofs.Pouw.HeadlineDc.Sm90Measured
public import Proofs.ResourceAccounting.Lemmas.TS

/-! The H200's headline on its measured vector under (I3) at 3/5, which the statement takes as a hypothesis. -/

@[expose] public section

namespace ResourceAccounting.SecurityProofs.TS.HeadlineDc

open ResourceAccounting.TS ResourceAccounting.TS.Headline ResourceAccounting.Assumptions.TS Pouw.RelativeSecurity ResourceAccounting.SecurityProofs.TS
  ResourceAccounting.SecurityProofs.TS.Headline

def lamv90I : Sparse := [(5, 12809871, 15625000), (6, 16889149, 500000000), (10, 73194979, 500000000)]

def lam90I (r : Fin 15) : ℚ := Sparse.coeff lamv90I r.val

def wordPrice90I (i : ℕ) : ℚ :=
  lam90I 5 * (vfRow90 5 ((Sm90Measured.d2s.writers.getD i (0, 0)).1 % 50) /
      ((Sm90Measured.d2s.writers.getD i (0, 0)).2 : ℚ)) +
    lam90I 6 * (vfRow90 6 ((Sm90Measured.d2s.writers.getD i (0, 0)).1 % 50) /
      ((Sm90Measured.d2s.writers.getD i (0, 0)).2 : ℚ)) +
    lam90I 10 * (vfRow90 10 ((Sm90Measured.d2s.writers.getD i (0, 0)).1 % 50) /
      ((Sm90Measured.d2s.writers.getD i (0, 0)).2 : ℚ))

def vecPrice90I : ℚ := lam90I 5 * (3 / 5 / 32 * vfRow90 5 0) + lam90I 6 * (3 / 5 / 32 * vfRow90 6 0)

def w90I : ℚ := ((List.range 2).map wordPrice90I).foldr min vecPrice90I

def price90I (i : ℕ) : ℚ :=
  lam90I 5 * dot (vfRow90 5) 50 (Sm90Measured.d2s.folds.getD i []) +
    lam90I 6 * dot (vfRow90 6) 50 (Sm90Measured.d2s.folds.getD i []) +
    lam90I 10 * dot (vfRow90 10) 50 (Sm90Measured.d2s.folds.getD i [])

def phi90I : ℚ := ((List.range 56).map price90I).foldr min (price90I 0)

theorem sm90i_lamSum : dot (fun _ => 1) 15 lamv90I = 1 := by decide +kernel

theorem sm90i_writers : ∀ i < 2, 0 < (Sm90Measured.d2s.writers.getD i (0, 0)).2 ∧
    w90I ≤ lam90I 5 * (vfRow90 5 ((Sm90Measured.d2s.writers.getD i (0, 0)).1 % 50) /
        ((Sm90Measured.d2s.writers.getD i (0, 0)).2 : ℚ)) +
      lam90I 6 * (vfRow90 6 ((Sm90Measured.d2s.writers.getD i (0, 0)).1 % 50) /
        ((Sm90Measured.d2s.writers.getD i (0, 0)).2 : ℚ)) +
      lam90I 10 * (vfRow90 10 ((Sm90Measured.d2s.writers.getD i (0, 0)).1 % 50) /
        ((Sm90Measured.d2s.writers.getD i (0, 0)).2 : ℚ)) := by
  decide +kernel

theorem sm90i_vec : ∀ j ∈ Sm90MeasuredI3.d2s.vec, vfRow90 5 0 ≤ vfRow90 5 (j % 50) ∧
    vfRow90 6 0 ≤ vfRow90 6 (j % 50) := by
  decide +kernel

theorem sm90i_wv : w90I ≤ lam90I 5 * (3 / 5 / 32 * vfRow90 5 0) + lam90I 6 * (3 / 5 / 32 * vfRow90 6 0) := by
  decide +kernel

theorem sm90i_fold : ∀ i < 56, phi90I ≤
    lam90I 5 * dot (vfRow90 5) 50 (Sm90Measured.d2s.folds.getD i []) +
      lam90I 6 * dot (vfRow90 6) 50 (Sm90Measured.d2s.folds.getD i []) +
      lam90I 10 * dot (vfRow90 10) 50 (Sm90Measured.d2s.folds.getD i []) := by
  decide +kernel

theorem sm90i_folded_q :
    (1 - 1086 / 100000 : ℚ) * B90M ≤ 4096 * w90I * (1 - 41 * (6931471808 / 10000000000) / 6400) +
      4096 * phi90I := by
  decide +kernel

theorem sm90i_unfolded_q :
    (1 - 1086 / 100000 : ℚ) * B90M ≤ (4096 * w90I + 4096 * phi90I) * (1 - 41 / 48000) := by
  decide +kernel

theorem sm90i_hG : ∀ r k, Sm90MeasuredI3.model.G.a r k = (vfRow90 r k.val : ℝ) := fun _ _ => rfl

theorem sm90i_a_nonneg : ∀ r k, 0 ≤ Sm90MeasuredI3.model.G.a r k := fun r k => by
  rw [sm90i_hG]
  exact Rat.cast_nonneg.mpr (coeff_nonneg _ _)

theorem sm90i_bnd : Sm90MeasuredI3.model.bnd Sm90Measured.flat = (B90M : ℝ) := by
  unfold Model.bnd
  rw [show Sm90Measured.flat.n = Sparse.counts flat90M from
    labelled_n 50 Sm90.flatWords Sm90Measured.flatFold Sm90.flatFill Sm90Measured.flatSkip]
  exact bound_eq _ vfRow90 sm90i_hG flat90M 10 fun r => sm90m_max r.val r.isLt

theorem sm90i_Nw : Sm90MeasuredI3.model.Nw = 4096 := by
  show ((4096 : ℕ) : ℝ) = 4096
  norm_num

theorem sm90i_q : Sm90MeasuredI3.model.q = 3 / 5 := by
  show ((3 / 5 : ℚ) : ℝ) = 3 / 5
  norm_num

theorem sm90i_qT : Sm90MeasuredI3.model.qT = 3 / 5 := by
  show ((3 / 5 : ℚ) : ℝ) = 3 / 5
  norm_num

theorem sm90i_lam_nonneg (r : Fin 15) : (0 : ℝ) ≤ (lam90I r : ℝ) := Rat.cast_nonneg.mpr (coeff_nonneg _ _)

theorem sm90i_isCert : IsCert Sm90MeasuredI3.lamStar := by
  refine ⟨fun r => sm90i_lam_nonneg r, ?_⟩
  have h := sum_mul_coeff 15 (fun _ => 1) lamv90I
  rw [sm90i_lamSum] at h
  simp only [one_mul] at h
  show ∑ r : Fin 15, ((Sparse.coeff lamv90I r.val : ℚ) : ℝ) = 1
  exact_mod_cast h

theorem sm90i_wf : Sm90MeasuredI3.model.Wf Sm90Measured.flat := by
  refine ⟨sm90i_a_nonneg, ?_, ?_, ?_, fun t => ?_, ?_, ?_, le_rfl, fun u => u.elim, fun φ r => ?_, fun _ _ => le_rfl,
    le_rfl, ?_, le_rfl, ?_⟩
  · exact Nat.cast_pos.mpr (by decide)
  · rw [sm90i_Nw]
    norm_num
  · rw [sm90i_bnd]
    exact_mod_cast sm90m_Bpos
  · show (0 : ℝ) < ((Sm90Measured.d2s.writers.getD t.val (0, 0)).2 : ℝ)
    exact_mod_cast (sm90i_writers t.val t.isLt).1
  · rw [sm90i_q]
    norm_num
  · rw [sm90i_qT]
    norm_num
  · show 0 ≤ Sm90MeasuredI3.model.G.load (Sparse.counts (Sm90Measured.d2s.folds.getD φ.val [])) r
    exact load_counts_nonneg _ sm90i_a_nonneg _ _
  · show (0 : ℝ) ≤ 1
    norm_num
  · show (0 : ℝ) ≤ 1
    norm_num

theorem sm90i_vec_nonempty : Nonempty ↥Sm90MeasuredI3.model.vec :=
  ⟨⟨idx 50 9, List.mem_toFinset.mpr (List.mem_map.mpr ⟨9, by decide, rfl⟩)⟩⟩

theorem sm90i_vecMin (r : Fin 15) (h : ∀ j ∈ Sm90MeasuredI3.d2s.vec, vfRow90 r 0 ≤ vfRow90 r (j % 50)) :
    (vfRow90 r 0 : ℝ) ≤ Sm90MeasuredI3.model.vecMin r := by
  have := sm90i_vec_nonempty
  unfold Model.vecMin
  refine le_ciInf fun k => ?_
  obtain ⟨j, hj, hk⟩ := List.mem_map.mp (List.mem_toFinset.mp k.2)
  rw [sm90i_hG, ← hk]
  exact Rat.cast_le.mpr (h j hj)

theorem sm90i_vecMin_nonneg (r : Fin 15) : 0 ≤ Sm90MeasuredI3.model.vecMin r := by
  have := sm90i_vec_nonempty
  unfold Model.vecMin
  exact le_ciInf fun k => sm90i_a_nonneg r k.1

theorem sm90i_wStar : (w90I : ℝ) ≤ Sm90MeasuredI3.model.wStar Sm90MeasuredI3.lamStar := by
  have hvec : (w90I : ℝ) ≤ (lam90I 5 : ℝ) * (3 / 5 / 32 * Sm90MeasuredI3.model.vecMin 5) +
      (lam90I 6 : ℝ) * (3 / 5 / 32 * Sm90MeasuredI3.model.vecMin 6) +
      (lam90I 10 : ℝ) * (3 / 5 / 32 * Sm90MeasuredI3.model.vecMin 10) := by
    have hq : ((w90I : ℚ) : ℝ) ≤
        ((lam90I 5 * (3 / 5 / 32 * vfRow90 5 0) + lam90I 6 * (3 / 5 / 32 * vfRow90 6 0) : ℚ) : ℝ) :=
      Rat.cast_le.mpr sm90i_wv
    push_cast at hq
    have e5 := mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
      (sm90i_vecMin 5 fun j hj => (sm90i_vec j hj).1) (by norm_num : (0 : ℝ) ≤ 3 / 5 / 32))
      (sm90i_lam_nonneg 5)
    have e6 := mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
      (sm90i_vecMin 6 fun j hj => (sm90i_vec j hj).2) (by norm_num : (0 : ℝ) ≤ 3 / 5 / 32))
      (sm90i_lam_nonneg 6)
    have e10 : (0 : ℝ) ≤ (lam90I 10 : ℝ) * (3 / 5 / 32 * Sm90MeasuredI3.model.vecMin 10) :=
      mul_nonneg (sm90i_lam_nonneg 10) (mul_nonneg (by norm_num) (sm90i_vecMin_nonneg 10))
    linarith
  refine wStar_ge_triple Sm90MeasuredI3.model Sm90Measured.flat Sm90MeasuredI3.lamStar sm90i_isCert sm90i_wf 5 6 10
    (by decide) (by decide) (by decide) _ fun h => ?_
  cases h with
  | writer t =>
    have hw := sm90i_writers t.val t.isLt
    show (w90I : ℝ) ≤
      (lam90I 5 : ℝ) * ((vfRow90 5 ((Sm90Measured.d2s.writers.getD t.val (0, 0)).1 % 50) : ℝ) /
        ((Sm90Measured.d2s.writers.getD t.val (0, 0)).2 : ℝ)) +
      (lam90I 6 : ℝ) * ((vfRow90 6 ((Sm90Measured.d2s.writers.getD t.val (0, 0)).1 % 50) : ℝ) /
        ((Sm90Measured.d2s.writers.getD t.val (0, 0)).2 : ℝ)) +
      (lam90I 10 : ℝ) * ((vfRow90 10 ((Sm90Measured.d2s.writers.getD t.val (0, 0)).1 % 50) : ℝ) /
        ((Sm90Measured.d2s.writers.getD t.val (0, 0)).2 : ℝ))
    exact_mod_cast hw.2
  | vector =>
    show (w90I : ℝ) ≤ (lam90I 5 : ℝ) * (Sm90MeasuredI3.model.q / 32 * Sm90MeasuredI3.model.vecMin 5) +
      (lam90I 6 : ℝ) * (Sm90MeasuredI3.model.q / 32 * Sm90MeasuredI3.model.vecMin 6) +
      (lam90I 10 : ℝ) * (Sm90MeasuredI3.model.q / 32 * Sm90MeasuredI3.model.vecMin 10)
    rw [sm90i_q]
    exact hvec
  | table l =>
    show (w90I : ℝ) ≤
      (lam90I 5 : ℝ) * max (Sm90MeasuredI3.model.qT / 32 * Sm90MeasuredI3.model.vecMin 5)
        (Sm90MeasuredI3.model.bT * Sm90MeasuredI3.model.G.a 5 (Sm90MeasuredI3.model.bytes l)) +
      (lam90I 6 : ℝ) * max (Sm90MeasuredI3.model.qT / 32 * Sm90MeasuredI3.model.vecMin 6)
        (Sm90MeasuredI3.model.bT * Sm90MeasuredI3.model.G.a 6 (Sm90MeasuredI3.model.bytes l)) +
      (lam90I 10 : ℝ) * max (Sm90MeasuredI3.model.qT / 32 * Sm90MeasuredI3.model.vecMin 10)
        (Sm90MeasuredI3.model.bT * Sm90MeasuredI3.model.G.a 10 (Sm90MeasuredI3.model.bytes l))
    refine le_trans ?_ (add_le_add (add_le_add
      (mul_le_mul_of_nonneg_left (le_max_left _ _) (sm90i_lam_nonneg 5))
      (mul_le_mul_of_nonneg_left (le_max_left _ _) (sm90i_lam_nonneg 6)))
      (mul_le_mul_of_nonneg_left (le_max_left _ _) (sm90i_lam_nonneg 10)))
    rw [sm90i_qT]
    exact hvec
  | side u => exact u.elim

theorem sm90i_phiStar : (phi90I : ℝ) ≤ Sm90MeasuredI3.model.phiStar Sm90MeasuredI3.lamStar := by
  refine phiStar_ge_triple Sm90MeasuredI3.model Sm90Measured.flat Sm90MeasuredI3.lamStar sm90i_isCert sm90i_wf 5 6
    10 (by decide) (by decide) (by decide) _ fun φ => ?_
  show (phi90I : ℝ) ≤
    (lam90I 5 : ℝ) * Sm90MeasuredI3.model.G.load (Sparse.counts (Sm90Measured.d2s.folds.getD φ.val [])) 5 +
      (lam90I 6 : ℝ) * Sm90MeasuredI3.model.G.load (Sparse.counts (Sm90Measured.d2s.folds.getD φ.val [])) 6 +
      (lam90I 10 : ℝ) * Sm90MeasuredI3.model.G.load (Sparse.counts (Sm90Measured.d2s.folds.getD φ.val [])) 10
  rw [load_counts _ vfRow90 sm90i_hG, load_counts _ vfRow90 sm90i_hG, load_counts _ vfRow90 sm90i_hG]
  have h := (Rat.cast_le (K := ℝ)).mpr (sm90i_fold φ.val φ.isLt)
  push_cast at h
  linarith

theorem sm90i_fillOpt {Ω : Type} [Fintype Ω] (Pl : Play Ω Sm90.checks (Fin 50) (Fin 2) Unit Empty)
    (hSplit : ∀ x, (Pl.run x).Split) : ∀ x, FillOpt Sm90MeasuredI3.model (Pl.run x) := fun x =>
  fillOpt_of_zero Sm90MeasuredI3.model sm90i_a_nonneg (fun _ _ => rfl) rfl (Pl.run x) (hSplit x)

theorem sm90i_folded (δ' : ℝ) (hδ' : (1 / 2 : ℝ) ^ 41 ≤ δ') :
    1 - 1086 / 100000 ≤ 1 - Sm90MeasuredI3.model.epsSkip Sm90Measured.flat Sm90MeasuredI3.lamStar -
      Sm90MeasuredI3.model.epsCert Sm90Measured.flat Sm90MeasuredI3.lamStar -
      Sm90MeasuredI3.model.epsRoute Sm90Measured.flat Sm90MeasuredI3.lamStar -
      Sm90MeasuredI3.model.epsMand Sm90Measured.flat Sm90MeasuredI3.lamStar -
      Sm90MeasuredI3.model.epsJl Sm90Measured.flat Sm90MeasuredI3.lamStar (7 / 8) δ' Sm90.checks := by
  refine folded_of_lower_at Sm90MeasuredI3.model Sm90Measured.flat Sm90MeasuredI3.lamStar sm90i_isCert sm90i_wf rfl
    rfl Sm90.checks sm90_S δ' hδ' _ _ _ sm90i_wStar sm90i_phiStar (41 * (6931471808 / 10000000000) / 6400) le_rfl
    (by norm_num) ?_
  have hB : (0 : ℝ) < (B90M : ℝ) := by exact_mod_cast sm90m_Bpos
  have h := (Rat.cast_le (K := ℝ)).mpr sm90i_folded_q
  push_cast at h
  rw [sm90i_Nw, sm90i_bnd, le_div_iff₀ hB]
  linarith

theorem sm90i_unfolded :
    1 - 1086 / 100000 ≤ (Sm90MeasuredI3.model.Nw * Sm90MeasuredI3.model.wStar Sm90MeasuredI3.lamStar +
      (1 - Sm90MeasuredI3.model.muF) * Sm90MeasuredI3.model.Nw *
        Sm90MeasuredI3.model.phiStar Sm90MeasuredI3.lamStar) /
        Sm90MeasuredI3.model.bnd Sm90Measured.flat * (1 - 41 / 48000) := by
  refine unfolded_of_lower Sm90MeasuredI3.model Sm90Measured.flat Sm90MeasuredI3.lamStar sm90i_wf rfl _ _ _
    sm90i_wStar sm90i_phiStar ?_
  have hB : (0 : ℝ) < (B90M : ℝ) := by exact_mod_cast sm90m_Bpos
  have h := (Rat.cast_le (K := ℝ)).mpr sm90i_unfolded_q
  push_cast at h
  rw [sm90i_Nw, sm90i_bnd, div_mul_eq_mul_div, le_div_iff₀ hB]
  linarith

end ResourceAccounting.SecurityProofs.TS.HeadlineDc

namespace ResourceAccounting.SecurityProofs.TS

open ResourceAccounting.TS ResourceAccounting.TS.Headline ResourceAccounting.Assumptions.TS Pouw.RelativeSecurity
  ResourceAccounting.SecurityProofs.TS.Headline ResourceAccounting.SecurityProofs.TS.HeadlineDc

/-- The H200's headline on DC-90-ri under (I3) at 3/5: `Whole.dIdealFloor_whole` at π* (`Sm90MeasuredI3.lamStar`, on
JI90_0, JI90_1 and JPA90_0), `p = 7/8`, `p₀ = ½`, `Λ = max(κ/ln(8/7), (κ + f)/ln 2)`, `S = 48,000` and `δ = 2^-40`,
(I2)/(I4) `WordsWritten` and (I3) `ScalarRoute` the statement's own hypotheses, (I7) from the receipt split
(`luck_of_receipt_half`), (I8) from its pieces (`ProofsSoundOfPinned`, binding at `√δ_fork` by
`Whole.moved_le_sqrt`), (I5) `SideUnits` and FILL-OPT from the data, and the terms bounded by
`Whole.headline_of_branches`: the folded branch by `sm90i_folded`, the unfolded one by `sm90i_unfolded`. -/
theorem Sm90MeasuredI3Headline : ResourceAccounting.Guarantees.TS.Sm90MeasuredI3Headline := by
  intro Ω _ Pl G Uf V nπ pf δfork δzk hδ hSplit h1 hW hSR hN hF hG hPinG hOp hRL hWL hOf hB hZK hPin hRF
  set δbind := Real.sqrt δfork
  have hL := luck_of_receipt_half Pl.game G hG hPinG hOp hRL
  have hPS := ProofsSoundOfPinned Ω Sm90.checks nπ Pl pf Sm90MeasuredI3.model.Nw δbind δzk hOf
    (Whole.moved_le_sqrt pf δfork hB) hZK hPin hRF
  have hδlo : (1 / 2 : ℝ) ^ 41 ≤ (1 / 2) ^ 40 - (Sm90.checks : ℝ) * (δbind + nπ * δzk) := by
    have e : ((1 : ℝ) / 2) ^ 40 = 2 * (1 / 2) ^ 41 := by norm_num
    linarith
  have hδpos : 0 < (1 / 2 : ℝ) ^ 40 - (Sm90.checks : ℝ) * (δbind + nπ * δzk) :=
    lt_of_lt_of_le (by positivity) hδlo
  obtain ⟨hκ0, -⟩ := Whole.kappa_f_nonneg Sm90MeasuredI3.model Sm90Measured.flat Sm90MeasuredI3.lamStar
    sm90i_isCert sm90i_wf
  obtain ⟨hΛ, hΛ1, hΛ2⟩ := Whole.max_budget
    (f := (1 - Sm90MeasuredI3.model.muF) * Sm90MeasuredI3.model.Nw *
      Sm90MeasuredI3.model.phiStar Sm90MeasuredI3.lamStar / Sm90MeasuredI3.model.bnd Sm90Measured.flat) hκ0
    (Real.log_pos (by norm_num) : 0 < Real.log (1 / (7 / 8 : ℝ)))
    (Real.log_pos (by norm_num) : 0 < Real.log (1 / (1 / 2 : ℝ)))
  have hmain := Whole.dIdealFloor_whole Sm90MeasuredI3.model Sm90Measured.flat Pl G Uf
    Sm90MeasuredI3.lamStar (7 / 8) (1 / 2) _ ((1 / 2) ^ 40) δbind δzk nπ sm90i_isCert sm90i_wf
    (by decide) (by norm_num) (by norm_num) hΛ hΛ1 hΛ2 hδpos hSplit h1 hW hSR
    (fun _ u => u.elim) hN hF (sm90i_fillOpt Pl hSplit) hG hPinG hL hWL hPS
  refine le_trans (prob_mono ?_) hmain
  rintro x ⟨ha, hlt⟩
  exact ⟨ha, lt_of_lt_of_le hlt (Whole.headline_of_branches Sm90MeasuredI3.model Sm90Measured.flat
    Sm90MeasuredI3.lamStar sm90i_isCert sm90i_wf Sm90.checks sm90_S _ hδlo _
    (sm90i_folded _ hδlo) sm90i_unfolded)⟩

end ResourceAccounting.SecurityProofs.TS
