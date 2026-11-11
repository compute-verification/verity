import Proofs.Accounting.Headline.Sm120

/-!
Scratch (CMP-392 plan, never landed): the sheet headline with the data sheet, the model, the floor program and the
certificate as parameters, and `Sm120SheetHeadline` as its corollary.
-/

namespace Accounting.TS.Scratch

open Accounting.TS Accounting.TS.Headline Accounting.Assumptions.TS Pouw.RelativeSecurity Accounting.SecurityProofs.TS
  Accounting.SecurityProofs.TS.Headline

/-- What the headline asks of a table: the model is on the data sheet's machine, `lam` is a certificate, the model is
well formed at the floor program, and its scalar route is at the half. -/
def SheetTable {K P T L U F X : Type} [Fintype K] [Fintype P] [DecidableEq P] [Fintype T] [Fintype L] [Fintype U]
    [Fintype F] [Fintype X] [Nonempty F] [Nonempty X] (Sh : Sheet K P) (Z : Model K (SheetRow P) T L U F X)
    (Fl : Labelled K) (lam : SheetRow P → ℝ) : Prop :=
  Z.G = Sh.machine ∧ IsCert lam ∧ Z.Wf Fl ∧ Z.q ≤ 1 / 2 ∧ Z.qT ≤ 1 / 2 ∧ Z.bT ≤ 0

/-- The floor a table gives, at `p = 7/8`, `p₀ = ½`, `S` checks and `δ′`. -/
noncomputable def sheetFloor {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U]
    [Fintype F] [Fintype X] [Nonempty F] [Nonempty X] (Z : Model K R T L U F X) (Fl : Labelled K) (lam : R → ℝ)
    (S : ℕ) (δ' : ℝ) : ℝ :=
  1 - Z.epsSkip Fl lam - Z.epsCert Fl lam - Z.epsRoute Fl lam - Z.epsMand Fl lam -
    max (Z.kappa Fl lam / Real.log (1 / (7 / 8)))
      ((Z.kappa Fl lam + (1 - Z.muF) * Z.Nw * Z.phiStar lam / Z.bnd Fl) / Real.log (1 / (1 / 2))) *
        Real.log (1 / δ') / S

/-- The sheet headline at any table. -/
def SheetHeadlineAt : Prop :=
  ∀ {K P T L U F X : Type} [Fintype K] [Fintype P] [DecidableEq P] [Fintype T] [Fintype L] [Fintype U] [Fintype F]
    [Fintype X] [Nonempty F] [Nonempty X] (Sh : Sheet K P) (Z : Model K (SheetRow P) T L U F X) (Fl : Labelled K)
    (lam : SheetRow P → ℝ) (S : ℕ) (Ω : Type) [Fintype Ω] (Pl : Play Ω S K T L U) (G : ReceiptGame Ω S)
    (Uf : Unfolded Ω S) {V : Type} {nπ : ℕ} (pf : ProofLayer Ω S nπ V) (δ δfork δzk : ℝ),
    SheetTable Sh Z Fl lam → 0 < S → 0 < δ - S * (Real.sqrt δfork + nπ * δzk) → (∀ x, (Pl.run x).Split) →
    (∀ x, SheetClock Sh (Pl.run x).run) → (∀ x, SheetDispatch (Pl.run x).run) →
    (∀ x, SheetTensor Sh (Pl.run x).run) → (∀ x, SheetPipe Sh (Pl.run x).run) →
    (∀ x, SheetSmem Sh (Pl.run x).run) → (∀ x, SheetWords Z (Pl.run x)) →
    (∀ x, SheetSemantics Z (Pl.run x)) →
    (∀ x, FoldOptFoldedAt Z (Pl.run x) ((1 - Uf.share x / S) * Z.Nw) lam) → (∀ x, FillOpt Z (Pl.run x)) →
    Pl.game = G.game → G.Pinned → OperandHMu (3 / 4) G → ReceiptLuckAt (1 / 2) G → WholeLuckAt (1 / 2) G Uf →
    pf.Of Pl → BindingForkAt pf δfork → ProofSoundAt pf δzk → ProofsPin pf → RightFixed Pl pf Z.Nw →
    prob (fun x => Pl.accepts x ∧ (Pl.run x).run.time / Z.credit Fl <
      sheetFloor Z Fl lam S (δ - S * (Real.sqrt δfork + nπ * δzk))) ≤ δ

theorem sheetHeadlineAt : SheetHeadlineAt := by
  intro K P T L U F X _ _ _ _ _ _ _ _ _ _ Sh Z Fl lam S Ω _ Pl G Uf V nπ pf δ δfork δzk hT hS hδ hSplit h1 h2 h3 h4
    h5 h6 h7 hF hX hG hPinG hOp hRL hWL hOf hB hZK hPin hRF
  obtain ⟨hGm, hC, hZ, hq, hqT, hb⟩ := hT
  have hL := luck_of_receipt_half Pl.game G hG hPinG hOp hRL
  have hPS := ProofsSoundOfPinned Ω S nπ Pl pf Z.Nw (Real.sqrt δfork) δzk hOf (Whole.moved_le_sqrt pf δfork hB) hZK
    hPin hRF
  obtain ⟨hκ0, -⟩ := Whole.kappa_f_nonneg Z Fl lam hC hZ
  obtain ⟨hΛ, hΛ1, hΛ2⟩ := Whole.max_budget
    (f := (1 - Z.muF) * Z.Nw * Z.phiStar lam / Z.bnd Fl) hκ0
    (Real.log_pos (by norm_num) : 0 < Real.log (1 / (7 / 8 : ℝ)))
    (Real.log_pos (by norm_num) : 0 < Real.log (1 / (1 / 2 : ℝ)))
  unfold sheetFloor
  exact Whole.sheetDIdealFloor_whole Sh Z Fl Pl G Uf lam (7 / 8) (1 / 2) _ δ (Real.sqrt δfork) δzk nπ hGm hC hZ hS
    (by norm_num) (by norm_num) hΛ hΛ1 hΛ2 hδ hSplit hq hqT hb h1 h2 h3 h4 h5 h6 h7 hF hX hG hPinG hL hWL hPS

/-- What the headline takes from sm_120's table. -/
theorem sm120_table : SheetTable Sm120.sheet Sm120.model Sm120.flat Sm120.piStar :=
  ⟨rfl, isCert_piStar, sm120_wf, le_of_eq sm120_q, le_of_eq sm120_qT, le_rfl⟩

/-- The listed statement at numbers, as a corollary: the general statement at sm_120's table, the table's floor
bounded by `1 − 5860/100000` (`Whole.headline_of_branches` with the two data lemmas), FILL-OPT from the data. -/
theorem sm120SheetHeadline_of_general : Accounting.Guarantees.TS.Sm120SheetHeadline := by
  intro Ω _ Pl G Uf V nπ pf δfork δzk hδ hSplit h1 h2 h3 h4 h5 h6 h7 hF hG hPinG hOp hRL hWL hOf hB hZK hPin hRF
  have hδlo : (1 / 2 : ℝ) ^ 41 ≤ (1 / 2) ^ 40 - (Sm120.checks : ℝ) * (Real.sqrt δfork + nπ * δzk) := by
    have e : ((1 : ℝ) / 2) ^ 40 = 2 * (1 / 2) ^ 41 := by norm_num
    linarith
  have h41 : (0 : ℝ) < (1 / 2) ^ 41 := by positivity
  have hδpos : 0 < (1 / 2 : ℝ) ^ 40 - (Sm120.checks : ℝ) * (Real.sqrt δfork + nπ * δzk) := lt_of_lt_of_le h41 hδlo
  have hmain := sheetHeadlineAt Sm120.sheet Sm120.model Sm120.flat Sm120.piStar Sm120.checks Ω Pl G Uf pf
    ((1 / 2) ^ 40) δfork δzk sm120_table (by decide) hδpos hSplit h1 h2 h3 h4 h5 h6 h7 hF
    (sm120_fillOpt Pl hSplit) hG hPinG hOp hRL hWL hOf hB hZK hPin hRF
  refine le_trans (prob_mono ?_) hmain
  rintro x ⟨ha, hlt⟩
  exact ⟨ha, lt_of_lt_of_le hlt (Whole.headline_of_branches Sm120.model Sm120.flat Sm120.piStar
    isCert_piStar sm120_wf Sm120.checks sm120_S _ hδlo _ (le_trans (by norm_num) (sm120_eps_le _ hδlo))
    sm120_unfolded)⟩

end Accounting.TS.Scratch

#print axioms Accounting.TS.Scratch.sheetHeadlineAt
#print axioms Accounting.TS.Scratch.sm120SheetHeadline_of_general
