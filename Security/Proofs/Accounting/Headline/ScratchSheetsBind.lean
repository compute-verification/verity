import Proofs.Accounting.Headline.FpG1
import Proofs.Accounting.Headline.Sm120Measured
import Proofs.Accounting.HeadlineDc.Sm90Measured

/-!
Scratch (CMP-392 plan, never landed): the two general headline statements in the binding form, the sheet one and the
measured one, and three of today's statements at numbers as their corollaries: FP8-g1's (binding form, a premise more),
sm_120's measured one (fork form, (I3) `ScalarRoute` taken) and the H200's measured one (fork form, A6 `SheetWords`
taken).
-/

namespace Accounting.TS.Scratch

open Accounting.TS Accounting.TS.Headline Accounting.Assumptions.TS Pouw.RelativeSecurity Accounting.SecurityProofs.TS
  Accounting.SecurityProofs.TS.Headline Accounting.SecurityProofs.TS.HeadlineDc Pouw.FpG1

/-- What the sheet headline asks of a table. -/
def SheetTable {K P T L U F X : Type} [Fintype K] [Fintype P] [DecidableEq P] [Fintype T] [Fintype L] [Fintype U]
    [Fintype F] [Fintype X] [Nonempty F] [Nonempty X] (Sh : Sheet K P) (Z : Model K (SheetRow P) T L U F X)
    (Fl : Labelled K) (lam : SheetRow P → ℝ) : Prop :=
  Z.G = Sh.machine ∧ IsCert lam ∧ Z.Wf Fl ∧ Z.q ≤ 1 / 2 ∧ Z.qT ≤ 1 / 2 ∧ Z.bT ≤ 0

/-- What the measured headline asks of a table. -/
def MeasuredTable {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U]
    [Fintype F] [Fintype X] [Nonempty F] [Nonempty X] (Z : Model K R T L U F X) (Fl : Labelled K) (lam : R → ℝ) :
    Prop :=
  IsCert lam ∧ Z.Wf Fl

/-- The floor a table gives, at `p = 7/8`, `p₀ = ½`, `S` checks and `δ′`. -/
noncomputable def sheetFloor {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U]
    [Fintype F] [Fintype X] [Nonempty F] [Nonempty X] (Z : Model K R T L U F X) (Fl : Labelled K) (lam : R → ℝ)
    (S : ℕ) (δ' : ℝ) : ℝ :=
  1 - Z.epsSkip Fl lam - Z.epsCert Fl lam - Z.epsRoute Fl lam - Z.epsMand Fl lam -
    max (Z.kappa Fl lam / Real.log (1 / (7 / 8)))
      ((Z.kappa Fl lam + (1 - Z.muF) * Z.Nw * Z.phiStar lam / Z.bnd Fl) / Real.log (1 / (1 / 2))) *
        Real.log (1 / δ') / S

/-- The sheet headline at any table, binding at `δ_bind`. -/
def SheetHeadlineBind : Prop :=
  ∀ {K P T L U F X : Type} [Fintype K] [Fintype P] [DecidableEq P] [Fintype T] [Fintype L] [Fintype U] [Fintype F]
    [Fintype X] [Nonempty F] [Nonempty X] (Sh : Sheet K P) (Z : Model K (SheetRow P) T L U F X) (Fl : Labelled K)
    (lam : SheetRow P → ℝ) (S : ℕ) (Ω : Type) [Fintype Ω] (Pl : Play Ω S K T L U) (G : ReceiptGame Ω S)
    (Uf : Unfolded Ω S) {V : Type} {nπ : ℕ} (pf : ProofLayer Ω S nπ V) (δ δbind δzk : ℝ),
    SheetTable Sh Z Fl lam → 0 < S → 0 < δ - S * (δbind + nπ * δzk) → (∀ x, (Pl.run x).Split) →
    (∀ x, SheetClock Sh (Pl.run x).run) → (∀ x, SheetDispatch (Pl.run x).run) →
    (∀ x, SheetTensor Sh (Pl.run x).run) → (∀ x, SheetPipe Sh (Pl.run x).run) →
    (∀ x, SheetSmem Sh (Pl.run x).run) → (∀ x, SheetWords Z (Pl.run x)) →
    (∀ x, SheetSemantics Z (Pl.run x)) →
    (∀ x, FoldOptFoldedAt Z (Pl.run x) ((1 - Uf.share x / S) * Z.Nw) lam) → (∀ x, FillOpt Z (Pl.run x)) →
    Pl.game = G.game → G.Pinned → OperandHMu (3 / 4) G → ReceiptLuckAt (1 / 2) G → WholeLuckAt (1 / 2) G Uf →
    pf.Of Pl → pf.BindingAt δbind → ProofSoundAt pf δzk → ProofsPin pf → RightFixed Pl pf Z.Nw →
    prob (fun x => Pl.accepts x ∧ (Pl.run x).run.time / Z.credit Fl <
      sheetFloor Z Fl lam S (δ - S * (δbind + nπ * δzk))) ≤ δ

theorem sheetHeadlineBind : SheetHeadlineBind := by
  intro K P T L U F X _ _ _ _ _ _ _ _ _ _ Sh Z Fl lam S Ω _ Pl G Uf V nπ pf δ δbind δzk hT hS hδ hSplit h1 h2 h3 h4
    h5 h6 h7 hF hX hG hPinG hOp hRL hWL hOf hB hZK hPin hRF
  obtain ⟨hGm, hC, hZ, hq, hqT, hb⟩ := hT
  have hL := luck_of_receipt_half Pl.game G hG hPinG hOp hRL
  have hPS := ProofsSoundOfPinned Ω S nπ Pl pf Z.Nw δbind δzk hOf hB hZK hPin hRF
  obtain ⟨hκ0, -⟩ := Whole.kappa_f_nonneg Z Fl lam hC hZ
  obtain ⟨hΛ, hΛ1, hΛ2⟩ := Whole.max_budget
    (f := (1 - Z.muF) * Z.Nw * Z.phiStar lam / Z.bnd Fl) hκ0
    (Real.log_pos (by norm_num) : 0 < Real.log (1 / (7 / 8 : ℝ)))
    (Real.log_pos (by norm_num) : 0 < Real.log (1 / (1 / 2 : ℝ)))
  unfold sheetFloor
  exact Whole.sheetDIdealFloor_whole Sh Z Fl Pl G Uf lam (7 / 8) (1 / 2) _ δ δbind δzk nπ hGm hC hZ hS
    (by norm_num) (by norm_num) hΛ hΛ1 hΛ2 hδ hSplit hq hqT hb h1 h2 h3 h4 h5 h6 h7 hF hX hG hPinG hL hWL hPS

/-- The measured headline at any table, binding at `δ_bind`. -/
def MeasuredHeadlineBind : Prop :=
  ∀ {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F]
    [Fintype X] [Nonempty R] [Nonempty F] [Nonempty X] (Z : Model K R T L U F X) (Fl : Labelled K)
    (lam : R → ℝ) (S : ℕ) (Ω : Type) [Fintype Ω] (Pl : Play Ω S K T L U) (G : ReceiptGame Ω S)
    (Uf : Unfolded Ω S) {V : Type} {nπ : ℕ} (pf : ProofLayer Ω S nπ V) (δ δbind δzk : ℝ),
    MeasuredTable Z Fl lam → 0 < S → 0 < δ - S * (δbind + nπ * δzk) → (∀ x, (Pl.run x).Split) →
    (∀ x, FacetCeilings Z.G (Pl.run x).run) → (∀ x, WordsWritten Z (Pl.run x)) →
    (∀ x, ScalarRoute Z (Pl.run x)) → (∀ x, SideUnits Z (Pl.run x)) → (∀ x, NoSum (Pl.run x)) →
    (∀ x, FoldOptFoldedAt Z (Pl.run x) ((1 - Uf.share x / S) * Z.Nw) lam) → (∀ x, FillOpt Z (Pl.run x)) →
    Pl.game = G.game → G.Pinned → OperandHMu (3 / 4) G → ReceiptLuckAt (1 / 2) G → WholeLuckAt (1 / 2) G Uf →
    pf.Of Pl → pf.BindingAt δbind → ProofSoundAt pf δzk → ProofsPin pf → RightFixed Pl pf Z.Nw →
    prob (fun x => Pl.accepts x ∧ (Pl.run x).run.time / Z.credit Fl <
      sheetFloor Z Fl lam S (δ - S * (δbind + nπ * δzk))) ≤ δ

theorem measuredHeadlineBind : MeasuredHeadlineBind := by
  intro K R T L U F X _ _ _ _ _ _ _ _ _ _ Z Fl lam S Ω _ Pl G Uf V nπ pf δ δbind δzk hT hS hδ hSplit h1 hW hSR hU hN
    hF hX hG hPinG hOp hRL hWL hOf hB hZK hPin hRF
  obtain ⟨hC, hZ⟩ := hT
  have hL := luck_of_receipt_half Pl.game G hG hPinG hOp hRL
  have hPS := ProofsSoundOfPinned Ω S nπ Pl pf Z.Nw δbind δzk hOf hB hZK hPin hRF
  obtain ⟨hκ0, -⟩ := Whole.kappa_f_nonneg Z Fl lam hC hZ
  obtain ⟨hΛ, hΛ1, hΛ2⟩ := Whole.max_budget
    (f := (1 - Z.muF) * Z.Nw * Z.phiStar lam / Z.bnd Fl) hκ0
    (Real.log_pos (by norm_num) : 0 < Real.log (1 / (7 / 8 : ℝ)))
    (Real.log_pos (by norm_num) : 0 < Real.log (1 / (1 / 2 : ℝ)))
  unfold sheetFloor
  exact Whole.dIdealFloor_whole Z Fl Pl G Uf lam (7 / 8) (1 / 2) _ δ δbind δzk nπ hC hZ hS
    (by norm_num) (by norm_num) hΛ hΛ1 hΛ2 hδ hSplit h1 hW hSR hU hN hF hX hG hPinG hL hWL hPS

theorem sm120_table : SheetTable Sm120.sheet Sm120.model Sm120.flat Sm120.piStar :=
  ⟨rfl, isCert_piStar, sm120_wf, le_of_eq sm120_q, le_of_eq sm120_qT, le_rfl⟩

/-- FP8-g1's statement at numbers: the general sheet statement at sm_120's table; its g1 premise is not used. -/
theorem sm120FpG1_of_general : Accounting.Guarantees.FpG1.Sm120FpG1SheetHeadline := by
  intro Ω _ Pl G Uf W nπ pf δbind δzk _ hδ hSplit h1 h2 h3 h4 h5 h6 h7 hF hG hPinG hOp hRL hWL hOf hB hZK hPin hRF
  have hδlo : (1 / 2 : ℝ) ^ 41 ≤
      (1 / 2) ^ 40 - (Sm120.checks : ℝ) * (δbind + ((nπ + 1 : ℕ) : ℝ) * δzk) := by
    have e : ((1 : ℝ) / 2) ^ 40 = 2 * (1 / 2) ^ 41 := by norm_num
    linarith
  have h41 : (0 : ℝ) < (1 / 2) ^ 41 := by positivity
  have hmain := sheetHeadlineBind Sm120.sheet Sm120.model Sm120.flat Sm120.piStar Sm120.checks Ω Pl G Uf pf
    ((1 / 2) ^ 40) δbind δzk sm120_table (by decide) (lt_of_lt_of_le h41 hδlo) hSplit h1 h2 h3 h4 h5 h6 h7 hF
    (sm120_fillOpt Pl hSplit) hG hPinG hOp hRL hWL hOf hB hZK hPin hRF
  refine le_trans (prob_mono ?_) hmain
  rintro x ⟨ha, hlt⟩
  exact ⟨ha, lt_of_lt_of_le hlt (Whole.headline_of_branches Sm120.model Sm120.flat Sm120.piStar
    isCert_piStar sm120_wf Sm120.checks sm120_S _ hδlo _ (le_trans (by norm_num) (sm120_eps_le _ hδlo))
    sm120_unfolded)⟩

/-- sm_120's measured statement at numbers: the general measured statement at VF3s-16's table, binding at
`√δ_fork`. -/
theorem sm120Measured_of_general : Accounting.Guarantees.TS.Sm120MeasuredHeadline := by
  intro Ω _ Pl G Uf V nπ pf δfork δzk hδ hSplit h1 hW hSR hN hF hG hPinG hOp hRL hWL hOf hB hZK hPin hRF
  have hδlo : (1 / 2 : ℝ) ^ 41 ≤ (1 / 2) ^ 40 - (Sm120.checks : ℝ) * (Real.sqrt δfork + nπ * δzk) := by
    have e : ((1 : ℝ) / 2) ^ 40 = 2 * (1 / 2) ^ 41 := by norm_num
    linarith
  have h41 : (0 : ℝ) < (1 / 2) ^ 41 := by positivity
  have hmain := measuredHeadlineBind Sm120Measured.model Sm120Measured.flat (onFacet Sm120Measured.piStar)
    Sm120.checks Ω Pl G Uf pf ((1 / 2) ^ 40) (Real.sqrt δfork) δzk ⟨isCert_onFacet _, sm120m_wf⟩ (by decide)
    (lt_of_lt_of_le h41 hδlo) hSplit h1 hW hSR (fun _ u => u.elim) hN hF (sm120m_fillOpt Pl hSplit) hG hPinG hOp
    hRL hWL hOf (Whole.moved_le_sqrt pf δfork hB) hZK hPin hRF
  refine le_trans (prob_mono ?_) hmain
  rintro x ⟨ha, hlt⟩
  exact ⟨ha, lt_of_lt_of_le hlt (Whole.headline_of_branches Sm120Measured.model Sm120Measured.flat
    (onFacet Sm120Measured.piStar) (isCert_onFacet _) sm120m_wf Sm120.checks sm120_S _ hδlo _
    (sm120m_eps_le _ hδlo) sm120m_unfolded)⟩

/-- The H200's measured statement at numbers: the general measured statement at DC-90-ri's table, (I2)/(I4) and
(I3) from A6 at the table's half floors. -/
theorem sm90Measured_of_general : Accounting.Guarantees.TS.Sm90MeasuredHeadline := by
  intro Ω _ Pl G Uf V nπ pf δfork δzk hδ hSplit h1 h6 hN hF hG hPinG hOp hRL hWL hOf hB hZK hPin hRF
  have hδlo : (1 / 2 : ℝ) ^ 41 ≤ (1 / 2) ^ 40 - (Sm90.checks : ℝ) * (Real.sqrt δfork + nπ * δzk) := by
    have e : ((1 : ℝ) / 2) ^ 40 = 2 * (1 / 2) ^ 41 := by norm_num
    linarith
  have h41 : (0 : ℝ) < (1 / 2) ^ 41 := by positivity
  have hmain := measuredHeadlineBind Sm90Measured.model Sm90Measured.flat Sm90Measured.lamStar
    Sm90.checks Ω Pl G Uf pf ((1 / 2) ^ 40) (Real.sqrt δfork) δzk ⟨sm90m_isCert, sm90m_wf⟩ (by decide)
    (lt_of_lt_of_le h41 hδlo) hSplit h1 (fun x => (h6 x).1)
    (fun x => scalarRoute_half Sm90Measured.model _ (hSplit x) (h6 x) (le_of_eq sm90m_q) (le_of_eq sm90m_qT)
      le_rfl)
    (fun _ u => u.elim) hN hF (sm90m_fillOpt Pl hSplit) hG hPinG hOp
    hRL hWL hOf (Whole.moved_le_sqrt pf δfork hB) hZK hPin hRF
  refine le_trans (prob_mono ?_) hmain
  rintro x ⟨ha, hlt⟩
  exact ⟨ha, lt_of_lt_of_le hlt (Whole.headline_of_branches Sm90Measured.model Sm90Measured.flat
    Sm90Measured.lamStar sm90m_isCert sm90m_wf Sm90.checks sm90_S _ hδlo _
    (sm90m_folded _ hδlo) sm90m_unfolded)⟩

end Accounting.TS.Scratch

#print axioms Accounting.TS.Scratch.sheetHeadlineBind
#print axioms Accounting.TS.Scratch.measuredHeadlineBind
#print axioms Accounting.TS.Scratch.sm120FpG1_of_general
#print axioms Accounting.TS.Scratch.sm120Measured_of_general
#print axioms Accounting.TS.Scratch.sm90Measured_of_general
