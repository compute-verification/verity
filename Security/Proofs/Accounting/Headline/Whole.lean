import Proofs.Accounting.TS

/-!
# Theorem D-ideal with fold skippers charged

`DIdealFloor` with FOLD-OPT per word folded at the certificate it prices (`FoldOptFoldedAt`)
and the whole-unit luck (`WholeLuckAt`), the form the headlines take. A round's saving is
`κ·d_w + f·e_w`, `d_w` its deficient share and `e_w` the share of its words whose fold the run skips, with
`κ = N_w w*/B_♭` and `f = (1 − μ_f)N_w φ*/B_♭`. The tail bound is
PoUW's exponential round bound (`prob_accepts_le`, `c = 1`) on the log-luck game `wgame`, whose round weighs
`ln(1/p₀)` when its unfolded share is positive, else `ln(1/p)` when it is deficient, else `0`:

* `wgame_expBound`: the weight game has `E[e^W·1{pass}] ≤ 1` a round, from the pinned openings and the whole-unit luck
  when `e_w > 0` (`Pr[pass] ≤ Pr[same] ≤ p₀`), else from (L) at `p`;
* `saving_le`: along every path `κ·Σd + f·Σe ≤ Λ·ΣW` when `κ ≤ Λ·ln(1/p)` and `κ + f ≤ Λ·ln(1/p₀)`;
* `dIdealFloor_whole`: outside `δ`, every accepted window has `T/T_cred ≥ 1 − ε_skip − ε_cert − ε_route − ε_mand −
  Λ·ln(1/δ′)/S`; `sheetDIdealFloor_whole` is it on the data-sheet machine;
* `max_budget`, `unfolded_floor` and `headline_of_branches`: the headlines' `Λ = max(κ/ln(1/p), (κ + f)/ln(1/p₀))`,
  the unfolded branch's floor `(N_w w* + (1 − μ_f)N_w φ*)/B_♭ · (1 − 41/48000)` at `p₀ = ½`, `S = 48,000` and
  `δ′ ≥ 2^-41`, and the headline's last step, a bound below both branches' floors.
-/

namespace Accounting.SecurityProofs.TS.Whole

open Accounting.TS Accounting.Assumptions.TS Accounting.SecurityProofs.TS Pouw.RelativeSecurity

variable {Ω : Type}

/-- The log-luck game: round `w` weighs `ln(1/p₀)` if its unfolded share is positive, else `ln(1/p)` if its share is,
else `0`, with the receipt game's checks. -/
noncomputable def wgame (p p0 : ℝ) : {n : ℕ} → ReceiptGame Ω n → Unfolded Ω n → RoundGame Ω n
  | _, .done, .done => .done
  | _, .round d pass _ _ next, .round e vnext =>
      .round (fun ω => if 0 < e then Real.log (1 / p0) else if 0 < d ω then Real.log (1 / p) else 0) pass
        fun ω => wgame p p0 (next ω) (vnext ω)

theorem wgame_accepts (p p0 : ℝ) : ∀ {n} (G : ReceiptGame Ω n) (Uf : Unfolded Ω n) (x : Paths Ω n),
    (wgame p p0 G Uf).accepts x ↔ G.game.accepts x
  | _, .done, .done, _ => by simp [wgame, ReceiptGame.game, RoundGame.accepts]
  | _, .round d pass _ _ next, .round e vnext, x => by
    simp only [wgame, ReceiptGame.game, RoundGame.accepts]
    exact and_congr_right fun _ => wgame_accepts p p0 (next x.1) (vnext x.1) x.2

/-- The log-luck game's round bound: `E[e^W·1{pass}] ≤ 1` at every round, whatever the past. -/
theorem wgame_expBound [Fintype Ω] {p p0 : ℝ} (hp : 0 < p) (hp0 : 0 < p0) :
    ∀ {n} (G : ReceiptGame Ω n) (Uf : Unfolded Ω n), G.Pinned → Accounting.TS.RoundGame.LuckAt p G.game →
      G.WholeLuck p0 Uf → (wgame p p0 G Uf).ExpBound 1
  | _, .done, .done, _, _, _ => trivial
  | _, .round d pass right same next, .round e vnext, hPin, hL, hW => by
    classical
    obtain ⟨hPass, hPinN⟩ := hPin
    obtain ⟨-, hLk, hLN⟩ := hL
    obtain ⟨-, hWe, hWN⟩ := hW
    refine ⟨?_, fun a => wgame_expBound hp hp0 (next a) (vnext a) (hPinN a) (hLN a) (hWN a)⟩
    try dsimp only
    by_cases he0 : 0 < e
    · refine le_trans (expect_mono (g := fun a => 1 / p0 * if same a then (1 : ℝ) else 0) fun a => ?_) ?_
      · try dsimp only
        rw [if_pos he0, one_mul]
        by_cases hpa : pass a
        · rw [if_pos hpa, if_pos (hPass a hpa), mul_one, Real.exp_log (one_div_pos.mpr hp0)]
        · rw [if_neg hpa]
          exact mul_nonneg (one_div_pos.mpr hp0).le (by split_ifs <;> norm_num)
      · rw [expect_mul_left, ← prob_eq_expect]
        calc 1 / p0 * prob same ≤ 1 / p0 * p0 :=
              mul_le_mul_of_nonneg_left (hWe he0) (one_div_pos.mpr hp0).le
          _ = 1 := one_div_mul_cancel hp0.ne'
    · refine le_trans (expect_mono (g := fun a => 1 / p * (if pass a ∧ 0 < d a then (1 : ℝ) else 0) +
          (if ¬0 < d a then (1 : ℝ) else 0)) fun a => ?_) ?_
      · try dsimp only
        rw [if_neg he0]
        by_cases hpa : pass a
        · by_cases hda : 0 < d a
          · rw [if_pos hpa, if_pos hda, if_pos (And.intro hpa hda), if_neg (not_not.mpr hda), one_mul, mul_one,
              add_zero, Real.exp_log (one_div_pos.mpr hp)]
          · rw [if_pos hpa, if_neg hda, if_neg (show ¬(pass a ∧ 0 < d a) from fun h => hda h.2), if_pos hda,
              mul_zero, mul_zero, zero_add, Real.exp_zero]
        · rw [if_neg hpa]
          exact add_nonneg (mul_nonneg (one_div_pos.mpr hp).le (by split_ifs <;> norm_num))
            (by split_ifs <;> norm_num)
      · rw [Rounds.expect_add, expect_mul_left, ← prob_eq_expect, ← prob_eq_expect]
        calc 1 / p * prob (fun a => pass a ∧ 0 < d a) + prob (fun a => ¬0 < d a)
            ≤ 1 / p * (p * prob (fun a => 0 < d a)) + prob (fun a => ¬0 < d a) := by gcongr
          _ = prob (fun a => 0 < d a) + prob (fun a => ¬0 < d a) := by
              rw [← mul_assoc, one_div_mul_cancel hp.ne', one_mul]
          _ ≤ 1 := prob_add_not_le _

/-- A round's saving is at most `Λ` times its weight: along every path, `κ·Σd + f·Σe ≤ Λ·ΣW`. -/
theorem saving_le {p p0 κ f Λ : ℝ} [Fintype Ω] (hκ : 0 ≤ κ) (hf : 0 ≤ f) (h1 : κ ≤ Λ * Real.log (1 / p))
    (h2 : κ + f ≤ Λ * Real.log (1 / p0)) :
    ∀ {n} (G : ReceiptGame Ω n) (Uf : Unfolded Ω n), Accounting.TS.RoundGame.LuckAt p G.game →
      G.WholeLuck p0 Uf → ∀ x, κ * G.game.deficiency x + f * Uf.share x ≤ Λ * (wgame p p0 G Uf).deficiency x
  | _, .done, .done, _, _, x => by simp [ReceiptGame.game, RoundGame.deficiency, Unfolded.share, wgame]
  | _, .round d pass right same next, .round e vnext, hL, hW, x => by
    obtain ⟨hd, -, hLN⟩ := hL
    obtain ⟨he, -, hWN⟩ := hW
    have ih := saving_le hκ hf h1 h2 (next x.1) (vnext x.1) (hLN x.1) (hWN x.1) x.2
    have hround : κ * d x.1 + f * e ≤
        Λ * (if 0 < e then Real.log (1 / p0) else if 0 < d x.1 then Real.log (1 / p) else 0) := by
      obtain ⟨hd0, hd1⟩ := hd x.1
      obtain ⟨he0, he1⟩ := he
      split_ifs with hpe hpd
      · have h3 := mul_le_mul_of_nonneg_left hd1 hκ
        have h4 := mul_le_mul_of_nonneg_left he1 hf
        linarith
      · have he' : e = 0 := le_antisymm (not_lt.mp hpe) he0
        rw [he', mul_zero, add_zero]
        have h3 := mul_le_mul_of_nonneg_left hd1 hκ
        linarith
      · have he' : e = 0 := le_antisymm (not_lt.mp hpe) he0
        have hd' : d x.1 = 0 := le_antisymm (not_lt.mp hpd) hd0
        rw [he', hd']
        simp
    simp only [ReceiptGame.game, RoundGame.deficiency, Unfolded.share, wgame]
    linarith

/-- **Theorem D-ideal (2.7♭) with fold skippers charged**: `DIdealFloor` with FOLD-OPT per word folded at `π`, each
run priced at the words it folds, `(1 − Σe/S)·N_w`, and the whole-unit luck at `p₀` beside (L) at `p`, for any `Λ ≥ 0`
with `κ ≤ Λ·ln(1/p)` and `κ + f ≤ Λ·ln(1/p₀)`: outside an event of probability at most `δ`, every accepted window has
`T/T_cred ≥ 1 − ε_skip − ε_cert − ε_route − ε_mand − Λ·ln(1/δ′)/S`. -/
theorem dIdealFloor_whole {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U]
    [Fintype F] [Fintype X] [Nonempty R] [Nonempty F] [Nonempty X]
    (Z : Model K R T L U F X) (Fl : Labelled K) [Fintype Ω] {S : ℕ} (P : Play Ω S K T L U)
    (G : ReceiptGame Ω S) (Uf : Unfolded Ω S) (lam : R → ℝ) (p p0 Λ δ δbind δzk : ℝ) (nπ : ℕ)
    (h : IsCert lam) (hZ : Z.Wf Fl) (hS : 0 < S) (hp : 0 < p) (hp0 : 0 < p0) (hΛ : 0 ≤ Λ)
    (hΛ1 : Z.kappa Fl lam ≤ Λ * Real.log (1 / p))
    (hΛ2 : Z.kappa Fl lam + (1 - Z.muF) * Z.Nw * Z.phiStar lam / Z.bnd Fl ≤ Λ * Real.log (1 / p0))
    (hδ : 0 < δ - S * (δbind + nπ * δzk)) (hSplit : ∀ x, (P.run x).Split)
    (hM : ∀ x, FacetCeilings Z.G (P.run x).run) (hW : ∀ x, WordsWritten Z (P.run x))
    (hSR : ∀ x, ScalarRoute Z (P.run x)) (hU : ∀ x, SideUnits Z (P.run x)) (hN : ∀ x, NoSum (P.run x))
    (hF : ∀ x, FoldOptFoldedAt Z (P.run x) ((1 - Uf.share x / S) * Z.Nw) lam) (hX : ∀ x, FillOpt Z (P.run x))
    (hG : P.game = G.game) (hPin : G.Pinned) (hL : LuckAt p P.game) (hWL : WholeLuckAt p0 G Uf)
    (hPS : ProofsSound P Z.Nw δbind δzk nπ) :
    prob (fun x => P.accepts x ∧ (P.run x).run.time / Z.credit Fl <
      1 - Z.epsSkip Fl lam - Z.epsCert Fl lam - Z.epsRoute Fl lam - Z.epsMand Fl lam -
        Λ * Real.log (1 / (δ - S * (δbind + nπ * δzk))) / S) ≤ δ := by
  set δ' := δ - S * (δbind + nπ * δzk) with hδ'
  obtain ⟨E, hE, hacc⟩ := hPS
  set t := Real.log (1 / δ') with ht
  have hLG : Accounting.TS.RoundGame.LuckAt p G.game := by
    rw [← hG]
    exact hL
  have hexp := wgame_expBound hp hp0 G Uf hPin hLG hWL
  have het : Real.exp (-(1 * t)) = δ' := by
    rw [one_mul, ht, one_div, Real.log_inv, neg_neg, Real.exp_log hδ]
  have hB := ((wgame p p0 G Uf).prob_accepts_le zero_le_one hexp t).trans het.le
  have hZ' := hZ
  obtain ⟨-, -, hNw, hB0, -, -, -, -, -, -, -, -, hmuF1, -, -⟩ := hZ'
  obtain ⟨-, hw0, hφ0, -⟩ := stars_nonneg Z Fl lam hZ h
  have hκ : 0 ≤ Z.kappa Fl lam := by
    rw [Model.kappa]
    exact div_nonneg (mul_nonneg hNw.le hw0) hB0.le
  have hf : 0 ≤ (1 - Z.muF) * Z.Nw * Z.phiStar lam / Z.bnd Fl :=
    div_nonneg (mul_nonneg (mul_nonneg (sub_nonneg.mpr hmuF1) hNw.le) hφ0) hB0.le
  have hS' : (0 : ℝ) < S := Nat.cast_pos.mpr hS
  rw [hG] at hacc
  calc prob (fun x => P.accepts x ∧ (P.run x).run.time / Z.credit Fl <
        1 - Z.epsSkip Fl lam - Z.epsCert Fl lam - Z.epsRoute Fl lam - Z.epsMand Fl lam - Λ * t / S)
      ≤ prob (fun x => E x ∨ ((wgame p p0 G Uf).accepts x ∧ t ≤ (wgame p p0 G Uf).deficiency x)) := by
        refine prob_mono ?_
        rintro x ⟨ha, hlt⟩
        by_contra hne
        push Not at hne
        obtain ⟨hgame, hfix⟩ := hacc x ha hne.1
        have hDw := hne.2 ((wgame_accepts p p0 G Uf x).mpr hgame)
        have hsv := saving_le hκ hf hΛ1 hΛ2 G Uf hLG hWL x
        have hΛt := mul_le_mul_of_nonneg_left hDw.le hΛ
        have hdiv := div_le_div_of_nonneg_right (hsv.trans hΛt) hS'.le
        have hprice := price_of_parts_folded Z Fl (P.run x) lam ((1 - Uf.share x / S) * Z.Nw) h hZ (hSplit x)
          (hW x) (hSR x) (hU x) (hN x) (hF x) (hX x)
        have hfixw := mul_le_mul_of_nonneg_right hfix hw0
        have hΦ : (1 - G.game.deficiency x / S) * Z.Nw * Z.wStar lam +
            (1 - Z.muF) * ((1 - Uf.share x / S) * Z.Nw) * Z.phiStar lam + (1 - Z.muX) * Z.chiStar lam ≤
              Z.G.price lam (P.run x).run.n := by
          linarith
        have hT := price_le_time Z.G lam (P.run x).run h hZ.2.1 (hM x)
        have key := ratio_floor hB0 (hΦ.trans hT)
        rw [Model.credit] at hlt
        have hid := identity Z Fl lam 0 0 hZ
        have hLuck : Z.epsLuck Fl lam 0 = 0 := zero_mul _
        have hAdapt : Z.epsAdapt Fl lam 0 = 0 := zero_mul _
        have hsplit : ((1 - 0 - 0) * Z.Nw * Z.wStar lam + (1 - Z.muF) * Z.Nw * Z.phiStar lam +
              (1 - Z.muX) * Z.chiStar lam) / Z.bnd Fl -
            ((1 - G.game.deficiency x / S) * Z.Nw * Z.wStar lam +
              (1 - Z.muF) * ((1 - Uf.share x / S) * Z.Nw) * Z.phiStar lam + (1 - Z.muX) * Z.chiStar lam) /
                Z.bnd Fl =
            (Z.kappa Fl lam * G.game.deficiency x +
              (1 - Z.muF) * Z.Nw * Z.phiStar lam / Z.bnd Fl * Uf.share x) / S := by
          rw [Model.kappa]
          ring
        linarith
    _ ≤ prob E + prob (fun x => (wgame p p0 G Uf).accepts x ∧ t ≤ (wgame p p0 G Uf).deficiency x) :=
        prob_or_le _ _
    _ ≤ S * (δbind + nπ * δzk) + δ' := add_le_add hE hB
    _ = δ := by rw [hδ']; ring

/-- `dIdealFloor_whole` on the data-sheet machine: its (I1) from `facetCeilings_of_sheet`, (I2)/(I4) and (I3) from A6
(`scalarRoute_half`), (I5) from A7, as `SheetDIdealFloor`. -/
theorem sheetDIdealFloor_whole {K P T L U F X : Type} [Fintype K] [Fintype P] [DecidableEq P] [Fintype T]
    [Fintype L] [Fintype U] [Fintype F] [Fintype X] [Nonempty F] [Nonempty X]
    (Sh : Sheet K P) (Z : Model K (SheetRow P) T L U F X) (Fl : Labelled K) [Fintype Ω] {S : ℕ}
    (Pl : Play Ω S K T L U) (G : ReceiptGame Ω S) (Uf : Unfolded Ω S) (lam : SheetRow P → ℝ)
    (p p0 Λ δ δbind δzk : ℝ) (nπ : ℕ) (hGm : Z.G = Sh.machine) (h : IsCert lam) (hZ : Z.Wf Fl) (hS : 0 < S)
    (hp : 0 < p) (hp0 : 0 < p0) (hΛ : 0 ≤ Λ) (hΛ1 : Z.kappa Fl lam ≤ Λ * Real.log (1 / p))
    (hΛ2 : Z.kappa Fl lam + (1 - Z.muF) * Z.Nw * Z.phiStar lam / Z.bnd Fl ≤ Λ * Real.log (1 / p0))
    (hδ : 0 < δ - S * (δbind + nπ * δzk)) (hSplit : ∀ x, (Pl.run x).Split) (hq : Z.q ≤ 1 / 2)
    (hqT : Z.qT ≤ 1 / 2) (hb : Z.bT ≤ 0) (h1 : ∀ x, SheetClock Sh (Pl.run x).run)
    (h2 : ∀ x, SheetDispatch (Pl.run x).run) (h3 : ∀ x, SheetTensor Sh (Pl.run x).run)
    (h4 : ∀ x, SheetPipe Sh (Pl.run x).run) (h5 : ∀ x, SheetSmem Sh (Pl.run x).run)
    (h6 : ∀ x, SheetWords Z (Pl.run x)) (h7 : ∀ x, SheetSemantics Z (Pl.run x))
    (hF : ∀ x, FoldOptFoldedAt Z (Pl.run x) ((1 - Uf.share x / S) * Z.Nw) lam) (hX : ∀ x, FillOpt Z (Pl.run x))
    (hG : Pl.game = G.game) (hPin : G.Pinned) (hL : LuckAt p Pl.game) (hWL : WholeLuckAt p0 G Uf)
    (hPS : ProofsSound Pl Z.Nw δbind δzk nπ) :
    prob (fun x => Pl.accepts x ∧ (Pl.run x).run.time / Z.credit Fl <
      1 - Z.epsSkip Fl lam - Z.epsCert Fl lam - Z.epsRoute Fl lam - Z.epsMand Fl lam -
        Λ * Real.log (1 / (δ - S * (δbind + nπ * δzk))) / S) ≤ δ :=
  dIdealFloor_whole Z Fl Pl G Uf lam p p0 Λ δ δbind δzk nπ h hZ hS hp hp0 hΛ hΛ1 hΛ2 hδ hSplit
    (fun x => by
      rw [hGm]
      exact facetCeilings_of_sheet Sh _ (run_nonneg_of_split (hSplit x)) (h1 x) (h2 x) (h3 x) (h4 x) (h5 x))
    (fun x => (h6 x).1) (fun x => scalarRoute_half Z _ (hSplit x) (h6 x) hq hqT hb) (fun x => (h7 x).2)
    (fun x => (h7 x).1) hF hX hG hPin hL hWL hPS

/-- Binding at `√δ_fork` from its fork form at `δ_fork` (`BindingOfFork`). -/
theorem moved_le_sqrt {Ω : Type} [Fintype Ω] {S nπ : ℕ} {V : Type} (pf : ProofLayer Ω S nπ V) (δfork : ℝ)
    (hB : BindingForkAt pf δfork) (w : Fin S) : prob (fun x => pf.Moved x w) ≤ Real.sqrt δfork := by
  have h := Accounting.SecurityProofs.TS.BindingOfFork Ω S nπ pf w
  rw [← Real.sqrt_sq (prob_nonneg (fun x => pf.Moved x w))]
  exact Real.sqrt_le_sqrt (le_trans h (hB w))

/-- The headlines' `Λ = max(κ/l, (κ + f)/l₀)` is `≥ 0` and meets both of `dIdealFloor_whole`'s bounds. -/
theorem max_budget {κ f l l0 : ℝ} (hκ : 0 ≤ κ) (hl : 0 < l) (hl0 : 0 < l0) :
    0 ≤ max (κ / l) ((κ + f) / l0) ∧ κ ≤ max (κ / l) ((κ + f) / l0) * l ∧
      κ + f ≤ max (κ / l) ((κ + f) / l0) * l0 :=
  ⟨(div_nonneg hκ hl.le).trans (le_max_left _ _), (div_le_iff₀ hl).mp (le_max_left _ _),
    (div_le_iff₀ hl0).mp (le_max_right _ _)⟩

/-- `κ ≥ 0` and `f ≥ 0` on a well-formed model at a certificate. -/
theorem kappa_f_nonneg {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U]
    [Fintype F] [Fintype X] [Nonempty F] [Nonempty X] (Z : Model K R T L U F X) (Fl : Labelled K) (lam : R → ℝ)
    (h : IsCert lam) (hZ : Z.Wf Fl) :
    0 ≤ Z.kappa Fl lam ∧ 0 ≤ (1 - Z.muF) * Z.Nw * Z.phiStar lam / Z.bnd Fl := by
  have hZ' := hZ
  obtain ⟨-, -, hNw, hB0, -, -, -, -, -, -, -, -, hmuF1, -, -⟩ := hZ'
  obtain ⟨-, hw0, hφ0, -⟩ := stars_nonneg Z Fl lam hZ h
  refine ⟨?_, div_nonneg (mul_nonneg (mul_nonneg (sub_nonneg.mpr hmuF1) hNw.le) hφ0) hB0.le⟩
  rw [Model.kappa]
  exact div_nonneg (mul_nonneg hNw.le hw0) hB0.le

/-- The unfolded branch at `p₀ = ½`, `S = 48,000` and `δ′ ≥ 2^-41`: with `Λ = (κ + f)/ln 2`,
`1 − ε_skip − ε_cert − ε_route − ε_mand − Λ·ln(1/δ′)/S ≥ (N_w w* + (1 − μ_f)N_w φ*)/B_♭ · (1 − 41/48000)`, since
`ln(1/δ′) ≤ 41 ln 2` and `χ* ≥ 0`. -/
theorem unfolded_floor {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U]
    [Fintype F] [Fintype X] [Nonempty F] [Nonempty X] (Z : Model K R T L U F X) (Fl : Labelled K) (lam : R → ℝ)
    (h : IsCert lam) (hZ : Z.Wf Fl) (S : ℕ) (hS : (S : ℝ) = 48000) (δ' : ℝ) (hδ' : (1 / 2 : ℝ) ^ 41 ≤ δ') :
    (Z.Nw * Z.wStar lam + (1 - Z.muF) * Z.Nw * Z.phiStar lam) / Z.bnd Fl * (1 - 41 / 48000) ≤
      1 - Z.epsSkip Fl lam - Z.epsCert Fl lam - Z.epsRoute Fl lam - Z.epsMand Fl lam -
        (Z.kappa Fl lam + (1 - Z.muF) * Z.Nw * Z.phiStar lam / Z.bnd Fl) / Real.log (1 / (1 / 2)) *
          Real.log (1 / δ') / S := by
  have hZ' := hZ
  obtain ⟨-, -, hNw, hB0, -, -, -, -, -, -, -, -, hmuF1, -, hmuX1⟩ := hZ'
  obtain ⟨-, hw0, hφ0, hχ0⟩ := stars_nonneg Z Fl lam hZ h
  have hid := identity Z Fl lam 0 0 hZ
  have hLuck : Z.epsLuck Fl lam 0 = 0 := zero_mul _
  have hAdapt : Z.epsAdapt Fl lam 0 = 0 := zero_mul _
  have hΦ : ((1 - 0 - 0) * Z.Nw * Z.wStar lam + (1 - Z.muF) * Z.Nw * Z.phiStar lam +
        (1 - Z.muX) * Z.chiStar lam) / Z.bnd Fl =
      (Z.Nw * Z.wStar lam + (1 - Z.muF) * Z.Nw * Z.phiStar lam) / Z.bnd Fl +
        (1 - Z.muX) * Z.chiStar lam / Z.bnd Fl := by
    ring
  have hY0 : 0 ≤ (Z.Nw * Z.wStar lam + (1 - Z.muF) * Z.Nw * Z.phiStar lam) / Z.bnd Fl :=
    div_nonneg (add_nonneg (mul_nonneg hNw.le hw0) (mul_nonneg (mul_nonneg (sub_nonneg.mpr hmuF1) hNw.le) hφ0))
      hB0.le
  have hχB : 0 ≤ (1 - Z.muX) * Z.chiStar lam / Z.bnd Fl :=
    div_nonneg (mul_nonneg (sub_nonneg.mpr hmuX1) hχ0) hB0.le
  have hκf : Z.kappa Fl lam + (1 - Z.muF) * Z.Nw * Z.phiStar lam / Z.bnd Fl =
      (Z.Nw * Z.wStar lam + (1 - Z.muF) * Z.Nw * Z.phiStar lam) / Z.bnd Fl := by
    rw [Model.kappa, add_div]
  have hl2 : 0 < Real.log 2 := Real.log_pos (by norm_num)
  have hδpos : 0 < δ' := lt_of_lt_of_le (by positivity) hδ'
  have hinv : 1 / δ' ≤ (2 : ℝ) ^ 41 := by
    have h := one_div_le_one_div_of_le (by positivity : (0 : ℝ) < (1 / 2) ^ 41) hδ'
    rwa [show (1 : ℝ) / (1 / 2) ^ 41 = 2 ^ 41 by norm_num] at h
  have ht : Real.log (1 / δ') ≤ 41 * Real.log 2 := by
    calc Real.log (1 / δ') ≤ Real.log ((2 : ℝ) ^ 41) := Real.log_le_log (one_div_pos.mpr hδpos) hinv
      _ = 41 * Real.log 2 := by rw [Real.log_pow]; norm_num
  have h41 : Real.log (1 / δ') / Real.log 2 ≤ 41 := (div_le_iff₀ hl2).mpr ht
  have hm := mul_le_mul_of_nonneg_left h41 hY0
  rw [hκf, one_div_one_div, hS,
    show (Z.Nw * Z.wStar lam + (1 - Z.muF) * Z.Nw * Z.phiStar lam) / Z.bnd Fl / Real.log 2 * Real.log (1 / δ') /
        48000 = (Z.Nw * Z.wStar lam + (1 - Z.muF) * Z.Nw * Z.phiStar lam) / Z.bnd Fl *
          (Real.log (1 / δ') / Real.log 2) / 48000 by ring]
  linarith

/-- The headlines' last step at `p = 7/8`, `p₀ = ½`, `S = 48,000` and `δ′ ≥ 2^-41`: at
`Λ = max(κ/ln(8/7), (κ + f)/ln 2)`, `Λ·ln(1/δ′)/S` is the folded branch's `ε_jl` or the unfolded branch's term, so a
bound `1 − ε` below both branches' floors is below `dIdealFloor_whole`'s. -/
theorem headline_of_branches {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U]
    [Fintype F] [Fintype X] [Nonempty F] [Nonempty X] (Z : Model K R T L U F X) (Fl : Labelled K) (lam : R → ℝ)
    (h : IsCert lam) (hZ : Z.Wf Fl) (S : ℕ) (hS : (S : ℝ) = 48000) (δ' : ℝ) (hδ' : (1 / 2 : ℝ) ^ 41 ≤ δ')
    (ε : ℝ)
    (hfold : 1 - ε ≤ 1 - Z.epsSkip Fl lam - Z.epsCert Fl lam - Z.epsRoute Fl lam - Z.epsMand Fl lam -
      Z.epsJl Fl lam (7 / 8) δ' S)
    (hunf : 1 - ε ≤ (Z.Nw * Z.wStar lam + (1 - Z.muF) * Z.Nw * Z.phiStar lam) / Z.bnd Fl * (1 - 41 / 48000)) :
    1 - ε ≤ 1 - Z.epsSkip Fl lam - Z.epsCert Fl lam - Z.epsRoute Fl lam - Z.epsMand Fl lam -
      max (Z.kappa Fl lam / Real.log (1 / (7 / 8)))
        ((Z.kappa Fl lam + (1 - Z.muF) * Z.Nw * Z.phiStar lam / Z.bnd Fl) / Real.log (1 / (1 / 2))) *
          Real.log (1 / δ') / S := by
  rcases le_total (Z.kappa Fl lam / Real.log (1 / (7 / 8)))
      ((Z.kappa Fl lam + (1 - Z.muF) * Z.Nw * Z.phiStar lam / Z.bnd Fl) / Real.log (1 / (1 / 2))) with hAB | hAB
  · rw [max_eq_right hAB]
    exact hunf.trans (unfolded_floor Z Fl lam h hZ S hS δ' hδ')
  · rw [max_eq_left hAB, show Z.kappa Fl lam / Real.log (1 / (7 / 8)) * Real.log (1 / δ') / S =
        Z.epsJl Fl lam (7 / 8) δ' S by rw [Model.epsJl]; ring]
    exact hfold

end Accounting.SecurityProofs.TS.Whole
