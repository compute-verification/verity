import Proofs.Accounting.TS.FanIn
import Proofs.Accounting.TS.FoldFanIn
import Proofs.Accounting.TS.LowBit
import Proofs.Accounting.TS.Price
import Proofs.Accounting.TS.Rounds
import Proofs.Accounting.TS.Facets
import Proofs.Accounting.TS.Receipt
import Proofs.Accounting.TS.Tensor
import Proofs.Accounting.TS.Sheet
import Proofs.Accounting.TS.ProofsSound
import Proofs.Accounting.TS.Fork
import Proofs.Accounting.TS.Drawn
import Proofs.Accounting.TS.Keyed
import Proofs.Pouw.RelativeSecurity

/-!
# The tensor–scalar model's proofs

Each guarantee of `Definitions.Accounting.Guarantees.TS` as `Accounting.SecurityProofs.TS.G`. One file per prover
group: the price lemmas (`TS/Price.lean`), the round lemmas
(`TS/Rounds.lean`), Lemma 1 with Corollaries D1 and D2 (`TS/Facets.lean`), T1 and Lemma 2 (`TS/Tensor.lean`), the
fan-in count with FOLD-OPT and FILL-OPT from it (`TS/FanIn.lean`), the fan-in floor of a fold of binding steps and
of duo's and fx1's runs (`TS/FoldFanIn.lean`), the low bit of a one-register fold and of li (`TS/LowBit.lean`); this file keeps the theorems built from them
(EPS1-SPEC §2.2, §2.4, §2.7).

All five run one chain: a price lower bound `Φ(π) ≤ π·n(M)` (from (RD), `price_of_rd`, or from the parts,
`price_of_parts`), then `π·n ≤ B(n)` (`price_le_bound`) and `B(n) ≤ f_max·T(M)` (`TimeBound`), and `identity` reads
`Φ(π)/B(L)` as the terms (`honest_of_price`, `floor_of_price`). D-ideal's luck is `ε_luck` at
`d̄ = ln(1/δ′)/(ln(1/p)·S)`, off `luck_rounds`' event at `K = ⌈ln(1/δ′)/ln(1/p)⌉` (`p^K ≤ δ′`, `pow_ceil_le`); Lemma C's
is `ε_luck` at `d̄ = Λ·ln(1/δ′)/S`, off `budget_tail`'s event at `t = ln(1/δ′)`.
-/

namespace Accounting.SecurityProofs.TS

open Accounting.TS Accounting.Assumptions.TS Pouw.RelativeSecurity

/-- Steps 1 and 3 of Theorem D(a): `π·n ≤ B(n) ≤ f_max·T(M)` on a run whose facets are ceilings. -/
theorem price_le_time {K R : Type} [Fintype K] [Fintype R] [Nonempty R] (G : FacetCosts K R) (lam : R → ℝ)
    (M : Run K) (h : IsCert lam) (hf : 0 < G.fmax) (hM : FacetCeilings G M) :
    G.price lam M.n ≤ G.fmax * M.time := by
  have h1 := price_le_bound G lam M.n h
  have h2 : G.bound M.n / G.fmax ≤ M.time := TimeBound G M hM
  rw [div_le_iff₀ hf] at h2
  linarith

/-- `Φ/B · B/C_H · f_H/f_max ≤ T/(C_H/f_H)` when `Φ ≤ f_max·T`. -/
theorem ratio_honest {Φ B CH fH fmax t : ℝ} (hB : 0 < B) (hCH : 0 < CH) (hfH : 0 < fH) (hf : 0 < fmax)
    (h : Φ ≤ fmax * t) : Φ / B * (1 - (1 - B / CH)) * (1 - (1 - fH / fmax)) ≤ t / (CH / fH) := by
  have := hB.ne'
  have := hCH.ne'
  have := hfH.ne'
  have := hf.ne'
  rw [sub_sub_cancel, sub_sub_cancel]
  have e1 : Φ / B * (B / CH) * (fH / fmax) = Φ * fH / (CH * fmax) := by field_simp
  have e2 : t / (CH / fH) = t * fH / CH := by field_simp
  rw [e1, e2, div_le_div_iff₀ (mul_pos hCH hf) hCH]
  have := mul_le_mul_of_nonneg_right h (le_of_lt (mul_pos hfH hCH))
  linarith

/-- `Φ/B ≤ T/(B/f_max)` when `Φ ≤ f_max·T`. -/
theorem ratio_floor {Φ B fmax t : ℝ} (hB : 0 < B) (h : Φ ≤ fmax * t) :
    Φ / B ≤ t / (B / fmax) := by
  rw [div_div_eq_mul_div]
  exact div_le_div_of_nonneg_right (by linarith) hB.le

/-- The honest form of steps 3–4: a price lower bound `Φ(π) ≤ π·n(M)` on a run whose facets are ceilings gives
`T(M)/T(H) ≥ (1 − ε_skip − ε_mand − ε_machine − ε_luck − ε_adapt)(1 − ε_sched)(1 − ε_clock)`. -/
theorem honest_of_price {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U]
    [Fintype F] [Fintype X] [Nonempty R] (Z : Model K R T L U F X) (H : Honest K) (lam : R → ℝ)
    (dbar abar : ℝ) (M : Run K) (h : IsCert lam) (hZ : Z.Wf H.prog) (hCH : 0 < H.CH) (hfH : 0 < H.fH)
    (hM : FacetCeilings Z.G M)
    (hΦ : (1 - dbar - abar) * Z.Nw * Z.wStar lam + (1 - Z.muF) * Z.Nw * Z.phiStar lam +
      (1 - Z.muX) * Z.chiStar lam ≤ Z.G.price lam M.n) :
    (1 - Z.epsSkip H.prog lam - Z.epsMand H.prog lam - Z.epsMachine H.prog lam - Z.epsLuck H.prog lam dbar -
        Z.epsAdapt H.prog lam abar) * (1 - Z.epsSched H) * (1 - Z.epsClock H) ≤ M.time / H.time := by
  have hf : 0 < Z.G.fmax := hZ.2.1
  have hB : 0 < Z.bnd H.prog := hZ.2.2.2.1
  have e : 1 - Z.epsSkip H.prog lam - Z.epsMand H.prog lam - Z.epsMachine H.prog lam -
      Z.epsLuck H.prog lam dbar - Z.epsAdapt H.prog lam abar =
      ((1 - dbar - abar) * Z.Nw * Z.wStar lam + (1 - Z.muF) * Z.Nw * Z.phiStar lam +
        (1 - Z.muX) * Z.chiStar lam) / Z.bnd H.prog := by
    rw [identity Z H.prog lam dbar abar hZ, Model.epsMachine]
    ring
  rw [e, Model.epsSched, Model.epsClock, Honest.time]
  exact ratio_honest hB hCH hfH hf (hΦ.trans (price_le_time Z.G lam M h hf hM))

/-- The floor form of steps 3–4: a price lower bound `Φ(π) ≤ π·n(M)` on a run whose facets are ceilings gives
`T(M)/T_cred ≥ 1 − ε^♭_skip − ε^♭_cert − ε^♭_route − ε^♭_mand − ε^♭_luck − ε^♭_adapt`. -/
theorem floor_of_price {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U]
    [Fintype F] [Fintype X] [Nonempty R] (Z : Model K R T L U F X) (Fl : Labelled K) (lam : R → ℝ)
    (dbar abar : ℝ) (M : Run K) (h : IsCert lam) (hZ : Z.Wf Fl) (hM : FacetCeilings Z.G M)
    (hΦ : (1 - dbar - abar) * Z.Nw * Z.wStar lam + (1 - Z.muF) * Z.Nw * Z.phiStar lam +
      (1 - Z.muX) * Z.chiStar lam ≤ Z.G.price lam M.n) :
    1 - Z.epsSkip Fl lam - Z.epsCert Fl lam - Z.epsRoute Fl lam - Z.epsMand Fl lam - Z.epsLuck Fl lam dbar -
        Z.epsAdapt Fl lam abar ≤ M.time / Z.credit Fl := by
  have hf : 0 < Z.G.fmax := hZ.2.1
  have hB : 0 < Z.bnd Fl := hZ.2.2.2.1
  have e : 1 - Z.epsSkip Fl lam - Z.epsCert Fl lam - Z.epsRoute Fl lam - Z.epsMand Fl lam -
      Z.epsLuck Fl lam dbar - Z.epsAdapt Fl lam abar =
      ((1 - dbar - abar) * Z.Nw * Z.wStar lam + (1 - Z.muF) * Z.Nw * Z.phiStar lam +
        (1 - Z.muX) * Z.chiStar lam) / Z.bnd Fl := by
    rw [identity Z Fl lam dbar abar hZ]
    ring
  rw [e, Model.credit]
  exact ratio_floor hB (hΦ.trans (price_le_time Z.G lam M h hf hM))

/-- The fixed words' part of `Φ`: `fixed ≥ (1 − d̄ − ā)·N_w` and (RD) in price form give `Φ(π) ≤ π·n`. -/
theorem phi_le_of_fixed {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U]
    [Fintype F] [Fintype X] (Z : Model K R T L U F X) (lam : R → ℝ) (dbar abar fixed price : ℝ)
    (hw : 0 ≤ Z.wStar lam) (hfix : (1 - dbar - abar) * Z.Nw ≤ fixed)
    (hp : fixed * Z.wStar lam + (1 - Z.muF) * Z.Nw * Z.phiStar lam + (1 - Z.muX) * Z.chiStar lam ≤ price) :
    (1 - dbar - abar) * Z.Nw * Z.wStar lam + (1 - Z.muF) * Z.Nw * Z.phiStar lam +
      (1 - Z.muX) * Z.chiStar lam ≤ price := by
  have := mul_le_mul_of_nonneg_right hfix hw
  linarith

/-- One bet a round reaches `δ′`: `p^⌈ln(1/δ′)/ln(1/p)⌉ ≤ δ′` for `p ∈ (0, 1)` and `δ′ > 0`. -/
theorem pow_ceil_le {p δ' : ℝ} (hp0 : 0 < p) (hp1 : p < 1) (hδ : 0 < δ') :
    p ^ ⌈Real.log (1 / δ') / Real.log (1 / p)⌉₊ ≤ δ' := by
  set t := Real.log (1 / δ') / Real.log (1 / p) with ht
  have hlogp : Real.log p < 0 := Real.log_neg hp0 hp1
  have htl : t * Real.log p = Real.log δ' := by
    rw [ht, one_div, one_div, Real.log_inv, Real.log_inv, neg_div_neg_eq]
    exact div_mul_cancel₀ _ hlogp.ne
  have hk := mul_le_mul_of_nonpos_right (Nat.le_ceil t) hlogp.le
  calc p ^ ⌈t⌉₊ = Real.exp (Real.log (p ^ ⌈t⌉₊)) := (Real.exp_log (pow_pos hp0 _)).symm
    _ = Real.exp (⌈t⌉₊ * Real.log p) := by rw [Real.log_pow]
    _ ≤ Real.exp (Real.log δ') := Real.exp_le_exp.mpr (by linarith)
    _ = δ' := Real.exp_log hδ

/-- Theorem D(a): `price_le_bound`, `price_of_rd`, `identity` and `TimeBound`. -/
theorem DecompositionD : Accounting.Guarantees.TS.DecompositionD := by
  intro K R T L U F X _ _ _ _ _ _ _ _ _ _ Z H Ω _ accepts M fixed lam dbar abar δ h hZ hCH hfH hM hRD hLk
  refine le_trans (prob_mono ?_) hLk
  rintro ω ⟨ha, hlt⟩
  refine ⟨ha, ?_⟩
  by_contra hfix
  push Not at hfix
  refine absurd hlt (not_lt.mpr (honest_of_price Z H lam dbar abar (M ω) h hZ hCH hfH (hM ω) ?_))
  exact phi_le_of_fixed Z lam dbar abar (fixed ω) _ (stars_nonneg Z H.prog lam hZ h).2.1 hfix
    (price_of_rd Z H.prog (M ω).n (fixed ω) lam h hZ (hRD ω ha))

/-- D-ideal's union bound: off `ProofsSound`'s event `E` and the luck's event `B`, an accepted window is accepted by
the ideal game with deficiency `< D₀`, so its fixed words are at least `(1 − D₀/S)·N_w`. -/
theorem fixed_of_tail {Ω K T L U : Type} [Fintype Ω] {S : ℕ} (P : Play Ω S K T L U) (Nw D₀ : ℝ) (hS : 0 < S)
    (hNw : 0 < Nw) (E : Paths Ω S → Prop)
    (hacc : ∀ x, P.accepts x → ¬ E x → P.game.accepts x ∧ (1 - P.game.deficiency x / S) * Nw ≤ (P.run x).fixed)
    (x : Paths Ω S) (ha : P.accepts x) (hE : ¬ E x) (hD : P.game.accepts x → P.game.deficiency x < D₀) :
    (1 - D₀ / S - 0) * Nw ≤ (P.run x).fixed := by
  obtain ⟨hg, hfix⟩ := hacc x ha hE
  have hS' : (0 : ℝ) < S := Nat.cast_pos.mpr hS
  have h1 : P.game.deficiency x / S ≤ D₀ / S := div_le_div_of_nonneg_right (hD hg).le hS'.le
  have := mul_le_mul_of_nonneg_right h1 hNw.le
  linarith

/-- Theorem D-ideal (2.7): `luck_rounds`, `deficiency_le_rounds`, `price_of_parts`, `identity`, `TimeBound`. -/
theorem DIdeal : Accounting.Guarantees.TS.DIdeal := by
  intro K R T L U F X _ _ _ _ _ _ _ _ _ _ Z H Ω _ S P lam p δ δbind δzk nπ h hZ hCH hfH hS hp0 hp1 hδ hSplit hM
    hW hSR hU hN hF hX hL hPS
  set δ' := δ - S * (δbind + nπ * δzk) with hδ'
  obtain ⟨E, hE, hacc⟩ := hPS
  set t := Real.log (1 / δ') / Real.log (1 / p) with ht
  have hB := (luck_rounds P.game p hp0.le hL ⌈t⌉₊).trans (pow_ceil_le hp0 hp1 hδ)
  calc prob (fun x => P.accepts x ∧ (P.run x).run.time / H.time <
        (1 - Z.epsSkip H.prog lam - Z.epsMand H.prog lam - Z.epsMachine H.prog lam -
          Z.epsJl H.prog lam p δ' S) * (1 - Z.epsSched H) * (1 - Z.epsClock H))
      ≤ prob (fun x => E x ∨ (P.game.accepts x ∧ ⌈t⌉₊ ≤ Accounting.TS.RoundGame.deficientRounds P.game x)) := by
        refine prob_mono ?_
        rintro x ⟨ha, hlt⟩
        by_contra hne
        push Not at hne
        have hfix := fixed_of_tail P Z.Nw t hS hZ.2.2.1 E hacc x ha hne.1 fun hg =>
          (deficiency_le_rounds P.game p hL x).trans_lt (Nat.lt_ceil.mp (hne.2 hg))
        have key := honest_of_price Z H lam (t / S) 0 (P.run x).run h hZ hCH hfH (hM x)
          (phi_le_of_fixed Z lam (t / S) 0 _ _ (stars_nonneg Z H.prog lam hZ h).2.1 hfix
            (price_of_parts Z H.prog (P.run x) lam h hZ (hSplit x) (hW x) (hSR x) (hU x) (hN x) (hF x) (hX x)))
        have e : Z.epsLuck H.prog lam (t / S) = Z.epsJl H.prog lam p δ' S := by
          rw [Model.epsLuck, Model.epsJl, ht]
          ring
        rw [e, Model.epsAdapt, zero_mul, sub_zero] at key
        exact absurd hlt (not_lt.mpr key)
    _ ≤ prob E + prob (fun x => P.game.accepts x ∧ ⌈t⌉₊ ≤ Accounting.TS.RoundGame.deficientRounds P.game x) :=
        prob_or_le _ _
    _ ≤ S * (δbind + nπ * δzk) + δ' := add_le_add hE hB
    _ = δ := by rw [hδ']; ring

/-- Theorem D-ideal (2.7♭): as `DIdeal`, with the identity at ♭. -/
theorem DIdealFloor : Accounting.Guarantees.TS.DIdealFloor := by
  intro K R T L U F X _ _ _ _ _ _ _ _ _ _ Z Fl Ω _ S P lam p δ δbind δzk nπ h hZ hS hp0 hp1 hδ hSplit hM
    hW hSR hU hN hF hX hL hPS
  set δ' := δ - S * (δbind + nπ * δzk) with hδ'
  obtain ⟨E, hE, hacc⟩ := hPS
  set t := Real.log (1 / δ') / Real.log (1 / p) with ht
  have hB := (luck_rounds P.game p hp0.le hL ⌈t⌉₊).trans (pow_ceil_le hp0 hp1 hδ)
  calc prob (fun x => P.accepts x ∧ (P.run x).run.time / Z.credit Fl <
        1 - Z.epsSkip Fl lam - Z.epsCert Fl lam - Z.epsRoute Fl lam - Z.epsMand Fl lam -
          Z.epsJl Fl lam p δ' S)
      ≤ prob (fun x => E x ∨ (P.game.accepts x ∧ ⌈t⌉₊ ≤ Accounting.TS.RoundGame.deficientRounds P.game x)) := by
        refine prob_mono ?_
        rintro x ⟨ha, hlt⟩
        by_contra hne
        push Not at hne
        have hfix := fixed_of_tail P Z.Nw t hS hZ.2.2.1 E hacc x ha hne.1 fun hg =>
          (deficiency_le_rounds P.game p hL x).trans_lt (Nat.lt_ceil.mp (hne.2 hg))
        have key := floor_of_price Z Fl lam (t / S) 0 (P.run x).run h hZ (hM x)
          (phi_le_of_fixed Z lam (t / S) 0 _ _ (stars_nonneg Z Fl lam hZ h).2.1 hfix
            (price_of_parts Z Fl (P.run x) lam h hZ (hSplit x) (hW x) (hSR x) (hU x) (hN x) (hF x) (hX x)))
        have e : Z.epsLuck Fl lam (t / S) = Z.epsJl Fl lam p δ' S := by
          rw [Model.epsLuck, Model.epsJl, ht]
          ring
        rw [e, Model.epsAdapt, zero_mul, sub_zero] at key
        exact absurd hlt (not_lt.mpr key)
    _ ≤ prob E + prob (fun x => P.game.accepts x ∧ ⌈t⌉₊ ≤ Accounting.TS.RoundGame.deficientRounds P.game x) :=
        prob_or_le _ _
    _ ≤ S * (δbind + nπ * δzk) + δ' := add_le_add hE hB
    _ = δ := by rw [hδ']; ring

/-- Theorem D-ideal (2.7♭) on the data-sheet machine: `DIdealFloor`, its (I1) from `facetCeilings_of_sheet`, (I2)/(I4)
and (I3) from A6 (`scalarRoute_half`), (I5) from A7. -/
theorem SheetDIdealFloor : Accounting.Guarantees.TS.SheetDIdealFloor := by
  intro K P T L U F X _ _ _ _ _ _ _ _ _ _ Sh Z Fl Ω _ S Pl lam p δ δbind δzk nπ hG h hZ hS hp0 hp1 hδ hSplit hq
    hqT hb h1 h2 h3 h4 h5 h6 h7 hF hX hL hPS
  refine DIdealFloor Z Fl Ω S Pl lam p δ δbind δzk nπ h hZ hS hp0 hp1 hδ hSplit (fun x => ?_) (fun x => (h6 x).1)
    (fun x => scalarRoute_half Z _ (hSplit x) (h6 x) hq hqT hb) (fun x => (h7 x).2) (fun x => (h7 x).1) hF hX hL hPS
  rw [hG]
  exact facetCeilings_of_sheet Sh _ (run_nonneg_of_split (hSplit x)) (h1 x) (h2 x) (h3 x) (h4 x) (h5 x)

/-- Lemma C: `budget_tail`, `price_of_parts`, `identity`, `TimeBound`. -/
theorem CompilerC : Accounting.Guarantees.TS.CompilerC := by
  intro K R T L U F X _ _ _ _ _ _ _ _ _ _ Z H Ω _ S P lam Λ p δ δbind δzk nπ h hZ hCH hfH hS hp0 hp1 hΛ hδ
    hSplit hM hW hSR hU hN hF hX hBud hPS
  set δ' := δ - S * (δbind + nπ * δzk) with hδ'
  obtain ⟨E, hE, hacc⟩ := hPS
  set t := Real.log (1 / δ') with ht
  have het : Real.exp (-t) = δ' := by
    rw [ht, one_div, Real.log_inv, neg_neg, Real.exp_log hδ]
  have hB := (budget_tail P.game Λ hΛ hBud t).trans het.le
  calc prob (fun x => P.accepts x ∧ (P.run x).run.time / H.time <
        (1 - Z.epsSkip H.prog lam - Z.epsMand H.prog lam - Z.epsMachine H.prog lam -
          Z.epsJl H.prog lam p δ' S - Z.epsSample H.prog lam Λ p δ' S) * (1 - Z.epsSched H) *
            (1 - Z.epsClock H))
      ≤ prob (fun x => E x ∨ (P.game.accepts x ∧ Λ * t ≤ P.game.deficiency x)) := by
        refine prob_mono ?_
        rintro x ⟨ha, hlt⟩
        by_contra hne
        push Not at hne
        have hfix := fixed_of_tail P Z.Nw (Λ * t) hS hZ.2.2.1 E hacc x ha hne.1 hne.2
        have key := honest_of_price Z H lam (Λ * t / S) 0 (P.run x).run h hZ hCH hfH (hM x)
          (phi_le_of_fixed Z lam (Λ * t / S) 0 _ _ (stars_nonneg Z H.prog lam hZ h).2.1 hfix
            (price_of_parts Z H.prog (P.run x) lam h hZ (hSplit x) (hW x) (hSR x) (hU x) (hN x) (hF x) (hX x)))
        have e : 1 - Z.epsSkip H.prog lam - Z.epsMand H.prog lam - Z.epsMachine H.prog lam -
            Z.epsLuck H.prog lam (Λ * t / S) - Z.epsAdapt H.prog lam 0 =
            1 - Z.epsSkip H.prog lam - Z.epsMand H.prog lam - Z.epsMachine H.prog lam -
              Z.epsJl H.prog lam p δ' S - Z.epsSample H.prog lam Λ p δ' S := by
          rw [Model.epsLuck, Model.epsAdapt, Model.epsJl, Model.epsSample, ← ht]
          ring
        rw [e] at key
        exact absurd hlt (not_lt.mpr key)
    _ ≤ prob E + prob (fun x => P.game.accepts x ∧ Λ * t ≤ P.game.deficiency x) := prob_or_le _ _
    _ ≤ S * (δbind + nπ * δzk) + δ' := add_le_add hE hB
    _ = δ := by rw [hδ']; ring

/-- A star is at most the price of each route of its menu. -/
theorem iInf_le_price {ι R : Type} [Finite ι] [Fintype R] (lam : R → ℝ) (d : ι → R → ℝ) (i : ι) :
    ⨅ j, demandPrice lam (d j) ≤ demandPrice lam (d i) :=
  ciInf_le (Set.finite_range _).bddBelow i

/-- The price of a load that is a nonnegative combination of routes: `π·(Σ_i c_i d_i) = Σ_i c_i·(π·d_i)`. -/
theorem price_of_combination {K R ι : Type} [Fintype K] [Fintype R] [Fintype ι] (G : FacetCosts K R)
    (lam : R → ℝ) (n : K → ℝ) (c : ι → ℝ) (d : ι → R → ℝ) (hn : ∀ r, G.load n r = ∑ i, c i * d i r) :
    G.price lam n = ∑ i, c i * demandPrice lam (d i) := by
  unfold FacetCosts.price demandPrice
  simp_rw [hn, Finset.mul_sum]
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun r _ => by ring

/-- Corollary D3: `price_of_rd`, `identity` at ♭, `TimeBound`; the signs from `FloorKnown`. -/
theorem FloorCreditD3 : Accounting.Guarantees.TS.FloorCreditD3 := by
  intro K R T L U F X _ _ _ _ _ _ _ _ _ _ Z H Fl Ω _ accepts M fixed lam dbar abar δ h hZ hCH hfH hM hRD hLk
  have hZ' := hZ
  obtain ⟨ha0, hf, hNw, hB, -, -, -, -, -, -, -, hmuF0, -, hmuX0, -⟩ := hZ'
  obtain ⟨-, hw, hφ, hχ⟩ := stars_nonneg Z Fl lam hZ h
  refine ⟨?_, ?_, ?_⟩
  · refine le_trans (prob_mono ?_) hLk
    rintro ω ⟨ha, hlt⟩
    refine ⟨ha, ?_⟩
    by_contra hfix
    push Not at hfix
    refine absurd hlt (not_lt.mpr (floor_of_price Z Fl lam dbar abar (M ω) h hZ (hM ω) ?_))
    exact phi_le_of_fixed Z lam dbar abar (fixed ω) _ hw hfix
      (price_of_rd Z Fl (M ω).n (fixed ω) lam h hZ (hRD ω ha))
  · rintro ⟨⟨N, hN0, hNsum, hNload⟩, ⟨φ, hφload⟩, ⟨ψ, hψload⟩⟩ hns hd ha
    have hκ : 0 ≤ Z.kappa Fl lam := by
      rw [Model.kappa]
      exact div_nonneg (mul_nonneg hNw.le hw) hB.le
    have hpw : Z.Nw * Z.wStar lam ≤ Z.G.price lam Fl.nw := by
      rw [price_of_combination Z.G lam Fl.nw N Z.demand hNload, ← hNsum, Finset.sum_mul]
      refine Finset.sum_le_sum fun hd _ => mul_le_mul_of_nonneg_left ?_ (hN0 hd)
      exact iInf_le_price lam Z.demand hd
    have hpf : Z.Nw * Z.phiStar lam ≤ Z.G.price lam Fl.nf := by
      have h1 : Z.phiStar lam ≤ demandPrice lam (Z.fold φ) := iInf_le_price lam Z.fold φ
      rw [price_of_combination Z.G lam Fl.nf (fun _ : Unit => Z.Nw) (fun _ => Z.fold φ) (by simpa using hφload)]
      simpa using mul_le_mul_of_nonneg_left h1 hNw.le
    have hpx : Z.chiStar lam ≤ Z.G.price lam Fl.nx := by
      have h1 : Z.chiStar lam ≤ demandPrice lam (Z.fill ψ) := iInf_le_price lam Z.fill ψ
      rw [price_of_combination Z.G lam Fl.nx (fun _ : Unit => 1) (fun _ => Z.fill ψ) (by simpa using hψload)]
      simpa using h1
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [Model.epsSkip]
      refine div_nonneg ?_ hB.le
      unfold FacetCosts.price FacetCosts.load
      exact Finset.sum_nonneg fun r _ => mul_nonneg (h.1 r) (Finset.sum_nonneg fun k _ =>
        mul_nonneg (ha0 r k) (hns k))
    · rw [Model.epsCert]
      exact sub_nonneg.mpr ((div_le_one hB).mpr (price_le_bound Z.G lam Fl.n h))
    · rw [Model.epsRoute]
      refine div_nonneg ?_ hB.le
      have e : Z.Nw * (Z.G.price lam Fl.nw / Z.Nw - Z.wStar lam) +
          Z.Nw * (Z.G.price lam Fl.nf / Z.Nw - Z.phiStar lam) + (Z.G.price lam Fl.nx - Z.chiStar lam) =
          (Z.G.price lam Fl.nw - Z.Nw * Z.wStar lam) + (Z.G.price lam Fl.nf - Z.Nw * Z.phiStar lam) +
            (Z.G.price lam Fl.nx - Z.chiStar lam) := by
        have := hNw.ne'
        field_simp
      rw [e]
      linarith
    · rw [Model.epsMand]
      exact div_nonneg (add_nonneg (mul_nonneg (mul_nonneg hmuF0 hNw.le) hφ) (mul_nonneg hmuX0 hχ)) hB.le
    · rw [Model.epsLuck]
      exact mul_nonneg hd hκ
    · rw [Model.epsAdapt]
      exact mul_nonneg ha hκ
  · have := hB.ne'
    have := hf.ne'
    have := hfH.ne'
    rw [Honest.time, Model.gammaFloor, Model.credit]
    field_simp

end Accounting.SecurityProofs.TS
