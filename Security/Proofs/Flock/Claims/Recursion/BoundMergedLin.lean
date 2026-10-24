import Proofs.Flock.Claims.Recursion.BoundMerged
import Proofs.Flock.Claims.Recursion.KSLin

/-!
# Route 2B, linearly: `KSLin` from (P1) at `DE` through one plurality

`Bound.innerKS_of_plainKSAbort_binds` reaches the inner stage from (P1) at `DE` with no bridge, but charges `P_O`'s bad
draws a root: `Pr[BadO] ≤ √(k_I·Σ_j Pr[forkColl j])` (`Bound.Pr_badO_le_fork`, through `Pr_plur_fork`). Here that root
goes, as `Pr_bad_le_lin` takes `Pr_bad_le`'s:

* At inner round `j`, a draw misses `P_O`'s plurality at most as often as `Fdev`'s link at the fork at `j` succeeds,
  plus `1/(M+1)` (`Pr_plur_fork_lin` at `P_O`'s reading, `Bound.oMsg`). `Fdev` itself reads only whether `E_O` gives
  witnesses, never the opening, so it is the same finder at either plurality.
* Where the link succeeds, the first draw and the loop's first good draw are forked at `j`, give witnesses, and open
  round `j` differently, so their calls collide (`Bound.fork_collO`, `Bound.linkOut_collO`).

So `Pr[BadO] ≤ k_I·(Pr[Fdev] + 1/(M+1))` (`Bound.Pr_badO_le_PrFdev`), and the inner stage is at most
`κ_I + √(γ_BI·Pr[B_I]) + β·(Pr[Fdev] + 1/(M+1))` (`Bound.innerLin_of_plainKSAbort_binds`): `E^rep` at `DE` against
`P_O`'s aborting runs by (P1), with `B` there at most `B_I` (`Bound.PrBAbort_le_merged`) and `γ_DE ≤ γ_I` where
`DE`'s weight is at most the bound record's; a rejecting judged draw at most `Pr[B_I]` (`Pr_innerRejects_le_PrBI`).
`Bound.KSLin_of_plainKSAbort_binds` is `KSLin` at the bound record from it, on `Bound.KS_of_plainKSAbort_binds`'
premises: no bridge and no `WeightCovers`.
-/

namespace Claims.Recursion

open Claims

section MergedLin

variable {Coin Col T Col' T' : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
  {Bd : Bound Coin Col T} {decI : List (ℕ × Col) → List T} {H : Msg → Msg} {D : Deploy Coin}
  {DO : Flock.Deploy Coin Col' T'} {decO : List (ℕ × Col') → List T'} {Ω : Type} {P : Prover Ω} {x : Msg}

/-- **A link that succeeds at `P_O`'s reading names a collision**: the first draw and the loop's first where `E_O`
gives witnesses open round `j` differently, at two draws forked at `j` (`fork_fork`), which `fork_collO` makes a
collision among their recorded calls, the pair `linkOut` outputs. -/
theorem Bound.linkOut_collO (hB : Bd.Binds) (hV : InnerViewRun H D (Inner.ofFlock Bd.deploy decI) DO)
    (hT : CallsTrue H D (Inner.ofFlock Bd.deploy decI) DO) (hL : CallsLogColl H D (Inner.ofFlock Bd.deploy decI) DO)
    (hFk : ForksBindAt H D (Inner.ofFlock Bd.deploy decI) DO x)
    (hA : InnerAcceptsCollAt H D (Inner.ofFlock Bd.deploy decI) DO x)
    {M : ℕ} (j : Fin ((Inner.ofFlock Bd.deploy decI).V.k (xI D x))) (ω : Ω) (r : Fin (D.V.k x) → Coin)
    (u : (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x)
    (us : Fin M → (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x)
    (hg : (eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x ω (fork j r u.1, u.2)).isSome = true)
    (hd : firstDiffers
      (fun v => (eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x ω (fork j r v.1, v.2)).isSome = true)
      (fun v => Bd.oMsg decI H D DO decO P x ω (fork j r v.1, v.2) j)
      (Bd.oMsg decI H D DO decO P x ω (fork j r u.1, u.2) j) M us) :
    ∃ a b, linkOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x
        ((j, ω, r, u, us) : LinCoins D (Inner.ofFlock Bd.deploy decI) DO decO Ω x M) = some (a, b) ∧
      Collision H a b := by
  obtain ⟨v, hv, hgv, hfv⟩ := firstPick_of_firstDiffers M us hd
  have hfc := Bound.fork_collO (decO := decO) (P := P) (x := x) hB hV hT hL hFk hA (j : ℕ) j.isLt
    (ω, fork j r u.1, v.1, u.2, v.2)
    ⟨hg, by simp only [fork_fork]; exact hgv, by simp only [fork_fork]; exact Ne.symm hfv⟩
  simp only [forkColl, fork_fork] at hfc
  have hv' : firstPick (linkG H D (Inner.ofFlock Bd.deploy decI) DO decO P x j ω r) M us = some v := hv
  simp only [linkOut, hv', Option.getD_some]
  exact hfc

/-- **`P_O`'s bad draws, linearly**: at most `k_I` times `Fdev`'s success, plus `k_I/(M+1)`, where
`Pr_badO_le_fork` is `√(k_I·Σ_j Pr[forkColl j])`. -/
theorem Bound.Pr_badO_le_lin (hB : Bd.Binds) (hV : InnerViewRun H D (Inner.ofFlock Bd.deploy decI) DO)
    (hT : CallsTrue H D (Inner.ofFlock Bd.deploy decI) DO) (hL : CallsLogColl H D (Inner.ofFlock Bd.deploy decI) DO)
    (hFk : ForksBindAt H D (Inner.ofFlock Bd.deploy decI) DO x)
    (hA : InnerAcceptsCollAt H D (Inner.ofFlock Bd.deploy decI) DO x)
    [Fintype Ω] [Nonempty Ω] (M : ℕ) (hk : 0 < (Inner.ofFlock Bd.deploy decI).V.k (xI D x)) :
    Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x =>
        Bd.BadO decI H D DO decO P x t.1 t.2) ≤
      (Inner.ofFlock Bd.deploy decI).V.k (xI D x) * (Fdev H D (Inner.ofFlock Bd.deploy decI) DO decO P x M hk).Pr H +
        (Inner.ofFlock Bd.deploy decI).V.k (xI D x) / ((M : ℝ) + 1) := by
  have h1 : Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x =>
        Bd.BadO decI H D DO decO P x t.1 t.2) ≤
      ∑ j : Fin ((Inner.ofFlock Bd.deploy decI).V.k (xI D x)),
        Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x =>
          (eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x t.1 t.2).isSome = true ∧
            Bd.oMsg decI H D DO decO P x t.1 t.2 j ≠
              Bd.opAtO decI H D DO decO P x t.1 j ((List.range j).map (coinSeq t.2.1))) := by
    refine (Pr_mono fun t ht => ?_).trans (Pr_exists_le _)
    obtain ⟨hs, hna⟩ := ht
    simp only [Bound.AgreesO, not_forall] at hna
    obtain ⟨j, hj, hne⟩ := hna
    exact ⟨⟨j, hj⟩, hs, hne⟩
  have h2 : ∀ j : Fin ((Inner.ofFlock Bd.deploy decI).V.k (xI D x)),
      Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x =>
        (eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x t.1 t.2).isSome = true ∧
          Bd.oMsg decI H D DO decO P x t.1 t.2 j ≠
            Bd.opAtO decI H D DO decO P x t.1 j ((List.range j).map (coinSeq t.2.1))) ≤
      Pr (fun t : Ω × (Fin (D.V.k x) → Coin) ×
          ((Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) ×
          (Fin M → (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) =>
        ∃ a b, linkOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x
          ((j, t) : LinCoins D (Inner.ofFlock Bd.deploy decI) DO decO Ω x M) = some (a, b) ∧ Collision H a b) +
        1 / ((M : ℝ) + 1) := fun j => by
    have hp := Pr_plur_fork_lin (j : ℕ)
      (fun ω d => (eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x ω d).isSome = true)
      (fun ω d => Bd.oMsg decI H D DO decO P x ω d j) M
    refine (le_of_eq (Pr_congr fun t => ?_)).trans (hp.trans (add_le_add (Pr_mono fun t ht => ?_) le_rfl))
    · simp only [Bound.opAtO, fork_preCoins]
    · exact Bound.linkOut_collO hB hV hT hL hFk hA j t.1 t.2.1 t.2.2.1 t.2.2.2 ht.1 ht.2
  have hF : ((Inner.ofFlock Bd.deploy decI).V.k (xI D x) : ℝ) *
      (Fdev H D (Inner.ofFlock Bd.deploy decI) DO decO P x M hk).Pr H =
      ∑ j : Fin ((Inner.ofFlock Bd.deploy decI).V.k (xI D x)), Pr (fun t : Ω × (Fin (D.V.k x) → Coin) ×
          ((Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) ×
          (Fin M → (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) =>
        ∃ a b, linkOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x
          ((j, t) : LinCoins D (Inner.ofFlock Bd.deploy decI) DO decO Ω x M) = some (a, b) ∧ Collision H a b) := by
    have hP : (Fdev H D (Inner.ofFlock Bd.deploy decI) DO decO P x M hk).Pr H =
        Ex fun j : Fin ((Inner.ofFlock Bd.deploy decI).V.k (xI D x)) =>
          Pr fun t : Ω × (Fin (D.V.k x) → Coin) ×
              ((Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) ×
              (Fin M → (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) =>
            ∃ a b, linkOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x
              ((j, t) : LinCoins D (Inner.ofFlock Bd.deploy decI) DO decO Ω x M) = some (a, b) ∧ Collision H a b :=
      Pr_prod (fun s : LinCoins D (Inner.ofFlock Bd.deploy decI) DO decO Ω x M =>
        ∃ a b, linkOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x s = some (a, b) ∧ Collision H a b)
    rw [hP]
    unfold Ex
    rw [Fintype.card_fin, mul_div_cancel₀ _ (by exact_mod_cast hk.ne')]
  calc Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x =>
        Bd.BadO decI H D DO decO P x t.1 t.2)
      ≤ ∑ j : Fin ((Inner.ofFlock Bd.deploy decI).V.k (xI D x)), (Pr (fun t : Ω × (Fin (D.V.k x) → Coin) ×
          ((Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) ×
          (Fin M → (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) =>
        ∃ a b, linkOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x
          ((j, t) : LinCoins D (Inner.ofFlock Bd.deploy decI) DO decO Ω x M) = some (a, b) ∧ Collision H a b) +
          1 / ((M : ℝ) + 1)) :=
        h1.trans (Finset.sum_le_sum fun j _ => h2 j)
    _ = (Inner.ofFlock Bd.deploy decI).V.k (xI D x) *
          (Fdev H D (Inner.ofFlock Bd.deploy decI) DO decO P x M hk).Pr H +
          (Inner.ofFlock Bd.deploy decI).V.k (xI D x) / ((M : ℝ) + 1) := by
        rw [Finset.sum_add_distrib, ← hF, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
          mul_one_div]

/-- **`P_O`'s bad draws at most `k_I·(Pr[Fdev] + 1/(M+1))`**: `Pr_badO_le_lin`, and no bad draw where the inner
protocol takes no rounds. -/
theorem Bound.Pr_badO_le_PrFdev (hB : Bd.Binds) (hV : InnerViewRun H D (Inner.ofFlock Bd.deploy decI) DO)
    (hT : CallsTrue H D (Inner.ofFlock Bd.deploy decI) DO) (hL : CallsLogColl H D (Inner.ofFlock Bd.deploy decI) DO)
    (hFk : ForksBindAt H D (Inner.ofFlock Bd.deploy decI) DO x)
    (hA : InnerAcceptsCollAt H D (Inner.ofFlock Bd.deploy decI) DO x)
    [Fintype Ω] [Nonempty Ω] (M : ℕ) :
    Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x =>
        Bd.BadO decI H D DO decO P x t.1 t.2) ≤
      (Inner.ofFlock Bd.deploy decI).V.k (xI D x) *
        (PrFdev H D (Inner.ofFlock Bd.deploy decI) DO decO P x M + 1 / ((M : ℝ) + 1)) := by
  by_cases hk : 0 < (Inner.ofFlock Bd.deploy decI).V.k (xI D x)
  · have h := Bound.Pr_badO_le_lin (decO := decO) (P := P) (x := x) hB hV hT hL hFk hA M hk
    simp only [PrFdev, hk, ↓reduceDIte]
    rw [mul_add, mul_one_div]
    exact h
  · have h0 : (Inner.ofFlock Bd.deploy decI).V.k (xI D x) = 0 := Nat.eq_zero_of_not_pos hk
    have h0' : Bd.DE.V.k (xI D x) = 0 := h0
    have hb : Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x =>
        Bd.BadO decI H D DO decO P x t.1 t.2) = 0 :=
      le_antisymm ((Pr_mono fun t ht => ht.2 fun j hj => absurd hj (by omega)).trans_eq Pr_false) (Pr_nonneg _)
    rw [hb, h0, Nat.cast_zero, zero_mul]

/-- **The inner stage at the bound record from (P1) at `DE`, linearly**: at most
`κ_I + √(γ_BI·Pr[B_I]) + β·(Pr[Fdev] + 1/(M+1))`, at every `M`. `E^rep` at `DE` against `P_O`'s aborting runs is (P1)'s
`κ + √(γ_DE·Pr[B at P_O's aborting runs])`, with that `B` at most `B_I` (`PrBAbort_le_merged`) and `γ_DE ≤ γ_I` where
`DE`'s weight is at most the bound record's (`hw`); `P_O`'s bad draws are `Pr_badO_le_PrFdev`'s; a rejecting judged
draw is at most `Pr[B_I]`, which `γ_BI` holds under `B_I`'s root (`sqrt_add_le_γBI`). -/
theorem Bound.innerLin_of_plainKSAbort_binds [DecidableEq Col'] [Zero Col'] (decC : Msg → Msg → Coin)
    (hKS : (Inner.ofFlock Bd.DE decI).KSAbort H) (hw : ∀ x, Bd.DE.weight x ≤ Bd.weight x)
    (hdec : ∀ x c, decC x (Bd.ans c) = c) (hB : Bd.Binds) (hR : InnerRounds D (Inner.ofFlock Bd.deploy decI))
    (hV : InnerViewRun H D (Inner.ofFlock Bd.deploy decI) DO) (hT : CallsTrue H D (Inner.ofFlock Bd.deploy decI) DO)
    (hL : CallsLogColl H D (Inner.ofFlock Bd.deploy decI) DO) (hFk : ForksBind H D (Inner.ofFlock Bd.deploy decI) DO)
    (hA : InnerAcceptsColl H D (Inner.ofFlock Bd.deploy decI) DO) (hM : RoundsMargin H D (Inner.ofFlock Bd.deploy decI))
    (hN0 : CallsNil H D) [Fintype Ω] [Nonempty Ω] (M : ℕ) :
    PrBadGood H D.V (innerR D (Inner.ofFlock Bd.deploy decI)) outR (EO D (Inner.ofFlock Bd.deploy decI) DO decO)
        (dec H D) (fun x => (Inner.ofFlock Bd.deploy decI).waits (xI D x))
        (fun x => (Inner.ofFlock Bd.deploy decI).k (xI D x)) (fun x => (Inner.ofFlock Bd.deploy decI).cap (xI D x))
        (Inner.ofFlock Bd.deploy decI).opens (fun x => (Inner.ofFlock Bd.deploy decI).assemble (xI D x))
        (fun x => (Inner.ofFlock Bd.deploy decI).check (xI D x)) x P ≤
      (Inner.ofFlock Bd.deploy decI).κ (xI D x) +
        Real.sqrt (γBI D (Inner.ofFlock Bd.deploy decI) x * PrBI H D (Inner.ofFlock Bd.deploy decI) DO decO P x) +
        βdev D (Inner.ofFlock Bd.deploy decI) x *
          (PrFdev H D (Inner.ofFlock Bd.deploy decI) DO decO P x M + 1 / ((M : ℝ) + 1)) := by
  have h1 := Bound.PrBadGood_le_merged (decO := decO) (P := P) (x := x) decC hdec hV hR
  have h2 : (Inner.ofFlock Bd.DE decI).PrBadAbort H (xI D x) (Bd.innerPO decI H D DO decO P x decC)
      (Bd.keepAO decI H D DO decO P x) ≤
      (Inner.ofFlock Bd.deploy decI).κ (xI D x) + Real.sqrt ((Inner.ofFlock Bd.DE decI).γ (xI D x) *
        PrBAbort H Bd.DE.V (Bd.innerPO decI H D DO decO P x decC) (xI D x) (Bd.keepAO decI H D DO decO P x)) :=
    hKS (xI D x) Ω (Bd.innerPO decI H D DO decO P x decC)
      ((Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) (Bd.keepAO decI H D DO decO P x)
  have h3 := Bound.PrBAbort_le_merged (decO := decO) (P := P) (x := x) decC hdec hV hL hT hN0 hR
  have hBA0 : 0 ≤ PrBAbort H Bd.DE.V (Bd.innerPO decI H D DO decO P x decC) (xI D x)
      (Bd.keepAO decI H D DO decO P x) := PrBAbort_nonneg H _ _ _ _
  have hk0 : (0 : ℝ) ≤ Bd.DE.V.k (xI D x) := Nat.cast_nonneg _
  have hs : Real.sqrt (2 * (Bd.DE.V.k (xI D x) : ℝ) * Bd.DE.weight (xI D x)) ≤
      Real.sqrt (2 * (Bd.DE.V.k (xI D x) : ℝ) * Bd.weight (xI D x)) :=
    Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left (hw _) (by positivity))
  have h0 : 0 ≤ Real.sqrt (2 * (Bd.DE.V.k (xI D x) : ℝ) * Bd.DE.weight (xI D x)) + Bd.DE.V.k (xI D x) := by
    positivity
  have hs' : Real.sqrt (2 * (Bd.DE.V.k (xI D x) : ℝ) * Bd.DE.weight (xI D x)) + Bd.DE.V.k (xI D x) ≤
      Real.sqrt (2 * (Bd.DE.V.k (xI D x) : ℝ) * Bd.weight (xI D x)) + Bd.DE.V.k (xI D x) := by linarith
  have hγ : (Inner.ofFlock Bd.DE decI).γ (xI D x) ≤ (Inner.ofFlock Bd.deploy decI).γ (xI D x) := by
    show 16 * (Real.sqrt (2 * (Bd.DE.V.k (xI D x) : ℝ) * Bd.DE.weight (xI D x)) + Bd.DE.V.k (xI D x)) ^ 2 ≤
      16 * (Real.sqrt (2 * (Bd.DE.V.k (xI D x) : ℝ) * Bd.weight (xI D x)) + Bd.DE.V.k (xI D x)) ^ 2
    nlinarith [mul_le_mul hs' hs' h0 (h0.trans hs')]
  have hγ0 : 0 ≤ (Inner.ofFlock Bd.DE decI).γ (xI D x) := by
    show 0 ≤ 16 * (Real.sqrt (2 * (Bd.DE.V.k (xI D x) : ℝ) * Bd.DE.weight (xI D x)) + Bd.DE.V.k (xI D x)) ^ 2
    positivity
  have h5 : Real.sqrt ((Inner.ofFlock Bd.DE decI).γ (xI D x) *
        PrBAbort H Bd.DE.V (Bd.innerPO decI H D DO decO P x decC) (xI D x) (Bd.keepAO decI H D DO decO P x)) ≤
      Real.sqrt ((Inner.ofFlock Bd.deploy decI).γ (xI D x) * PrBI H D (Inner.ofFlock Bd.deploy decI) DO decO P x) :=
    Real.sqrt_le_sqrt (mul_le_mul hγ h3 hBA0 (hγ0.trans hγ))
  have h4 := mul_le_mul_of_nonneg_left (Bound.Pr_badO_le_PrFdev (decO := decO) (P := P) (x := x)
    hB hV hT hL (hFk.toAt x) (hA.toAt x) M) (Nat.cast_nonneg (NI D (Inner.ofFlock Bd.deploy decI) x))
  have h6 := Pr_innerRejects_le_PrBI hA hT hN0 hM decO x P
  have h7 : Real.sqrt ((Inner.ofFlock Bd.deploy decI).γ (xI D x) * PrBI H D (Inner.ofFlock Bd.deploy decI) DO decO P x) +
      PrBI H D (Inner.ofFlock Bd.deploy decI) DO decO P x ≤
      Real.sqrt (γBI D (Inner.ofFlock Bd.deploy decI) x * PrBI H D (Inner.ofFlock Bd.deploy decI) DO decO P x) :=
    sqrt_add_le_γBI (PrBI_nonneg H D (Inner.ofFlock Bd.deploy decI) DO decO P x)
      (PrBI_le_one H D (Inner.ofFlock Bd.deploy decI) DO decO P x)
  have hβ : βdev D (Inner.ofFlock Bd.deploy decI) x *
      (PrFdev H D (Inner.ofFlock Bd.deploy decI) DO decO P x M + 1 / ((M : ℝ) + 1)) =
      NI D (Inner.ofFlock Bd.deploy decI) x * ((Inner.ofFlock Bd.deploy decI).V.k (xI D x) *
        (PrFdev H D (Inner.ofFlock Bd.deploy decI) DO decO P x M + 1 / ((M : ℝ) + 1))) := by
    unfold βdev; ring
  linarith [h2.trans (add_le_add le_rfl h5)]

/-- **The recursion's KS at the bound record, linearly, from Flock's KS at the outer deployment and (P1) at `DE`**,
with no bridge and no `WeightCovers`: `KSLin_of` with `innerLin_of_plainKSAbort_binds` for its inner stage, on
`KS_of_plainKSAbort_binds`' premises. -/
theorem Bound.KSLin_of_plainKSAbort_binds [DecidableEq Col'] [Zero Col'] (decC : Msg → Msg → Coin)
    (hF : Flock.KS H DO) (hKS : (Inner.ofFlock Bd.DE decI).KSAbort H) (hw : ∀ x, Bd.DE.weight x ≤ Bd.weight x)
    (hdec : ∀ x c, decC x (Bd.ans c) = c) (hB : Bd.Binds) (hRef : Refuses H D)
    (hRounds : InnerRounds D (Inner.ofFlock Bd.deploy decI)) (hN : SessionNums D DO) (hε : TableErrNonneg D)
    (hRd : SessionRounds D (Inner.ofFlock Bd.deploy decI) DO) (hS : SessionAccepts D)
    (hRs : SessionRuns H D (Inner.ofFlock Bd.deploy decI) DO) (hV : InnerViewRun H D (Inner.ofFlock Bd.deploy decI) DO)
    (hT : CallsTrue H D (Inner.ofFlock Bd.deploy decI) DO) (hL : CallsLogColl H D (Inner.ofFlock Bd.deploy decI) DO)
    (hFk : ForksBind H D (Inner.ofFlock Bd.deploy decI) DO) (hA : InnerAcceptsColl H D (Inner.ofFlock Bd.deploy decI) DO)
    (hM : RoundsMargin H D (Inner.ofFlock Bd.deploy decI)) (hN0 : CallsNil H D) :
    KSLin H D (Inner.ofFlock Bd.deploy decI) DO :=
  KSLin_of H D (Inner.ofFlock Bd.deploy decI) DO hRef (outerKS_of_flockKS hF hN hε hRd hS hRs)
    fun decO _ x _ _ _ P M => Bound.innerLin_of_plainKSAbort_binds (decO := decO) (P := P) (x := x) decC hKS hw hdec
      hB hRounds hV hT hL hFk hA hM hN0 M

end MergedLin

end Claims.Recursion
