module

public import Proofs.Flock.Claims.Recursion.BoundMerged
public import Proofs.Flock.Claims.Recursion.BoundMergedLin
public import Proofs.Flock.Claims.Recursion.PlurLin
public import Proofs.Flock.Claims.Recursion.VCallsAt
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.Prob

/-!
# `Fdev`'s link over the samples' evaluations

`Bound.Pr_badO_le_PrFdev` reads `InnerAcceptsCollAt` and `ForksBindAt` at the two samples its link forks. With their
twins over the samples' evaluations (`InnerAcceptsCollAtV`, `ForksBindAtV`), the link's collision is one among the two
samples' evaluations (`drawCalls`), which `FdevV`'s output searches:

* `Bound.fork_collOV`: `Bound.fork_collO` with the escapes `drawCalls` holds, through `Bound.fork_collO_coreBy` at
  each sample's `callsAt`;
* `Bound.PrBAbort_le_PrBinnerV`: `B` at `P_O`'s aborting runs is at most `B_I` over the samples' evaluations;
* `Bound.linkOutV_collO`, `Bound.Pr_badO_le_linV`, `Bound.Pr_badO_le_PrFdevV`: `P_O`'s bad samples at most
  `k_I·(Pr[FdevV] + 1/(M+1))`.
-/

@[expose] public section

namespace Claims.Recursion

open Claims

section MergedLinV

variable {Coin Col T Col' T' : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
  [DecidableEq Col'] {Bd : Bound Coin Col T} {decI : List (ℕ × Col) → List T} {H : Msg → Msg} {D : Deploy Coin}
  {DO : Flock.Deploy Coin Col' T'} {decO : List (ℕ × Col') → List T'} {Ω : Type} {P : Prover Ω} {x : Msg}

/-- **Two samples forked at inner round `j`, with witnesses, whose openings differ at `j`, collide among their
evaluations.** A rejecting inner view is a collision among its sample's evaluations (`InnerAcceptsCollAtV`). Inner views
that differ at message `j` are `ForksBindAtV`'s. Inner views that agree at message `j` are `fork_collO_core`'s,
whose collision among the recorded calls is one among the evaluations. -/
theorem Bound.fork_collOV (hB : Bd.Binds) (hV : InnerViewRun H D (Inner.ofFlock Bd.deploy decI) DO)
    (hT : CallsTrue H D (Inner.ofFlock Bd.deploy decI) DO) (hL : CallsLogCollAt H D (Inner.ofFlock Bd.deploy decI) DO)
    (hN0 : CallsNil H D) (hFk : ForksBindAtV H D (Inner.ofFlock Bd.deploy decI) DO x)
    (hA : InnerAcceptsCollAtV H D (Inner.ofFlock Bd.deploy decI) DO x)
    (j : ℕ) (hj : j < Bd.DE.V.k (xI D x))
    (t : Ω × (Fin (D.V.k x) → Coin) × (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x ×
      (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x)
    (h : (eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x t.1 (t.2.1, t.2.2.2.1)).isSome = true ∧
      (eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x t.1 (fork j t.2.1 t.2.2.1, t.2.2.2.2)).isSome = true ∧
      Bd.oMsg decI H D DO decO P x t.1 (t.2.1, t.2.2.2.1) j ≠
        Bd.oMsg decI H D DO decO P x t.1 (fork j t.2.1 t.2.2.1, t.2.2.2.2) j) :
    forkCollV H D (Inner.ofFlock Bd.deploy decI) DO decO P x j t := by
  obtain ⟨hg₁, hg₂, hne⟩ := h
  obtain ⟨ws, hws⟩ := Option.isSome_iff_exists.1 hg₁
  obtain ⟨ws', hws'⟩ := Option.isSome_iff_exists.1 hg₂
  have hE₁ : EOGives H D (Inner.ofFlock Bd.deploy decI) DO decO P x t.1 t.2.1 t.2.2.2.1 ws := eoGives_of_some hws
  have hE₂ : EOGives H D (Inner.ofFlock Bd.deploy decI) DO decO P x t.1 (fork j t.2.1 t.2.2.1) t.2.2.2.2 ws' :=
    eoGives_of_some hws'
  have hl₁ := drawCalls_eq hE₁
  have hl₂ := drawCalls_eq hE₂
  have hlift : ∀ l : List (Msg × Msg),
      (∀ p ∈ l, p ∈ drawCalls H D (Inner.ofFlock Bd.deploy decI) DO decO P x t.1 (t.2.1, t.2.2.2.1) ++
        drawCalls H D (Inner.ofFlock Bd.deploy decI) DO decO P x t.1 (fork j t.2.1 t.2.2.1, t.2.2.2.2)) →
      (findColl l).isSome → forkCollV H D (Inner.ofFlock Bd.deploy decI) DO decO P x j t :=
    fun l hsub hs => VStar.coll_of_sub (drawCalls_pair_true hT hN0 _ _ _) hsub hs
  have hlift' : ∀ l : List (Msg × Msg),
      (∀ p ∈ l, p ∈ D.callsAt H x t.2.1 ws ++ D.callsAt H x (fork j t.2.1 t.2.2.1) ws') → (findColl l).isSome →
      forkCollV H D (Inner.ofFlock Bd.deploy decI) DO decO P x j t := fun l hsub hs =>
    hlift l (fun p hp => (List.mem_append.1 (hsub p hp)).elim
      (fun h => List.mem_append_left _ (by rw [hl₁]; exact List.mem_append_left _ h))
      (fun h => List.mem_append_right _ (by rw [hl₂]; exact List.mem_append_left _ h))) hs
  rcases hA decO Ω P t.1 t.2.1 t.2.2.2.1 ws hE₁ with hacc₁ | ⟨_, _, huv, _⟩
  swap
  · exact hlift _ (fun p hp => List.mem_append_left _ (by rw [hl₁]; exact mem_callsAt_append hp))
      (Option.isSome_iff_exists.2 ⟨_, huv⟩)
  rcases hA decO Ω P t.1 (fork j t.2.1 t.2.2.1) t.2.2.2.2 ws' hE₂ with hacc₂ | ⟨_, _, huv, _⟩
  swap
  · exact hlift _ (fun p hp => List.mem_append_right _ (by rw [hl₂]; exact mem_callsAt_append hp))
      (Option.isSome_iff_exists.2 ⟨_, huv⟩)
  by_cases hraw : ((D.innerView H x t.2.1 ws).msgs.map Prod.fst).getD j [] =
      ((D.innerView H x (fork j t.2.1 t.2.2.1) ws').msgs.map Prod.fst).getD j []
  swap
  · exact hFk decO Ω P j t ⟨ws, ws', hE₁, hE₂, hraw⟩
  exact Bound.fork_collO_coreBy (cl := fun x r ws => D.callsAt H x r ws) hB hV
    (fun ω r c ws hg => callsAt_true_of (hT decO x Ω P ω r c ws hg)) hL j hj t hws hws' hacc₁ hacc₂ hraw hne hlift'

/-- **A link that succeeds at `P_O`'s reading names a collision among its samples' evaluations**: `linkOut_collO`, with
`fork_collOV`, the pair `linkOutV` outputs. -/
theorem Bound.linkOutV_collO (hB : Bd.Binds) (hV : InnerViewRun H D (Inner.ofFlock Bd.deploy decI) DO)
    (hT : CallsTrue H D (Inner.ofFlock Bd.deploy decI) DO) (hL : CallsLogCollAt H D (Inner.ofFlock Bd.deploy decI) DO)
    (hN0 : CallsNil H D) (hFk : ForksBindAtV H D (Inner.ofFlock Bd.deploy decI) DO x)
    (hA : InnerAcceptsCollAtV H D (Inner.ofFlock Bd.deploy decI) DO x)
    {M : ℕ} (j : Fin ((Inner.ofFlock Bd.deploy decI).V.k (xI D x))) (ω : Ω) (r : Fin (D.V.k x) → Coin)
    (u : (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x)
    (us : Fin M → (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x)
    (hg : (eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x ω (fork j r u.1, u.2)).isSome = true)
    (hd : firstDiffers
      (fun v => (eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x ω (fork j r v.1, v.2)).isSome = true)
      (fun v => Bd.oMsg decI H D DO decO P x ω (fork j r v.1, v.2) j)
      (Bd.oMsg decI H D DO decO P x ω (fork j r u.1, u.2) j) M us) :
    ∃ a b, linkOutV H D (Inner.ofFlock Bd.deploy decI) DO decO P x
        ((j, ω, r, u, us) : LinCoins D (Inner.ofFlock Bd.deploy decI) DO decO Ω x M) = some (a, b) ∧
      Collision H a b := by
  obtain ⟨v, hv, hgv, hfv⟩ := firstPick_of_firstDiffers M us hd
  have hfc := Bound.fork_collOV (decO := decO) (P := P) (x := x) hB hV hT hL hN0 hFk hA (j : ℕ) j.isLt
    (ω, fork j r u.1, v.1, u.2, v.2)
    ⟨hg, by simp only [fork_fork]; exact hgv, by simp only [fork_fork]; exact Ne.symm hfv⟩
  simp only [forkCollV, fork_fork] at hfc
  have hv' : firstPick (linkG H D (Inner.ofFlock Bd.deploy decI) DO decO P x j ω r) M us = some v := hv
  have hg' : linkG H D (Inner.ofFlock Bd.deploy decI) DO decO P x j ω r u := hg
  simp only [linkOutV, hg', hv', ↓reduceIte, Option.getD_some]
  exact hfc

/-- **`P_O`'s bad samples, linearly, over the samples' evaluations**: `Pr_badO_le_lin` with `FdevV`. -/
theorem Bound.Pr_badO_le_linV (hB : Bd.Binds) (hV : InnerViewRun H D (Inner.ofFlock Bd.deploy decI) DO)
    (hT : CallsTrue H D (Inner.ofFlock Bd.deploy decI) DO) (hL : CallsLogCollAt H D (Inner.ofFlock Bd.deploy decI) DO)
    (hN0 : CallsNil H D) (hFk : ForksBindAtV H D (Inner.ofFlock Bd.deploy decI) DO x)
    (hA : InnerAcceptsCollAtV H D (Inner.ofFlock Bd.deploy decI) DO x)
    [Fintype Ω] [Nonempty Ω] (M : ℕ) (hk : 0 < (Inner.ofFlock Bd.deploy decI).V.k (xI D x)) :
    Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x =>
        Bd.BadO decI H D DO decO P x t.1 t.2) ≤
      (Inner.ofFlock Bd.deploy decI).V.k (xI D x) * (FdevV H D (Inner.ofFlock Bd.deploy decI) DO decO P x M hk).Pr H +
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
        ∃ a b, linkOutV H D (Inner.ofFlock Bd.deploy decI) DO decO P x
          ((j, t) : LinCoins D (Inner.ofFlock Bd.deploy decI) DO decO Ω x M) = some (a, b) ∧ Collision H a b) +
        1 / ((M : ℝ) + 1) := fun j => by
    have hp := Pr_plur_fork_lin (j : ℕ)
      (fun ω d => (eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x ω d).isSome = true)
      (fun ω d => Bd.oMsg decI H D DO decO P x ω d j) M
    refine (le_of_eq (Pr_congr fun t => ?_)).trans (hp.trans (add_le_add (Pr_mono fun t ht => ?_) le_rfl))
    · simp only [Bound.opAtO, fork_preCoins]
    · exact Bound.linkOutV_collO hB hV hT hL hN0 hFk hA j t.1 t.2.1 t.2.2.1 t.2.2.2 ht.1 ht.2
  have hF : ((Inner.ofFlock Bd.deploy decI).V.k (xI D x) : ℝ) *
      (FdevV H D (Inner.ofFlock Bd.deploy decI) DO decO P x M hk).Pr H =
      ∑ j : Fin ((Inner.ofFlock Bd.deploy decI).V.k (xI D x)), Pr (fun t : Ω × (Fin (D.V.k x) → Coin) ×
          ((Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) ×
          (Fin M → (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) =>
        ∃ a b, linkOutV H D (Inner.ofFlock Bd.deploy decI) DO decO P x
          ((j, t) : LinCoins D (Inner.ofFlock Bd.deploy decI) DO decO Ω x M) = some (a, b) ∧ Collision H a b) := by
    have hP : (FdevV H D (Inner.ofFlock Bd.deploy decI) DO decO P x M hk).Pr H =
        Ex fun j : Fin ((Inner.ofFlock Bd.deploy decI).V.k (xI D x)) =>
          Pr fun t : Ω × (Fin (D.V.k x) → Coin) ×
              ((Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) ×
              (Fin M → (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) =>
            ∃ a b, linkOutV H D (Inner.ofFlock Bd.deploy decI) DO decO P x
              ((j, t) : LinCoins D (Inner.ofFlock Bd.deploy decI) DO decO Ω x M) = some (a, b) ∧ Collision H a b :=
      Pr_prod (fun s : LinCoins D (Inner.ofFlock Bd.deploy decI) DO decO Ω x M =>
        ∃ a b, linkOutV H D (Inner.ofFlock Bd.deploy decI) DO decO P x s = some (a, b) ∧ Collision H a b)
    rw [hP]
    unfold Ex
    rw [Fintype.card_fin, mul_div_cancel₀ _ (by exact_mod_cast hk.ne')]
  calc Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x =>
        Bd.BadO decI H D DO decO P x t.1 t.2)
      ≤ ∑ j : Fin ((Inner.ofFlock Bd.deploy decI).V.k (xI D x)), (Pr (fun t : Ω × (Fin (D.V.k x) → Coin) ×
          ((Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) ×
          (Fin M → (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) =>
        ∃ a b, linkOutV H D (Inner.ofFlock Bd.deploy decI) DO decO P x
          ((j, t) : LinCoins D (Inner.ofFlock Bd.deploy decI) DO decO Ω x M) = some (a, b) ∧ Collision H a b) +
          1 / ((M : ℝ) + 1)) :=
        h1.trans (Finset.sum_le_sum fun j _ => h2 j)
    _ = (Inner.ofFlock Bd.deploy decI).V.k (xI D x) *
          (FdevV H D (Inner.ofFlock Bd.deploy decI) DO decO P x M hk).Pr H +
          (Inner.ofFlock Bd.deploy decI).V.k (xI D x) / ((M : ℝ) + 1) := by
        rw [Finset.sum_add_distrib, ← hF, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
          mul_one_div]

/-- **`P_O`'s bad samples at most `k_I·(Pr[FdevV] + 1/(M+1))`**: `Pr_badO_le_linV`, and no bad sample where the inner
protocol takes no rounds. -/
theorem Bound.Pr_badO_le_PrFdevV (hB : Bd.Binds) (hV : InnerViewRun H D (Inner.ofFlock Bd.deploy decI) DO)
    (hT : CallsTrue H D (Inner.ofFlock Bd.deploy decI) DO) (hL : CallsLogCollAt H D (Inner.ofFlock Bd.deploy decI) DO)
    (hN0 : CallsNil H D) (hFk : ForksBindAtV H D (Inner.ofFlock Bd.deploy decI) DO x)
    (hA : InnerAcceptsCollAtV H D (Inner.ofFlock Bd.deploy decI) DO x)
    [Fintype Ω] [Nonempty Ω] (M : ℕ) :
    Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x =>
        Bd.BadO decI H D DO decO P x t.1 t.2) ≤
      (Inner.ofFlock Bd.deploy decI).V.k (xI D x) *
        (PrFdevV H D (Inner.ofFlock Bd.deploy decI) DO decO P x M + 1 / ((M : ℝ) + 1)) := by
  by_cases hk : 0 < (Inner.ofFlock Bd.deploy decI).V.k (xI D x)
  · have h := Bound.Pr_badO_le_linV (decO := decO) (P := P) (x := x) hB hV hT hL hN0 hFk hA M hk
    simp only [PrFdevV, hk, ↓reduceDIte]
    rw [mul_add, mul_one_div]
    exact h
  · have h0 : (Inner.ofFlock Bd.deploy decI).V.k (xI D x) = 0 := Nat.eq_zero_of_not_pos hk
    have h0' : Bd.DE.V.k (xI D x) = 0 := h0
    have hb : Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x =>
        Bd.BadO decI H D DO decO P x t.1 t.2) = 0 :=
      le_antisymm ((Pr_mono fun t ht => ht.2 fun j hj => absurd hj (by omega)).trans_eq Pr_false) (Pr_nonneg _)
    rw [hb, h0, Nat.cast_zero, zero_mul]

/-- **`B` at `P_O`'s aborting runs finds only collisions `B_I` finds over the samples' evaluations**: `PrBAbort_le_by` at
the calls a finder searches at a sample (`Deploy.callsAt`, `CallsLogCollAt`), which the samples' evaluations hold
(`drawCalls`). -/
theorem Bound.PrBAbort_le_PrBinnerV (decC : Msg → Msg → Coin) (hdec : ∀ x c, decC x (Bd.ans c) = c)
    (hV : InnerViewRun H D (Inner.ofFlock Bd.deploy decI) DO)
    (hL : CallsLogCollAt H D (Inner.ofFlock Bd.deploy decI) DO)
    (hT : CallsTrue H D (Inner.ofFlock Bd.deploy decI) DO) (hN0 : CallsNil H D)
    (hR : InnerRounds D (Inner.ofFlock Bd.deploy decI)) [Fintype Ω] [Nonempty Ω] :
    PrBAbort H Bd.DE.V (Bd.innerPO decI H D DO decO P x decC) (xI D x) (Bd.keepAO decI H D DO decO P x) ≤
      PrBinnerV H D (Inner.ofFlock Bd.deploy decI) DO decO P x := by
  by_cases hk : 0 < Bd.DE.V.k (xI D x)
  · have hk' : 0 < (Inner.ofFlock Bd.deploy decI).V.k (xI D x) := hk
    refine (Bound.PrBAbort_le_by (cl := fun x r ws => D.callsAt H x r ws) decC hdec hV hL
      (fun ω d => callsAt_true hT hN0 ω d) hR hk).trans ?_
    simp only [PrBinnerV, hk', ↓reduceDIte]
    unfold Finder.Pr
    refine Pr_mono fun s hs => ?_
    obtain ⟨u, v, huv, -⟩ := hs
    refine VStar.coll_of_sub (drawCalls_pair_true hT hN0 _ _ _) (fun p hp => ?_) (Option.isSome_iff_exists.2 ⟨_, huv⟩)
    rcases List.mem_append.1 hp with h | h
    · exact List.mem_append_left _ (List.mem_append_left _ h)
    · exact List.mem_append_right _ (List.mem_append_left _ h)
  · unfold PrBAbort
    simp only [hk, ↓reduceDIte]
    exact PrBinnerV_nonneg H D (Inner.ofFlock Bd.deploy decI) DO decO P x

end MergedLinV

end Claims.Recursion
