import Proofs.Flock.Claims.Recursion.BadLin
import Proofs.Flock.Claims.Recursion.KSStated

/-!
# The recursion's KS, linearly

`KSLin` (`Definitions.Claims.RecursionLin`) at the same premises `KS_of_flockKS_binds` reads, which are the deployed
route's (`RecDeploy.ksBoundRoute_of_bridge`). The inner stage fails where `E^rep` fails against `P_I`'s aborting runs,
at a bad rerun, or where the judged draw's inner view rejects (`PrBadGood_le_abort_rej`):

* against `P_I`'s aborting runs, `κ_I + √(γ_I·Pr[B_I])` (`Inner.KSAbort`, `PrBAbort_le_PrBI`), with `B_I` its own
  finder (`Binner` is `BI`, `PrBinner_eq`), not `k(x)·Pr[Be]`;
* at the `N_I` expected reruns, a bad draw at most `k_I·(Pr[Fdev] + 1/(M+1))` (`Pr_bad_le_PrFdev`, from
  `Pr_bad_le_lin`), so `β·(Pr[Fdev] + 1/(M+1))` with `β = N_I·k_I`;
* a rejecting judged draw is a collision at every fork (`Pr_innerRejects_le_fork`), so at most the inner rounds'
  average, `Pr[B_I]` (`Pr_innerRejects_le_PrBI`), which `γ_BI = (√γ_I + 1)²` holds under `B_I`'s root
  (`sqrt_add_le_γBI`).

The outer sessions are `KS_of`'s (`κ` and `√(γ·Pr[B])`). `KSLin_of_flockKS_binds` is the result, and
`KSLin_of_flockKS` the same where the played rounds' openings are recorded (`forksBind_of_opening`).
-/

namespace Claims.Recursion

open Claims

section Coupling

variable {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **`B_I` as its own finder is `B_I`.** -/
theorem PrBinner_eq (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) :
    PrBinner H D I DO decO P x = PrBI H D I DO decO P x :=
  rfl

theorem PrBI_nonneg (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) :
    0 ≤ PrBI H D I DO decO P x := by
  unfold PrBI; split
  · exact Pr_nonneg _
  · exact le_rfl

theorem PrBI_le_one (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) :
    PrBI H D I DO decO P x ≤ 1 := by
  unfold PrBI; split
  · exact Pr_le_one _
  · exact zero_le_one

theorem PrFdev_nonneg (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) (M : ℕ) :
    0 ≤ PrFdev H D I DO decO P x M := by
  unfold PrFdev; split
  · exact Pr_nonneg _
  · exact le_rfl

/-- **A bad draw at most `k_I·(Pr[Fdev] + 1/(M+1))`**: `Pr_bad_le_lin`, and no bad draw where the inner protocol takes no
rounds. -/
theorem Pr_bad_le_PrFdev {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    (hFk : ForksBind H D I DO) (decO : List (ℕ × Col) → List T) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (M : ℕ) :
    Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x => Bad H D I DO decO P x t.1 t.2) ≤
      I.V.k (xI D x) * (PrFdev H D I DO decO P x M + 1 / ((M : ℝ) + 1)) := by
  by_cases hk : 0 < I.V.k (xI D x)
  · have h := Pr_bad_le_lin hFk decO x P M hk
    simp only [PrFdev, hk, ↓reduceDIte]
    rw [mul_add, mul_one_div]
    exact h
  · have h0 : I.V.k (xI D x) = 0 := Nat.eq_zero_of_not_pos hk
    have hb : Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x => Bad H D I DO decO P x t.1 t.2) = 0 :=
      le_antisymm ((Pr_mono fun t ht => ht.2 fun j hj => absurd hj (by omega)).trans_eq Pr_false) (Pr_nonneg _)
    rw [hb, h0, Nat.cast_zero, zero_mul]

/-- **A rejecting judged draw costs at most `Pr[B_I]`**: it is a collision at every fork (`Pr_innerRejects_le_fork`),
so at most their average over the inner rounds, of which an accepted run shows one (`RoundsMargin`). -/
theorem Pr_innerRejects_le_PrBI_at {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W}
    {DO : Flock.Deploy Coin Col T} {x : Msg} (hA : InnerAcceptsCollAt H D I DO x) (hT : CallsTrue H D I DO)
    (hN0 : CallsNil H D) (hM : RoundsMargin H D I) (decO : List (ℕ × Col) → List T) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) :
    Pr (InnerRejects H D I DO decO P x) ≤ PrBI H D I DO decO P x := by
  by_cases hE : Pr (InnerRejects H D I DO decO P x) = 0
  · rw [hE]
    exact PrBI_nonneg H D I DO decO P x
  obtain ⟨t, ws, hg, -⟩ : ∃ t, InnerRejects H D I DO decO P x t := by
    by_contra hn
    exact hE ((Pr_congr fun t => ⟨fun h => hn ⟨t, h⟩, False.elim⟩).trans Pr_false)
  have hkI : 0 < I.V.k (xI D x) := (hM x Ω P t.1 t.2.1 hg.1).1
  have hBI : PrBI H D I DO decO P x =
      (∑ j : Fin (I.V.k (xI D x)), Pr (forkColl H D I DO decO P x j)) / I.V.k (xI D x) := by
    simp only [PrBI, hkI, ↓reduceDIte]
    unfold Finder.Pr
    exact Pr_guess (fun s : (BI H D I DO decO P x hkI).S => ∃ u v, (BI H D I DO decO P x hkI).out s = some (u, v) ∧
      Collision H u v)
  rw [hBI, le_div_iff₀ (by exact_mod_cast hkI)]
  calc Pr (InnerRejects H D I DO decO P x) * I.V.k (xI D x)
      = ∑ _j : Fin (I.V.k (xI D x)), Pr (InnerRejects H D I DO decO P x) := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_comm]
    _ ≤ ∑ j : Fin (I.V.k (xI D x)), Pr (forkColl H D I DO decO P x j) :=
        Finset.sum_le_sum fun j _ => Pr_innerRejects_le_fork_at hA hT hN0 decO P j

theorem Pr_innerRejects_le_PrBI {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    (hA : InnerAcceptsColl H D I DO) (hT : CallsTrue H D I DO) (hN0 : CallsNil H D) (hM : RoundsMargin H D I)
    (decO : List (ℕ × Col) → List T) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    Pr (InnerRejects H D I DO decO P x) ≤ PrBI H D I DO decO P x :=
  Pr_innerRejects_le_PrBI_at (hA.toAt x) hT hN0 hM decO P

/-- `√(g·p) + p ≤ √((√g + 1)²·p)` at a probability `p`. -/
theorem sqrt_add_le_γBI {g p : ℝ} (hp0 : 0 ≤ p) (hp1 : p ≤ 1) :
    Real.sqrt (g * p) + p ≤ Real.sqrt ((Real.sqrt g + 1) ^ 2 * p) := by
  have hs : Real.sqrt (g * p) ≤ Real.sqrt g * Real.sqrt p := by
    by_cases hg : 0 ≤ g
    · rw [Real.sqrt_mul hg]
    · rw [Real.sqrt_eq_zero'.2 (mul_nonpos_of_nonpos_of_nonneg (not_le.1 hg).le hp0)]
      exact mul_nonneg (Real.sqrt_nonneg _) (Real.sqrt_nonneg _)
  have hq : p ≤ Real.sqrt p := by
    have h1 : Real.sqrt p ≤ 1 := (Real.sqrt_le_sqrt hp1).trans_eq Real.sqrt_one
    have h2 := Real.mul_self_sqrt hp0
    nlinarith [Real.sqrt_nonneg p]
  rw [Real.sqrt_mul' ((Real.sqrt g + 1) ^ 2) hp0, Real.sqrt_sq (by positivity)]
  nlinarith [Real.sqrt_nonneg p]

/-- **The inner stage, linearly, where the inner rounds' forks bind**: at most
`κ_I + √(γ_BI·Pr[B_I]) + β·(Pr[Fdev] + 1/(M+1))`, at every `M`. -/
theorem innerLin_of_KSAbort_binds [DecidableEq Col] [Zero Col] {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W}
    {DO : Flock.Deploy Coin Col T} {decC : Msg → Msg → Coin} (hKS : I.KSAbort H)
    (hR : InnerRounds D I) (hV : InnerViewRun H D I DO) (hT : CallsTrue H D I DO) (hL : CallsLogColl H D I DO)
    (hFk : ForksBind H D I DO) (hA : InnerAcceptsColl H D I DO) (hM : RoundsMargin H D I)
    (hpc : PublicCoin H I decC) (hN0 : CallsNil H D) (decO : List (ℕ × Col) → List T) (x : Msg) {Ω : Type}
    [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (M : ℕ) :
    PrBadGood H D.V (innerR D I) outR (EO D I DO decO) (dec H D) (fun x => I.waits (xI D x))
        (fun x => I.k (xI D x)) (fun x => I.cap (xI D x)) I.opens (fun x => I.assemble (xI D x))
        (fun x => I.check (xI D x)) x P ≤
      I.κ (xI D x) + Real.sqrt (γBI D I x * PrBI H D I DO decO P x) +
        βdev D I x * (PrFdev H D I DO decO P x M + 1 / ((M : ℝ) + 1)) := by
  have h1 := PrBadGood_le_abort_rej hV hpc hR decO x P
  have h2 := hKS (xI D x) Ω (innerP H D I DO decO P x decC) ((Fin (D.V.k x) → Coin) × (EO D I DO decO).C x)
    (keepA H D I DO decO P x)
  have h3 := PrBAbort_le_PrBI hL hT hN0 hpc hR decO x P
  have hBA0 : 0 ≤ PrBAbort H I.V (innerP H D I DO decO P x decC) (xI D x) (keepA H D I DO decO P x) := by
    unfold PrBAbort; split
    · exact Pr_nonneg _
    · exact le_rfl
  have h5 : Real.sqrt (I.γ (xI D x) * PrBAbort H I.V (innerP H D I DO decO P x decC) (xI D x)
      (keepA H D I DO decO P x)) ≤ Real.sqrt (I.γ (xI D x) * PrBI H D I DO decO P x) := by
    by_cases hγ : 0 ≤ I.γ (xI D x)
    · exact Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left h3 hγ)
    · rw [Real.sqrt_eq_zero'.2 (mul_nonpos_of_nonpos_of_nonneg (not_le.1 hγ).le hBA0)]
      exact Real.sqrt_nonneg _
  have h4 := mul_le_mul_of_nonneg_left (Pr_bad_le_PrFdev hFk decO x P M) (Nat.cast_nonneg (NI D I x))
  have h6 := Pr_innerRejects_le_PrBI hA hT hN0 hM decO x P
  have h7 : Real.sqrt (I.γ (xI D x) * PrBI H D I DO decO P x) + PrBI H D I DO decO P x ≤
      Real.sqrt (γBI D I x * PrBI H D I DO decO P x) :=
    sqrt_add_le_γBI (PrBI_nonneg H D I DO decO P x) (PrBI_le_one H D I DO decO P x)
  have hβ : βdev D I x * (PrFdev H D I DO decO P x M + 1 / ((M : ℝ) + 1)) =
      NI D I x * (I.V.k (xI D x) * (PrFdev H D I DO decO P x M + 1 / ((M : ℝ) + 1))) := by
    unfold βdev; ring
  linarith [h2.trans (add_le_add le_rfl h5)]

end Coupling

/-! ## The recursion's KS, linearly -/

section RecursionKS

variable {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-- **`KSLin` from the outer sessions' KS and the inner stage's linear bound**: `KS_of`'s split, with the inner stage
at `κ_I + √(γ_BI·Pr[B_I]) + β·(Pr[Fdev] + 1/(M+1))` in place of `InnerKS`. -/
theorem KSLin_of [Zero Col] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (hRef : Refuses H D) (hO : OuterKS H D I DO)
    (hI : ∀ decO, decOK DO decO → ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (M : ℕ),
      PrBadGood H D.V (innerR D I) outR (EO D I DO decO) (dec H D) (fun x => I.waits (xI D x))
          (fun x => I.k (xI D x)) (fun x => I.cap (xI D x)) I.opens (fun x => I.assemble (xI D x))
          (fun x => I.check (xI D x)) x P ≤
        I.κ (xI D x) + Real.sqrt (γBI D I x * PrBI H D I DO decO P x) +
          βdev D I x * (PrFdev H D I DO decO P x M + 1 / ((M : ℝ) + 1))) :
    KSLin H D I DO := by
  intro decO hdec x Ω _ _ P M
  set A : ℝ := (NI D I x : ℝ)
  set p := PrB H D.V P x
  set G := ∑ s : Fin (D.nS x), Real.sqrt ((D.nums x s).γ * D.V.k x)
  have hA : (1 + (I.waits (xI D x) : ℝ) * (I.k (xI D x) : ℝ)) = A := by simp only [A, NI]; push_cast; ring
  have hA0 : 0 ≤ A := Nat.cast_nonneg _
  have hp0 : 0 ≤ p := PrB_nonneg H D.V P x
  have hG0 : 0 ≤ G := Finset.sum_nonneg fun s _ => Real.sqrt_nonneg _
  have hsplit := servedERep_bad_le_good H D.V (R D I) outR (EO D I DO decO) (dec H D) (fun x => I.waits (xI D x))
    (fun x => I.k (xI D x)) (fun x => I.cap (xI D x)) I.opens (fun x => I.assemble (xI D x))
    (fun x => I.check (xI D x)) x P
  rw [hA] at hsplit
  have hgood : PrBadGood H D.V (R D I) outR (EO D I DO decO) (dec H D) (fun x => I.waits (xI D x))
      (fun x => I.k (xI D x)) (fun x => I.cap (xI D x)) I.opens (fun x => I.assemble (xI D x))
      (fun x => I.check (xI D x)) x P ≤
      PrBadGood H D.V (innerR D I) outR (EO D I DO decO) (dec H D) (fun x => I.waits (xI D x))
      (fun x => I.k (xI D x)) (fun x => I.cap (xI D x)) I.opens (fun x => I.assemble (xI D x))
      (fun x => I.check (xI D x)) x P := by
    refine Pr_mono fun t h => ⟨h.1, h.2.1, fun hi => h.2.2 ?_⟩
    obtain ⟨s, hs, hc⟩ := hRef x Ω P t.1 t.2.1 h.1
    exact R_of_innerR D I hs hc _ hi
  have hin := hI decO hdec x Ω P M
  have hEO : PrBad H D.V outR (EO D I DO decO) x P ≤
      ∑ s : Fin (D.nS x), (D.nums x s).κ + G * Real.sqrt p := by
    refine (PrBad_EO_le H D I DO decO x P).trans ?_
    rw [Finset.sum_mul, ← Finset.sum_add_distrib]
    refine Finset.sum_le_sum fun s _ => (hO decO hdec x Ω P s).trans (le_of_eq ?_)
    rw [Real.sqrt_mul (mul_nonneg (Nums.γ_nonneg _) (Nat.cast_nonneg _))]
  have hB : A * (G * Real.sqrt p) = Real.sqrt (γ D I x * p) := by
    have hγ : γ D I x = (A * G) ^ 2 := rfl
    rw [hγ, Real.sqrt_mul' _ hp0, Real.sqrt_sq (mul_nonneg hA0 hG0), mul_assoc]
  have hκ : κ D I x = I.κ (xI D x) + A * ∑ s : Fin (D.nS x), (D.nums x s).κ := rfl
  have hBI : PrBinner H D I DO decO P x = PrBI H D I DO decO P x := PrBinner_eq H D I DO decO P x
  show PrBad H D.V (R D I) (served D.V (EO D I DO decO) (dec H D) (fun x => I.waits (xI D x))
    (fun x => I.k (xI D x)) (fun x => I.cap (xI D x)) I.opens (fun x => I.assemble (xI D x))
    (fun x => I.check (xI D x))) x P ≤ _
  rw [served_eq, hκ, hBI]
  have hmul := mul_le_mul_of_nonneg_left hEO hA0
  nlinarith [hsplit, hgood, hin, hmul, hB]

/-- **The recursion's KS, linearly, from Flock's KS at the outer deployment and the inner protocol's KS against provers
that may abort, where the inner rounds' forks bind**: `KS_of_flockKS_binds`'s premises, `KSLin`'s bound. -/
theorem KSLin_of_flockKS_binds [Zero Col] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W)
    (DO : Flock.Deploy Coin Col T) (decC : Msg → Msg → Coin) (hF : Flock.KS H DO)
    (hKS : I.KSAbort H) (hRef : Refuses H D) (hRounds : InnerRounds D I) (hN : SessionNums D DO)
    (hε : TableErrNonneg D) (hRd : SessionRounds D I DO) (hS : SessionAccepts D) (hR : SessionRuns H D I DO)
    (hV : InnerViewRun H D I DO) (hT : CallsTrue H D I DO) (hL : CallsLogColl H D I DO)
    (hFk : ForksBind H D I DO) (hA : InnerAcceptsColl H D I DO) (hM : RoundsMargin H D I)
    (hpc : PublicCoin H I decC) (hN0 : CallsNil H D) :
    KSLin H D I DO :=
  KSLin_of H D I DO hRef (outerKS_of_flockKS hF hN hε hRd hS hR)
    fun decO _ x _ _ _ P M => innerLin_of_KSAbort_binds hKS hRounds hV hT hL hFk hA hM hpc hN0 decO x P M

/-- **`KSLin_of_flockKS_binds` where the played rounds' openings are recorded** (`forksBind_of_opening`). -/
theorem KSLin_of_flockKS [Zero Col] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W)
    (DO : Flock.Deploy Coin Col T) (opening : Opening) (decC : Msg → Msg → Coin) (hF : Flock.KS H DO)
    (hKS : I.KSAbort H) (hRef : Refuses H D) (hRounds : InnerRounds D I) (hN : SessionNums D DO)
    (hε : TableErrNonneg D) (hRd : SessionRounds D I DO) (hS : SessionAccepts D) (hR : SessionRuns H D I DO)
    (hV : InnerViewRun H D I DO) (hT : CallsTrue H D I DO) (hL : CallsLogColl H D I DO)
    (hO : CallsOpenPlayed H D I DO opening) (hA : InnerAcceptsColl H D I DO) (hM : RoundsMargin H D I)
    (hpc : PublicCoin H I decC) (hN0 : CallsNil H D) :
    KSLin H D I DO :=
  KSLin_of_flockKS_binds H D I DO decC hF hKS hRef hRounds hN hε hRd hS hR hV hT hL (forksBind_of_opening hV hO hT)
    hA hM hpc hN0

end RecursionKS

end Claims.Recursion
