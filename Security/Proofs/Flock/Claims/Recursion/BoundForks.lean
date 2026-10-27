import Proofs.Flock.Claims.Recursion.InnerAcceptsKS

/-!
# The recursion's KS where the inner rounds' forks bind

`KS_of_flockKS_coll` (`InnerAcceptsKS.lean`) reads `CallsOpenPlayed` at two places only, `Pr_bad_le_fork` and
`sum_opensDiffer_le`, and there only through `forkColl_of_opening`: two extractions at a fork at round `j` whose inner
views differ at message `j` show a collision `Be` finds at that fork. `ForksBind` is that conclusion, stated as the
premise. At option (d) the inner views' rounds before the last are the run's (`RecDeploy.ReadersRunPlayed`) and the
last is bound by `IVStar`'s decision (`Bound.LastBinds`), with no `Opening` of a commit string to read, so
`CallsOpenPlayed` has nothing to say there and `ForksBind` is proved directly.

This file is the chain with `ForksBind` in place of `CallsOpenPlayed` (`Pr_bad_le_fork_binds`, `Pr_bad_le_binds`,
`bad_rej_le_binds`, `innerKS_of_KSAbort_binds`, `KS_of_flockKS_binds`), at the same bound.
-/

namespace Claims.Recursion

open Claims

section Coupling

variable {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **The inner rounds' forks bind**: at every draw of `Be`'s fork at round `j` where `E_O` gives witnesses on both
runs and the inner views read off them differ at message `j`, `Be` finds a collision there. -/
def ForksBind (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T) : Prop :=
  ∀ (decO : List (ℕ × Col) → List T) (x : Msg) (Ω : Type) (P : Prover Ω) (j : ℕ)
    (t : Ω × (Fin (D.V.k x) → Coin) × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x × (EO D I DO decO).C x),
    OpensDiffer H D I DO decO P x j t → forkColl H D I DO decO P x j t

/-- **`ForksBind` at the statement `x`**: what the bound at one statement reads of it. -/
def ForksBindAt (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T) (x : Msg) :
    Prop :=
  ∀ (decO : List (ℕ × Col) → List T) (Ω : Type) (P : Prover Ω) (j : ℕ)
    (t : Ω × (Fin (D.V.k x) → Coin) × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x × (EO D I DO decO).C x),
    OpensDiffer H D I DO decO P x j t → forkColl H D I DO decO P x j t

theorem ForksBind.toAt {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    (h : ForksBind H D I DO) (x : Msg) : ForksBindAt H D I DO x :=
  fun decO => h decO x

/-- **A draw is bad at most as `√(k_I·Σ_{j<k_I} Pr[forkColl j])`**: `Pr_bad_le_fork`, from `ForksBind`. -/
theorem Pr_bad_le_fork_binds {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    (hFk : ForksBind H D I DO) (decO : List (ℕ × Col) → List T) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) :
    Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x => Bad H D I DO decO P x t.1 t.2) ≤
      Real.sqrt (I.V.k (xI D x) * ∑ j : Fin (I.V.k (xI D x)), Pr (forkColl H D I DO decO P x j)) := by
  exact (Pr_bad_le_opens H D I DO decO x P).trans (Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left
    (Finset.sum_le_sum fun j _ => Pr_mono fun t => hFk decO x Ω P j t) (Nat.cast_nonneg _)))

/-- **The inner rounds' forks cost at most `k(x)·Pr[Be]`**: they are among the compiled verifier's (`InnerRounds`),
whose forks' collisions average to `Pr[Be]`. -/
theorem sum_forkColl_inner_le {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    (hR : InnerRounds D I) (decO : List (ℕ × Col) → List T) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) :
    ∑ j : Fin (I.V.k (xI D x)), Pr (forkColl H D I DO decO P x j) ≤ D.V.k x * PrBe H D I DO decO P x := by
  have hq : ∀ j, 0 ≤ Pr (forkColl H D I DO decO P x j) := fun j => Pr_nonneg _
  have hle : ∑ j : Fin (I.V.k (xI D x)), Pr (forkColl H D I DO decO P x j) ≤
      ∑ j : Fin (D.V.k x), Pr (forkColl H D I DO decO P x j) := by
    rw [Fin.sum_univ_eq_sum_range (fun j => Pr (forkColl H D I DO decO P x j)),
      Fin.sum_univ_eq_sum_range (fun j => Pr (forkColl H D I DO decO P x j))]
    exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.range_mono (hR x)) fun j _ _ => hq j
  by_cases hk : 0 < D.V.k x
  · rw [PrBe_eq_avg H D I DO decO P x hk, mul_div_cancel₀ _ (by exact_mod_cast hk.ne')]
    exact hle
  · have h0 : D.V.k x = 0 := Nat.eq_zero_of_not_pos hk
    have hI : I.V.k (xI D x) = 0 := Nat.eq_zero_of_le_zero (h0 ▸ hR x)
    have hS : ∑ j : Fin (I.V.k (xI D x)), Pr (forkColl H D I DO decO P x j) = 0 := by
      rw [Fin.sum_univ_eq_sum_range (fun j => Pr (forkColl H D I DO decO P x j)), hI, Finset.sum_range_zero]
    rw [hS]
    exact mul_nonneg (Nat.cast_nonneg _) (PrBe_nonneg H D I DO decO P x)

/-- **A draw is bad at most as `√(k_I·k(x)·Pr[Be])`**: `Pr_bad_le`, from `ForksBind`. -/
theorem Pr_bad_le_binds {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    (hFk : ForksBind H D I DO) (hR : InnerRounds D I) (decO : List (ℕ × Col) → List T) (x : Msg) {Ω : Type}
    [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x => Bad H D I DO decO P x t.1 t.2) ≤
      Real.sqrt (I.V.k (xI D x) * D.V.k x * PrBe H D I DO decO P x) := by
  rw [mul_assoc]
  exact (Pr_bad_le_fork_binds hFk decO x P).trans
    (Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left (sum_forkColl_inner_le hR decO x P) (Nat.cast_nonneg _)))

/-- **Bad reruns and a rejecting judged draw together cost what `InnerKS` holds for bad reruns**: `bad_rej_le`, from
`ForksBind`. -/
theorem bad_rej_le_binds {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    (hFk : ForksBind H D I DO) (hT : CallsTrue H D I DO) (hR : InnerRounds D I) (hA : InnerAcceptsColl H D I DO)
    (hN0 : CallsNil H D) (hM : RoundsMargin H D I)
    (decO : List (ℕ × Col) → List T) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    NI D I x * Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x => Bad H D I DO decO P x t.1 t.2) +
        Pr (InnerRejects H D I DO decO P x) ≤
      NI D I x * Real.sqrt (I.V.k (xI D x) * D.V.k x * PrBe H D I DO decO P x) := by
  have hn0 : (0 : ℝ) ≤ NI D I x := Nat.cast_nonneg _
  by_cases hE : Pr (InnerRejects H D I DO decO P x) = 0
  · rw [hE, add_zero]
    exact mul_le_mul_of_nonneg_left (Pr_bad_le_binds hFk hR decO x P) hn0
  obtain ⟨t, ws, hg, -⟩ : ∃ t, InnerRejects H D I DO decO P x t := by
    by_contra hn
    refine hE ((Pr_congr fun t => ⟨fun h => hn ⟨t, h⟩, False.elim⟩).trans Pr_false)
  obtain ⟨hkI, hk3⟩ := hM x Ω P t.1 t.2.1 hg.1
  let S : ℕ → ℝ := fun j => Pr (forkColl H D I DO decO P x j)
  have hk : 0 < D.V.k x := by omega
  have hES : ∀ j, Pr (InnerRejects H D I DO decO P x) ≤ S j := fun j =>
    Pr_innerRejects_le_fork hA hT hN0 decO x P j
  have htot : (D.V.k x : ℝ) * PrBe H D I DO decO P x = ∑ j ∈ Finset.range (D.V.k x), S j := by
    rw [PrBe_eq_avg H D I DO decO P x hk, mul_div_cancel₀ _ (by exact_mod_cast hk.ne'),
      Fin.sum_univ_eq_sum_range (fun j => Pr (forkColl H D I DO decO P x j))]
  have hsplit : ∑ j ∈ Finset.range (D.V.k x), S j =
      ∑ j ∈ Finset.range (I.V.k (xI D x)), S j + ∑ j ∈ Finset.Ico (I.V.k (xI D x)) (D.V.k x), S j :=
    (Finset.sum_range_add_sum_Ico S (hR x)).symm
  have hB : 3 * Pr (InnerRejects H D I DO decO P x) ≤ ∑ j ∈ Finset.Ico (I.V.k (xI D x)) (D.V.k x), S j := by
    have hcard : (3 : ℝ) ≤ (Finset.Ico (I.V.k (xI D x)) (D.V.k x)).card := by
      rw [Nat.card_Ico]
      exact_mod_cast (by omega : 3 ≤ D.V.k x - I.V.k (xI D x))
    calc 3 * Pr (InnerRejects H D I DO decO P x)
        ≤ (Finset.Ico (I.V.k (xI D x)) (D.V.k x)).card * Pr (InnerRejects H D I DO decO P x) :=
          mul_le_mul_of_nonneg_right hcard (Pr_nonneg _)
      _ = ∑ _j ∈ Finset.Ico (I.V.k (xI D x)) (D.V.k x), Pr (InnerRejects H D I DO decO P x) := by
          rw [Finset.sum_const, nsmul_eq_mul]
      _ ≤ ∑ j ∈ Finset.Ico (I.V.k (xI D x)) (D.V.k x), S j := Finset.sum_le_sum fun j _ => hES j
  have hAa : ∑ j ∈ Finset.range (I.V.k (xI D x)), S j ≤ I.V.k (xI D x) := by
    calc ∑ j ∈ Finset.range (I.V.k (xI D x)), S j ≤ ∑ _j ∈ Finset.range (I.V.k (xI D x)), (1 : ℝ) :=
          Finset.sum_le_sum fun j _ => Pr_le_one _
      _ = I.V.k (xI D x) := by rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one]
  have hA0 : 0 ≤ ∑ j ∈ Finset.range (I.V.k (xI D x)), S j := Finset.sum_nonneg fun j _ => Pr_nonneg _
  have hbad := Pr_bad_le_fork_binds hFk decO x P
  rw [Fin.sum_univ_eq_sum_range (fun j => Pr (forkColl H D I DO decO P x j))] at hbad
  have hn1 : (1 : ℝ) ≤ NI D I x := by
    have : 1 ≤ NI D I x := by simp only [NI]; omega
    exact_mod_cast this
  have hm := sqrt_margin hn1 (by exact_mod_cast hkI : (1 : ℝ) ≤ I.V.k (xI D x)) hA0 hAa (Pr_nonneg _)
    (Pr_le_one _) hB
  calc (NI D I x : ℝ) * Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x =>
        Bad H D I DO decO P x t.1 t.2) + Pr (InnerRejects H D I DO decO P x)
      ≤ NI D I x * Real.sqrt (I.V.k (xI D x) * ∑ j ∈ Finset.range (I.V.k (xI D x)), S j) +
          Pr (InnerRejects H D I DO decO P x) :=
        add_le_add (mul_le_mul_of_nonneg_left hbad hn0) le_rfl
    _ ≤ NI D I x * Real.sqrt (I.V.k (xI D x) * (∑ j ∈ Finset.range (I.V.k (xI D x)), S j +
          ∑ j ∈ Finset.Ico (I.V.k (xI D x)) (D.V.k x), S j)) := hm
    _ = NI D I x * Real.sqrt (I.V.k (xI D x) * D.V.k x * PrBe H D I DO decO P x) := by
        rw [mul_assoc, htot, hsplit]

/-- **`InnerKS` from `Inner.KSAbort` where the inner rounds' forks bind**: `innerKS_of_KSAbort_coll`, from
`ForksBind` (`bad_rej_le_binds`), at the same bound. -/
theorem innerKS_of_KSAbort_binds [DecidableEq Col] [Zero Col] {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W}
    {DO : Flock.Deploy Coin Col T} {decC : Msg → Msg → Coin} (hKS : I.KSAbort H)
    (hR : InnerRounds D I) (hV : InnerViewRun H D I DO) (hT : CallsTrue H D I DO) (hL : CallsLogColl H D I DO)
    (hFk : ForksBind H D I DO) (hA : InnerAcceptsColl H D I DO) (hM : RoundsMargin H D I)
    (hpc : PublicCoin H I decC) (hN0 : CallsNil H D) : InnerKS H D I DO := by
  intro decO _ x Ω _ _ P
  have h1 := PrBadGood_le_abort_rej hV hpc hR decO x P
  have h2 := hKS (xI D x) Ω (innerP H D I DO decO P x decC) ((Fin (D.V.k x) → Coin) × (EO D I DO decO).C x)
    (keepA H D I DO decO P x)
  have h3 := PrBAbort_le_PrBI hL hT hN0 hpc hR decO x P
  have h4 := bad_rej_le_binds hFk hT hR hA hN0 hM decO x P
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
  linarith [h2.trans (add_le_add le_rfl h5)]

end Coupling

section RecursionKS

variable {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-- **The recursion's KS from Flock's KS at the outer deployment and the inner protocol's KS against provers that may
abort, where the inner rounds' forks bind**: `KS_of_flockKS_coll` with `ForksBind` for `CallsOpenPlayed`. -/
theorem KS_of_flockKS_binds [Zero Col] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W)
    (DO : Flock.Deploy Coin Col T) (decC : Msg → Msg → Coin) (hF : Flock.KS H DO)
    (hKS : I.KSAbort H) (hRef : Refuses H D) (hRounds : InnerRounds D I) (hN : SessionNums D DO)
    (hε : TableErrNonneg D) (hRd : SessionRounds D I DO) (hS : SessionAccepts D) (hR : SessionRuns H D I DO)
    (hV : InnerViewRun H D I DO) (hT : CallsTrue H D I DO) (hL : CallsLogColl H D I DO)
    (hFk : ForksBind H D I DO) (hA : InnerAcceptsColl H D I DO) (hM : RoundsMargin H D I)
    (hpc : PublicCoin H I decC) (hN0 : CallsNil H D) :
    KSA H D I DO :=
  KS_of H D I DO hRef hRounds (outerKS_of_flockKS hF hN hε hRd hS hR)
    (innerKS_of_KSAbort_binds hKS hRounds hV hT hL hFk hA hM hpc hN0)

end RecursionKS

end Claims.Recursion
