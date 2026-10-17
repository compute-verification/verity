import Proofs.Flock.Claims.KS.Wait
import Proofs.Flock.Claims.Recursion.Abort

/-!
# One wait, truncated, at runs that may abort

`ERep_trunc` over the draws `Inner.PrBadAbort` makes, each a run's coins with its abort coins `(r, a)`, at the runs as
they abort (`abortRun`):

* **Over draws** (`accPrD`, `accFailD`, `waitFailD`): a run's acceptance, `k` independent accepted runs that fail `G`,
  and one wait's failure, over any finite draws.
* **The bad event factors** (`PrBadAbort_le`): `E^rep` fails on a kept accepted run only if every wait fails, and the
  waits are independent given the tape, so `Pr[acc ∧ ¬R(x, E^rep)] ≤ 𝔼_ω[ε_ω · f_ω^m]` over draws.
* **Truncation** (`PrBadAbort_trunc`): one wait's failure is at most a short wait's probability plus the failure on `k`
  accepted runs (`waitFailD_le`, by `wait_couple`, which is over any draws), and `tape_bound`, `wait_short_le` and
  `short_le_exp` give `4·𝔼[a] + 2^{−m} + 2k/K + e^{−k/8}` as for `ERep_trunc`.
-/

namespace Claims.Recursion

open Claims

section Draws

variable {Dr W Op : Type} [Fintype Dr]

/-- `ε`: the probability that the run at a draw is accepted. -/
noncomputable def accPrD (run : Dr → Trace) : ℝ := Pr fun d => (run d).out = .acc

/-- `Q`: `k` independent draws, all accepted, whose views fail `G`. -/
noncomputable def accFailD (run : Dr → Trace) (k : ℕ) (G : List Seen → Prop) : ℝ :=
  Pr fun σ : Fin k → Dr => (∀ i, (run (σ i)).out = .acc) ∧ ¬ G ((List.finRange k).map fun i => (run (σ i)).seen)

/-- `f`: the probability that one wait over `K` fresh draws fails. -/
noncomputable def waitFailD (H : Msg → Msg) (run : Dr → Trace) (k K : ℕ) (opens : Seen → List Op)
    (assemble : List Op → Option W) (check : W → Bool) : ℝ :=
  Pr fun ρ : Fin K → Dr => ¬ WaitOk H run k opens assemble check ((List.finRange K).map ρ)

/-- **One wait's failure over draws**: at most the short-wait probability plus the failure on `k` accepted runs. -/
theorem waitFailD_le [Nonempty Dr] (H : Msg → Msg) (run : Dr → Trace) (k K : ℕ) (opens : Seen → List Op)
    (assemble : List Op → Option W) (check : W → Bool) (hε : 0 < accPrD run) :
    waitFailD H run k K opens assemble check ≤
      Pr (fun ρ : Fin K → Dr => (waitRes run k ((List.finRange K).map ρ) []).length < k) +
        accFailD run k (Assembles opens assemble check) / accPrD run ^ k := by
  unfold waitFailD
  set res := fun ρ : Fin K → Dr => waitRes run k ((List.finRange K).map ρ) []
  have hok : ∀ ρ, WaitOk H run k opens assemble check ((List.finRange K).map ρ) ↔
      Assembles opens assemble check (res ρ) := by
    intro ρ; unfold WaitOk Assembles; rw [waitAcc_res]
  refine (Pr_mono (q := fun ρ => (res ρ).length < k ∨ (k ≤ (res ρ).length ∧
      ¬ Assembles opens assemble check (res ρ))) fun ρ h => ?_).trans ((Pr_or_le _ _).trans ?_)
  · rw [hok] at h
    by_cases hl : (res ρ).length < k
    · exact Or.inl hl
    · exact Or.inr ⟨by omega, h⟩
  · refine add_le_add le_rfl ?_
    rw [le_div_iff₀ (pow_pos hε k), mul_comm]
    have h := wait_couple run k (fun vs => ¬ Assembles opens assemble check vs) K k [] (by simp)
    simp only [List.nil_append] at h
    exact h

/-- **One tape, truncated, over draws**: from one wait's bound on `k` accepted runs, `ε · Q/ε^k ≤ a + rate·ε` with
`rate ≤ 1/4`, `ε · f^m ≤ 4a + 2^{−m} + 2k/K + e^{−k/8}`. -/
theorem tape_trunc [Nonempty Dr] (H : Msg → Msg) (run : Dr → Trace) (m k K : ℕ) (opens : Seen → List Op)
    (assemble : List Op → Option W) (check : W → Bool) (hK : 0 < K) (hk : 8 ≤ k) {a rate : ℝ} (ha : 0 ≤ a)
    (hrate : rate ≤ 1 / 4)
    (hwait : 0 < accPrD run →
      accPrD run * (accFailD run k (Assembles opens assemble check) / accPrD run ^ k) ≤ a + rate * accPrD run) :
    accPrD run * waitFailD H run k K opens assemble check ^ m ≤
      4 * a + (2 ^ (-(m : ℝ)) + 2 * k / K + Real.exp (-(k : ℝ) / 8)) := by
  have hε0 : 0 ≤ accPrD run := Pr_nonneg _
  have hrε : rate * accPrD run ≤ accPrD run / 4 := by
    have := mul_le_mul_of_nonneg_right hrate hε0; linarith
  have h := tape_bound (ε := accPrD run) (f := waitFailD H run k K opens assemble check) (a := a) (m := m)
    hε0 (Pr_le_one _) (Pr_nonneg _) (Pr_le_one _) ha (Pr_nonneg _) hK hk
    (waitFailD_le H run k K opens assemble check)
    (fun hε => (hwait hε).trans (by linarith))
    (fun hεK => (wait_short_le run k K).trans (short_le_exp (Pr_le_one _) _ _ hεK))
  linarith

end Draws

section Abort

variable {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

omit [Nonempty Coin] in
/-- **The bad event factors at runs that may abort**: when `check` is sound for `R`,
`Pr[acc ∧ ¬R(x, E^rep)] ≤ 𝔼_ω[ε_ω · f_ω^m]`, over draws of a run's coins with its abort coins. -/
theorem PrBadAbort_le (H : Msg → Msg) (I : Inner Coin W) (hcheck : ∀ x w, I.check x w = true → I.R x w)
    (x : Msg) {Ω A : Type} [Fintype Ω] [Fintype A] (P : Prover Ω) (keep : Ω → (Fin (I.V.k x) → Coin) → A → Bool) :
    I.PrBadAbort H x P keep ≤ Ex fun ω => accPrD (abortRun H P I.V x keep ω) *
      waitFailD H (abortRun H P I.V x keep ω) (I.k x) (I.cap x) I.opens (I.assemble x) (I.check x) ^ I.waits x := by
  unfold Inner.PrBadAbort accPrD waitFailD
  simp only [← Pr_pi]
  rw [← Pr_indep]
  refine Pr_mono fun s ⟨hacc, hbad⟩ => ⟨hacc, fun i hok => hbad ?_⟩
  simp only [erep, RP.eval, Trace.seen, hacc, ↓reduceIte] at hbad ⊢
  generalize hres : ((firstWait (I.k x) I.opens (I.assemble x) (I.check x)
    ((List.finRange (I.waits x)).map fun i => (List.finRange (I.cap x)).map (s.2.2 i))).eval H
      (abortRun H P I.V x keep s.1)).1 = res at hbad ⊢
  cases res with
  | some w => exact hcheck x w (firstWait_some H _ _ _ _ _ _ w hres)
  | none =>
    exact absurd hok (firstWait_none H _ _ _ _ _ _ hres _ (List.mem_map.2 ⟨i, List.mem_finRange i, rfl⟩))

/-- **`E^rep` at runs that may abort, truncated**: when `check` is sound for `R`, `K > 0`, `k ≥ 8`, and each tape with
acceptance `ε > 0` over draws has one wait's knowledge bound on `k` accepted runs, `ε · Q/ε^k ≤ a + rate·ε` with
`rate ≤ 1/4`, `Pr[acc ∧ ¬R(x, E^rep)] ≤ 4·𝔼[a] + 2^{−m} + 2k/K + e^{−k/8}`. -/
theorem PrBadAbort_trunc (H : Msg → Msg) (I : Inner Coin W) (hcheck : ∀ x w, I.check x w = true → I.R x w)
    (x : Msg) {Ω A : Type} [Fintype Ω] [Nonempty Ω] [Fintype A] [Nonempty A] (P : Prover Ω)
    (keep : Ω → (Fin (I.V.k x) → Coin) → A → Bool) (hK : 0 < I.cap x) (hk : 8 ≤ I.k x) (a : Ω → ℝ)
    (ha : ∀ ω, 0 ≤ a ω) {rate : ℝ} (hrate : rate ≤ 1 / 4)
    (hwait : ∀ ω, 0 < accPrD (abortRun H P I.V x keep ω) →
      accPrD (abortRun H P I.V x keep ω) * (accFailD (abortRun H P I.V x keep ω) (I.k x)
        (Assembles I.opens (I.assemble x) (I.check x)) / accPrD (abortRun H P I.V x keep ω) ^ I.k x) ≤
        a ω + rate * accPrD (abortRun H P I.V x keep ω)) :
    I.PrBadAbort H x P keep ≤
      4 * Ex a + 2 ^ (-(I.waits x : ℝ)) + 2 * I.k x / I.cap x + Real.exp (-(I.k x : ℝ) / 8) := by
  refine (PrBadAbort_le H I hcheck x P keep).trans ?_
  calc Ex (fun ω => accPrD (abortRun H P I.V x keep ω) *
        waitFailD H (abortRun H P I.V x keep ω) (I.k x) (I.cap x) I.opens (I.assemble x) (I.check x) ^ I.waits x)
      ≤ Ex (fun ω => 4 * a ω + (2 ^ (-(I.waits x : ℝ)) + 2 * I.k x / I.cap x + Real.exp (-(I.k x : ℝ) / 8))) :=
        Ex_mono fun ω => tape_trunc H _ (I.waits x) (I.k x) (I.cap x) _ _ _ hK hk (ha ω) hrate (hwait ω)
    _ = 4 * Ex a + 2 ^ (-(I.waits x : ℝ)) + 2 * I.k x / I.cap x + Real.exp (-(I.k x : ℝ) / 8) := by
        rw [Ex_add, Ex_mul_left, Ex_const]; ring

end Abort

end Claims.Recursion
