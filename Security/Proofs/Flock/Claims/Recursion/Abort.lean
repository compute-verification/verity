import Proofs.Flock.Claims.Recursion.OuterKS

/-!
# The inner protocol's KS against provers that may abort

`E*` sees each inner rerun through `E_O` at fresh outer coins, rejected wherever the compiled run rejects there. The
outer sessions come after the inner rounds, so the compiled prover decides there, on the inner coins and on coins the
inner verifier never reads, whether `E*` sees an inner rerun accepted (`Recursion/Gap.lean`). The recursion's `KS` rests
on the inner protocol's KS against such provers, `Inner.KSAbort`, which the claims spec states with its abort model
(`Definitions/Claims/Recursion.lean`):

* **The abort model** (`abortRun`): after the inner rounds, a prover may abort its run, reading its tape, every coin of
  the run and fresh coins of its own for that run (`A`), which the inner verifier doesn't read. An aborted run sends
  nothing and is rejected (`abortTrace`). Its messages are the prover's, so they read only the history.
* **The claim** (`Inner.KSAbort`): `Inner.KS`'s inequality at every such prover. `E^rep` reruns at draws of the run's
  coins with fresh abort coins, the judged draw first, and sees each run as it aborts (`Inner.PrBadAbort`), so an
  aborted rerun is one it doesn't see accepted. `B` sees the two runs as they abort, each with its own abort coins
  (`BAbort`, `PrBAbort`).
* With no aborts (`A = Unit`, nothing refused) it is `Inner.KS` (`KSAbort_KS`).
-/

namespace Claims.Recursion

open Claims

/-! ## With no aborts, `Inner.KS` -/

theorem erep_eq {D W Op : Type} (m k K : ℕ) (opens : Seen → List Op) (assemble : List Op → Option W)
    (check : W → Bool) (d₀ : D) (ρ : Fin m → Fin K → D) :
    erep m k K opens assemble check d₀ ρ = erepProg m k K opens assemble check d₀ ρ := rfl

/-- **`E^rep` over relabeled draws**: `E^rep` at the draws `μ d₀` and `μ ∘ ρ` against `run₂` is `E^rep` at `d₀` and `ρ`
against `run₂ ∘ μ`. -/
theorem erepProg_eval_comp {D₁ D₂ W Op : Type} (H : Msg → Msg) (run₂ : D₂ → Trace) (μ : D₁ → D₂) (m k K : ℕ)
    (opens : Seen → List Op) (assemble : List Op → Option W) (check : W → Bool) (d₀ : D₁) (ρ : Fin m → Fin K → D₁) :
    ((erepProg m k K opens assemble check (μ d₀) (fun i j => μ (ρ i j))).eval H run₂).1 =
      ((erepProg m k K opens assemble check d₀ ρ).eval H (fun d => run₂ (μ d))).1 := by
  simp only [erepProg, RP.eval]
  split
  · rw [firstWait_eval_map H (fun d => run₂ (μ d)) run₂ μ k opens assemble check _ fun _ _ _ _ => rfl]
    simp only [List.map_map, Function.comp_def]
  · rfl

theorem abortRun_true {Coin : Type} [Inhabited Coin] {Ω A : Type} (H : Msg → Msg) (P : Prover Ω)
    (V : Verifier Coin) (x : Msg) (ω : Ω) (t : (Fin (V.k x) → Coin) × A) :
    abortRun H P V x (fun _ _ _ => true) ω t = run H P V x ω t.1 := rfl

theorem PrBadAbort_true {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg)
    (I : Inner Coin W) (x : Msg) {Ω : Type} [Fintype Ω] (P : Prover Ω) :
    I.PrBadAbort H x P (A := Unit) (fun _ _ _ => true) = PrBad H I.V I.R I.E x P := by
  let e : Ω × (Fin (I.V.k x) → Coin) × (Fin (I.waits x) → Fin (I.cap x) → Fin (I.V.k x) → Coin) ≃
      Ω × ((Fin (I.V.k x) → Coin) × Unit) × (Fin (I.waits x) → Fin (I.cap x) → (Fin (I.V.k x) → Coin) × Unit) :=
    { toFun := fun s => (s.1, (s.2.1, ()), fun i j => (s.2.2 i j, ()))
      invFun := fun s => (s.1, s.2.1.1, fun i j => (s.2.2 i j).1)
      left_inv := fun _ => rfl
      right_inv := fun _ => rfl }
  unfold Inner.PrBadAbort
  rw [← Pr_equiv e]
  refine Pr_congr fun s => ?_
  have h := erepProg_eval_comp H (abortRun H P I.V x (fun _ _ _ => true) s.1) (fun r => (r, ())) (I.waits x) (I.k x)
    (I.cap x) I.opens (I.assemble x) (I.check x) s.2.1 s.2.2
  simp only [abortRun_true] at h
  simp only [e, Equiv.coe_fn_mk, abortRun_true, erep_eq, h]
  rfl

theorem PrBAbort_true {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω) (x : Msg) :
    PrBAbort H V P x (A := Unit) (fun _ _ _ => true) = PrB H V P x := by
  unfold PrBAbort PrB
  split
  · rename_i hk
    let e : BCoins Ω Coin (V.k x) ≃ (BAbort H V P x (A := Unit) (fun _ _ _ => true) hk).S :=
      { toFun := fun s => (s, (), ())
        invFun := fun s => s.1
        left_inv := fun _ => rfl
        right_inv := fun _ => rfl }
    unfold Finder.Pr
    rw [← Pr_equiv e]
    rfl
  · rfl

/-- **With no aborts, `Inner.KSAbort` is `Inner.KS`**: at fresh coins `Unit` and a prover that never aborts, its runs
are the prover's, `E^rep`'s draws are the run's coins, and `B` is `B`. -/
theorem KSAbort_KS {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg)
    (I : Inner Coin W) (h : I.KSAbort H) : I.KS H := by
  intro x Ω _ _ P
  have hx := h x Ω P Unit fun _ _ _ => true
  rwa [PrBadAbort_true, PrBAbort_true] at hx

end Claims.Recursion
