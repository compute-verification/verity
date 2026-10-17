import Proofs.Flock.Claims.Recursion.OuterKS

/-!
# The inner protocol's KS against provers that may abort

`E*` sees each inner rerun through `E_O` at fresh outer coins, rejected wherever the compiled run rejects there. The outer
sessions come after the inner rounds, so the compiled prover decides there, on the inner coins and on coins the inner
verifier never reads, whether `E*` sees an inner rerun accepted (`Recursion/Gap.lean`). The recursion's `KS` rests on the
inner protocol's KS against such provers, `Inner.KSAbort`:

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

/-! ## The abort model -/

/-- An aborted run: nothing sent, rejected. -/
def abortTrace : Trace := ⟨[], [], [], .rej⟩

/-- **A run that may abort after the inner rounds**: `P`'s run at the tape `ω` and the coins `t.1`, unless `keep`, which
reads the tape, every coin of the run and the run's fresh coins `t.2`, refuses it; then the run aborts. -/
def abortRun {Coin : Type} [Inhabited Coin] {Ω A : Type} (H : Msg → Msg) (P : Prover Ω) (V : Verifier Coin)
    (x : Msg) (keep : Ω → (Fin (V.k x) → Coin) → A → Bool) (ω : Ω) (t : (Fin (V.k x) → Coin) × A) : Trace :=
  if keep ω t.1 t.2 then run H P V x ω t.1 else abortTrace

/-- **`Pr[run = acc ∧ ¬R(x, E^rep(x))]` at a prover that may abort**, over the tape, the judged run's coins with its
abort coins, and `E^rep`'s draws, each a run's coins with its abort coins: `E^rep` at the inner protocol's waits reruns at
the judged draw, then at its draws, and sees each run as it aborts. -/
noncomputable def Inner.PrBadAbort {Coin W : Type} [Fintype Coin] [Inhabited Coin] (H : Msg → Msg)
    (I : Inner Coin W) (x : Msg) {Ω A : Type} [Fintype Ω] [Fintype A] (P : Prover Ω)
    (keep : Ω → (Fin (I.V.k x) → Coin) → A → Bool) : ℝ :=
  Pr fun s : Ω × ((Fin (I.V.k x) → Coin) × A) × (Fin (I.waits x) → Fin (I.cap x) → (Fin (I.V.k x) → Coin) × A) =>
    (abortRun H P I.V x keep s.1 s.2.1).out = .acc ∧
      ¬ I.R.ext x ((erep (I.waits x) (I.k x) (I.cap x) I.opens (I.assemble x) (I.check x) s.2.1 s.2.2).eval H
        (abortRun H P I.V x keep s.1)).1

/-- **`B` at runs that may abort**: `B`'s fork (guess `j`, draw `ω, r_1..r_j`, run twice from the shared start with
fresh continuations), each run with its own abort coins, and the first collision among the verifier's SHA-512 calls in
the two runs as they abort. Its cost is both runs' calls, both parties'. -/
def BAbort {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {Ω A : Type} [Fintype Ω] [Nonempty Ω]
    [Fintype A] [Nonempty A] (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω) (x : Msg)
    (keep : Ω → (Fin (V.k x) → Coin) → A → Bool) (hk : 0 < V.k x) : Finder where
  S := BCoins Ω Coin (V.k x) × A × A
  ne := ⟨(⟨⟨0, hk⟩, Classical.arbitrary _, fun _ => default, fun _ => default⟩, Classical.arbitrary _,
    Classical.arbitrary _)⟩
  out s := findColl ((abortRun H P V x keep s.1.2.1 (s.1.2.2.1, s.2.1)).vLog ++
    (abortRun H P V x keep s.1.2.1 (forkCoins s.1.1 s.1.2.2.1 s.1.2.2.2, s.2.2)).vLog)
  cost s :=
    (abortRun H P V x keep s.1.2.1 (s.1.2.2.1, s.2.1)).pLog.length +
      (abortRun H P V x keep s.1.2.1 (s.1.2.2.1, s.2.1)).vLog.length +
      (abortRun H P V x keep s.1.2.1 (forkCoins s.1.1 s.1.2.2.1 s.1.2.2.2, s.2.2)).pLog.length +
      (abortRun H P V x keep s.1.2.1 (forkCoins s.1.1 s.1.2.2.1 s.1.2.2.2, s.2.2)).vLog.length

/-- `Pr[B^{P*}(x) outputs a collision]` at runs that may abort, `0` for a verifier that takes no rounds at `x`. -/
noncomputable def PrBAbort {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {Ω A : Type} [Fintype Ω]
    [Nonempty Ω] [Fintype A] [Nonempty A] (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω) (x : Msg)
    (keep : Ω → (Fin (V.k x) → Coin) → A → Bool) : ℝ :=
  if hk : 0 < V.k x then (BAbort H V P x keep hk).Pr H else 0

/-- **The inner protocol's KS against provers that may abort**: for every prover `P*` (any finite tape) and every way it
may abort after the inner rounds (`keep`, on any finite fresh coins `A` per run),
`Pr[run = acc ∧ ¬R(x, E^rep(x))] ≤ κ(x) + √(γ(x) · Pr[B^{P*}(x) outputs a collision])`, `E^rep` and `B` seeing each run
as it aborts. With no aborts it is `Inner.KS` (`KSAbort_KS`). -/
def Inner.KSAbort {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg)
    (I : Inner Coin W) : Prop :=
  ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (A : Type) [Fintype A] [Nonempty A]
    (keep : Ω → (Fin (I.V.k x) → Coin) → A → Bool),
    I.PrBadAbort H x P keep ≤ I.κ x + Real.sqrt (I.γ x * PrBAbort H I.V P x keep)

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
