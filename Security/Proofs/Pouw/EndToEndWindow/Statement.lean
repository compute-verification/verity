import Definitions.Pouw.Game.Params
import Models.Pouw.Assumptions.Game
import Proofs.Flock.Soundness.Audit.Window

/-!
# PoUW's work bound at C-Flock's window audit (trusted)

`EndToEndWindow` is `Pouw.Guarantees.EndToEnd` with its ideal audit (`t` work-weighted tickets, each sampled unit
recomputed) replaced by C-Flock's window audit at the law of record, the audit
`FlockSoundness.Audit.audit_window_split_of_record` bounds. It is a statement with no proof yet; its proof is to be
`Pouw.SecurityProofs.EndToEndWindow : Pouw.Guarantees.EndToEndWindow`.

C-Flock's audit definitions are still in the proofs package, so this module imports
`Proofs.Flock.Soundness.Audit.Window`, as `Flock.Guarantees.EndToEnd`'s statement does C-Flock's; `Models.lean`, which the
`security` package builds without `security_proofs`, does not import it.
-/

namespace Pouw.Window

open FlockSoundness.Audit

/-- **The window's law of record** (#364): every call drawn by its own work law at its share of `K = 27,713`, each
stratum at least its floor `f`, with closures `cl`. It is the law `audit_window_split_of_record` bounds. -/
noncomputable def lawOfRecord {n m C : ℕ} (σ : Fin n → Fin m) (call : Fin m → Fin C) (w f : Fin m → ℕ)
    (cl : Fin n → Finset (Fin n)) : Law n :=
  (Law.stratified σ (Law.windowK σ call w f 27713) (Law.windowK_le σ call w f 27713)).closure cl

/-- **The window realizes PoUW's units** (open obligation, to be proved), at the oracle `H` and the salt `s`, for an
analysis `A` of the window's audit and the map `tx` from its committed transcripts to PoUW transcripts. At every
committed transcript `X`, the declared work of the units that are not correct in `tx X` is at most the window's unsound
work at `X`'s wrong units:
$$\sum_{u < N,\ u \notin K(tx\,X)} W_{ref}(u) \le \mathrm{unsoundWork}_w(cl, \mathrm{wrong}\,X).$$
`K` is PoUW's `correctSet`: int7 activations, checked values equal to `Y` at the noise from `H` and `s`, and the useful
output `A·B` at the registered weights.

It is to be proved from two things:
* the PoUW templates: an audit unit that is not wrong proves its PoUW unit correct, at the noise the verifier computes
  from `s` and the operand roots;
* the work table: each PoUW unit's `W_ref` is at most the work of its audit units.

It fails when a PoUW unit spans several audit units and the closures don't tie them together: one wrong tile makes the
whole PoUW unit incorrect, but puts only that tile's work in the unsound work. -/
def Realizes {Q R S : Type} (P : Protocol Q R S) (U : Workload) (H : Q → R) (s : S) {n m : ℕ}
    (σ : Fin n → Fin m) (w : Fin m → ℕ) (cl : Fin n → Finset (Fin n)) {L : Law n} {Reg : Type}
    {session : Finset (Fin n) → Reg → FlockSoundness.Game Bool} (A : Analysis L Reg session)
    (tx : A.Tr → Pouw.Transcript) : Prop :=
  ∀ X : A.Tr, ∑ u ∈ Finset.range U.layout.N \ correctSet P U H s (tx X), (P.Wref U.layout u : ℝ) ≤
    (Law.unsoundWork σ w cl (A.wrong X) : ℝ)

end Pouw.Window

namespace Pouw.Guarantees

open Finset FlockSoundness.Audit
open scoped ENNReal

/-- **PoUW's work bound at C-Flock's window audit** (statement only, no proof yet). A prover that passes C-Flock's
window audit did at least a `(1 − γ)(1 − 1/1000)` share of the window's declared work, except with probability
`2^-40 + ε_ks + δ_link + ε(q, N)`.

**What it bounds.** The prover's work `T` is the cost of its online phase, from the salt to registration, in the cost
model `CM`'s units (`CM.cost`, which the model fixes, not the prover). It is compared with the window's declared work
`W = ∑_{u < N} W_ref(u)` in the same units, `W_ref(u)` being the honest reference's cost of unit `u` without hash calls.
Let `γ = 1 − (1 − γ₀)/Ω_max`. Averaged over the oracle `H` and the salt `s`, the window audit accepts while
`T < (1 − γ)(1 − 1/1000)·W` with probability at most `2^-40 + ε_ks + δ_link + ε(q, N)`:
$$\mathbb{E}_{H,s}\,\Pr[\text{accept} \wedge T < (1-\gamma)(1-10^{-3})\,W] \le 2^{-40} + \varepsilon_{ks} +
  \delta_{link} + \varepsilon(q, N).$$
It has `EndToEnd`'s form. The audit is C-Flock's one-stage audit (`FlockSoundness.Audit.audit`) at the window's law of
record (`Window.lawOfRecord`), the prover's strategy `σ'` may depend on `H` and `s`, `ε_s = 1/1000` is the law's sizing
share, and `2^-40 + ε_ks + δ_link` takes the place of `(1 − ε_s)^t`.

**What it assumes**, each a hypothesis:
* TT(γ₀) (`Assumptions.TT`, a named assumption), with the `Ω_max` bound `W_ref ≤ Ω_max·m·k·n`, as in `EndToEnd`;
* `KnowledgeSound ε_ks` and `LinkSound δ_link` at every oracle and salt, the backend's open obligations, as
  `audit_window_split_of_record` takes them;
* the law of record's sizing, as `audit_window_split_of_record` takes it, and the work table: the window's work is at
  most `W`;
* `Window.Realizes` at every oracle and salt (an open obligation): a unit that is not correct in the PoUW transcript a
  committed transcript carries has its `W_ref` in the window's unsound work there;
* the identification: the transcript the analysis fixes at registration carries the prover's online transcript
  `A.transcript U H s`. This is the modelling line PoUW's ideal commitments draw: the online phase ends at registration,
  and the audit audits what it committed.

**What it composes**, at every oracle and salt, by `Theorem1`'s split (either the prover wins `G_γ`, or its incorrect
work exceeds `W/1000`):
* `Pinned.GammaFromTT`: `G_γ` from TT;
* `FlockSoundness.Audit.audit_window_split_of_record`: the window audit's bound at a 0.1% share of unsound work. -/
def EndToEndWindow : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) (γ₀ Ωmax : ℝ) (ε : ℕ → ℕ → ℝ),
    γ₀ < 1 → 0 < Ωmax →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N,
      (P.Wref U.layout u : ℝ) ≤ Ωmax * ((U.layout.shape u).Wmm : ℝ)) →
    Assumptions.TT CM P D γ₀ ε →
    ∀ U : Workload, U.InDomain D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 →
    ∀ A : Adversary CM, A.Bounded U T q →
    -- the window: `n` audit units in `m` strata, stratum `t` in call `call t`, at work `w t` and y weight `v t`
    ∀ (n m C : ℕ) (σ : Fin n → Fin m) (call : Fin m → Fin C) (w v f : Fin m → ℕ) (cl : Fin n → Finset (Fin n)),
    -- the law of record's sizing
    0 < Law.totalWork σ w →
    (∀ t, 0 < w t → Law.callWork σ call w (call t) = w t * (Law.stratum σ t).card) →
    0 < Law.totalWork σ v →
    (∀ t, 27713 * v t * (Law.stratum σ t).card ≤ f t * Law.totalWork σ v) →
    -- the work table
    (Law.totalWork σ w : ℝ) ≤ totalWref P U.layout →
    -- the session at each oracle and salt, its analysis, and the PoUW transcript a committed transcript carries
    ∀ (Reg : Type) (session : (Q → R) → S → Finset (Fin n) → Reg → FlockSoundness.Game Bool)
      (An : (H : Q → R) → (s : S) → Analysis (Window.lawOfRecord σ call w f cl) Reg (session H s))
      (tx : (H : Q → R) → (s : S) → (An H s).Tr → Transcript) (εks δlink : ℝ≥0∞),
    (∀ H s, (An H s).KnowledgeSound εks) → (∀ H s, (An H s).LinkSound δlink) →
    (∀ H s, Window.Realizes P U H s σ w cl (An H s) (tx H s)) →
    -- the prover's audit strategy, whose registration commits its online transcript
    ∀ σ' : (H : Q → R) → (s : S) →
      FlockSoundness.Game.Strategy (audit (Window.lawOfRecord σ call w f cl) Reg (session H s)),
    (∀ H s, tx H s ((An H s).committedOf (σ' H s)) = A.transcript U H s) →
      FlockSoundness.Game.avg (fun x : (Q → R) × S =>
        FlockSoundness.Game.prob
          (fun o => o.1 = true ∧ T < (1 - reductionGamma γ₀ Ωmax) * (1 - 1 / 1000) * totalWref P U.layout)
          (audit (Window.lawOfRecord σ call w f cl) Reg (session x.1 x.2)) (σ' x.1 x.2))
        ≤ ((2 : ℝ≥0∞) ^ 40)⁻¹ + εks + δlink + ENNReal.ofReal (ε q U.layout.N)

end Pouw.Guarantees
