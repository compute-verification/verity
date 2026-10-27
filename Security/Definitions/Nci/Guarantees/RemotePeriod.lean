import Definitions.Nci.Guarantees.Remote

/-!
# The compute dampener, remote protocol, over a monitoring period: the statement

`DampenerRemote` over a monitoring period of `T` windows (results are per monitoring period), with its
three premises on the protocols, `PouwSound`, `SampledDetection` and the thread bound's `Complies`, replaced by what
each protocol's own guarantee gives on live coins. A window is one round of the verifier's live coins: window `t`'s
coins `Ω t` are uniform and independent of the other windows', and a run of the period is `ξ : (t : Fin T) → Ω t`. Each
leg is stated as `Proofs.Flock.Soundness.Audit.Periods` states a period's bound: at any value of the other windows'
coins, over window `t`'s own (`∀ ξ, P_a[E_t(ξ[t ↦ a])] ≤ p`).

* **PoUW** (in place of `PouwSound`): window `t`'s certificate is wrong (`wrong t`: accepted with the window's time
  below `(1 − ε)·T_cred`) with probability at most `p` over its coins. That is
  `Accounting.SecurityProofs.Compose.CircuitCredit`'s conclusion, `p = δ_s + δ_bind + δ_ks = 3 · 2⁻⁴⁰`, at window `t`'s
  play, for a run whose window `t` at each value of the other windows' coins is a play meeting `CircuitCredit`'s
  premises; that bridge is not stated. `ComposePeriod.CircuitCreditPeriod` credits live words over a period's
  accepted prefix, not time, so it does not give `wrong`'s form.
* **Sampled proofs** (in place of `SampledDetection`): a window's faulty blocks are fixed before its coins (its wrong
  units at registration, `wrongRegZ`), and its draw and verdict read its coins and the past. Two per-window bounds,
  each of which a window of `Flock.Guarantees.IntegrityPeriod` has at the strategy that plays it: the draw misses all
  `m` of the window's faulty blocks with probability at most `(1 − q)^m` (`Flock.Guarantees.HiddenWindow.drawMiss_le`,
  under A3: a uniform `kd`-subset of `n` units misses `m` fixed units with probability
  `C(n − m, kd)/C(n, kd) ≤ (1 − kd/n)^m`), and the window is accepted with a drawn faulty block with probability at most
  `ε_ks,t + δ_link,t` (`HiddenWindow.accepted_drawnWrong_le`, from `Flock.SecurityProofs.EndToEndHiddenDrawn`). That
  the deployment's window `t`, at each value of the other windows' coins, is such a window with its faulty blocks as
  its wrong units is a bridge not stated either: `IntegrityPeriod` plays the windows as one sequential game, not over a
  product of window coins. No guarantee implies `SampledDetection`'s per-block form: it conditions each block on its
  own window's other blocks, and it folds the knowledge term into `q`, where the protocol's is additive per window. The
  statement does not need it: a block's time is its window's, and each identity's escaped blocks are bounded at the
  predictable time its faulty blocks reach `K + 1`, the stopped, per-identity form of `IntegrityHorizon`'s own
  argument, which `FlockSoundness.Audit.Periods.prCoin_forall_sum_le_pow` bounds by `(1 − q)^(K+1)`. That is a tail
  bound, so the escape term needs no Markov's inequality: `J (1 − q)^(K+1)` at an escape share `J K B/C`, against
  `DampenerRemote`'s `J ((1 − q)/q) B/(θC)`.
* **The halt**: `TerminalDetection` stays a named premise. The halt is code (`Fleet`'s R5-all in
  `verity/core/protocols/sampled_proofs/service.py`, its verifier of record `SampledProofs.Fleet`), and
  `Properties.SampledProofs` proves what it does (`FleetHaltIsFinal`, `NoAcceptAfterHalt`), but not this form: it halts
  an operator at its `budget`-th failure, not its first (`budget ≥ 1`); it ends the operator's registrations, not its
  hardware's use; and nothing states the verdict's time `Δ` after the detected block.
* **NCI** (in place of `Complies` and `ThreadedThrottled`'s premises): each declared run `i` meets
  `NciCheck.Properties.Throttled`'s conclusion, `b_a |M_i| ≤ α Λ_i size(d_i)`, which
  `NciCheck.Properties.DescThrottles` gives for every declaration a verdict of compliance of the check of record lays
  out (`NciCheck.Baseline.evaluate`, whose verdict is `Desc.evaluate`'s), at `Γ = gamma` and `α = num/den`, under
  `BaselineRegistered` and `WeightsIncompressible`, which the check does not see. Reading the check itself needs the
  code (package `security_properties`), so the statement takes its verdict's conclusion. `DescThrottles` is not a
  guarantee yet: it rests on `Novel.ReadsExact`, through `DescAgrees`. The cover theorem's factor on the same verdicts,
  `NciCheck.Properties.CheckedCoverTraining`, would replace `ρ_T = b_a/(αΛ)` by its own.

Every other premise is `DampenerRemote`'s: the model's accounting of the four sources of counted multiplications, the
capacities `C` and `U` (`s = U/(C + U)`), and the hardware's rate `Φ` per identity. The developer's identities `I`
are fixed over the period: nothing here models or rests on re-registration.

`Nci.SecurityProofs.DampenerRemotePeriod` proves it. `DampenerRemotePeriod_honest` and `DampenerRemotePeriod_nontrivial`
are its witnesses: its premises hold at a constructed honest developer, and at `AcceptAll`, a PoUW verifier that accepts
every certificate, every other premise holds while PoUW's per-window bound and the conclusion fail.
-/

namespace Nci.Guarantees

open Verity Nci.Dampener

/-- **The compute dampener, remote protocol, over a monitoring period.** A monitoring period of `T` windows, window `t`
with its own uniform coins `Ω t`, independent of the others' (a window is a round of live coins: the deployment's
window, or a finer cut of it where its PoUW and sampled-proofs coins are drawn at different times). Every bound on a
window holds at any value `ξ` of the other windows' coins, over its own (`ξ[t ↦ a]`, `a` uniform on `Ω t`).

* PoUW certifies window `t` at capacity `c_t ≥ 0`, `∑ c_t ≤ C`, its certificate wrong (`wrong t`) with probability at
  most `p ≥ 0`: `CircuitCredit`'s conclusion at window `t`'s play.
* Sampled proofs: the developer's faulty proof blocks `bad ξ` (block `b` in window `win b`, skipping `0 ≤ w_b ≤ B` of
  declared work) are fixed before their window's coins; the units the window draws (`drawn`) and the faulty blocks it
  detects (`detected`, those of a window flock-verify rejects) read only its coins and the earlier windows'. The draw
  misses the window's `m` faulty blocks with probability at most `(1 − q)^m`, `0 ≤ q ≤ 1` (A3, `q ≤ kd/n`), and the
  window detects none of them while it draws one with probability at most `ks t` (`EndToEndHiddenDrawn`,
  `ks t = ε_ks + δ_link`).
* Window `t` is run by identity `idW t`, one of the `J = |I|` the developer controls, at time `τw t`; a block runs at
  its window's time, and each identity's windows run in their order. Detection is terminal at a lag `Δ ≥ 0`
  (`TerminalDetection`), and one identity's faulty blocks within any span of `Δ` skip at most `Φ Δ`, `Φ H ≤ C` over the
  period's length `H > 0`, as in `VerdictLag`.
* NCI: each declared run `i` (declaration `d_i`, containing a reference circuit `r_i` threaded by `Th_i`) meets
  `Throttled`'s conclusion, `b_a |M_i| ≤ α Λ_i size(d_i)` at `Λ_i = Th_i.bound (αΓ) b_w dd e ≤ Λ`: what
  `DescThrottles` gives for a declaration the check of record accepts. The runs fit on the monitored compute,
  `∑ size(d_i) ≤ C`.
* The counted multiplications `M` are as in `DampenerRemote`, with the declared runs' `∑ |M_i|`: unmonitored compute at
  `MFU₀ ∈ (0, 1]` (`M_U ≤ MFU₀ κ U/2`), the residual outside the work certified correctly and run
  (`M_r ≤ κ (C − (1 − ε_W) W_cert + wrongCapacity + skipped)/2`, `W_cert` the period's certified work, `ε_W` PoUW's
  deficit), and faulty proof blocks within the cheating share `χ`.

Then, for any `ε' > 0` and `K`, with probability at least `1 − δ` over the period's coins,
`δ = p/ε' + J (1 − q)^(K+1) + ∑_t ks t`, the compute efficiency is at most
`s + (1 − s)(r̂ + ε_W W_cert/C + ε' + JΔ/H + J K B/C) + ν_T + χ`, and the hardware multiple the developer needs is at
least `ρ = max(1, 1/(s + ((1 − s)(r̂ + ε_W W_cert/C + ε' + JΔ/H + J K B/C) + ν_T + χ)/MFU₀))`, with
`r̂ = 1 − W_cert/C`, `s = U/(C + U)` and `ν_T = (1 − s) · 2αΛ/(κ b_a)`. At a pooled `δ` (1% and 10⁻⁶),
`ε' = 2p/δ` and `K ≥ ln(2J/δ)/q` give `δ + ∑_t ks t`; the per-window figure is the statement at `T = 1`.
With every proof block proved (`q = 1`), `K = 0` and the escape share is 0. -/
def DampenerRemotePeriod : Prop :=
  ∀ (T : ℕ) (Ω : Fin T → Type) [∀ t, Fintype (Ω t)] [∀ t, Nonempty (Ω t)]
    (c : Fin T → ℝ) (wrong : Fin T → ((t : Fin T) → Ω t) → Prop) (p ε : ℝ)
    (β ι : Type) [DecidableEq ι] (bad drawn detected : ((t : Fin T) → Ω t) → Finset β) (win : β → Fin T)
    (I : Finset ι) (idW : Fin T → ι) (τw : Fin T → ℝ) (w : β → ℝ) (ks : Fin T → ℝ) (K : ℕ)
    (Φ Δ H q B εW κ C U χ mfu : ℝ) (W : ((t : Fin T) → Ω t) → ℝ) (M Mu Mr Mx : ((t : Fin T) → Ω t) → ℕ)
    (R : Type) [Fintype R] (r d : R → Declaration) (Th : (i : R) → Threading (r i)) (Γ : ℕ) (α b_w b_a Λ : ℝ)
    (dd e : ℕ),
    -- PoUW, per window
    0 ≤ p → 0 < ε → (∀ t, 0 ≤ c t) → ∑ t, c t ≤ C →
    (∀ t (ξ : (t : Fin T) → Ω t),
      prob (PMF.uniformOfFintype (Ω t)) (fun a => wrong t (Function.update ξ t a)) ≤ p) →
    -- sampled proofs, per window: faulty blocks fixed before the window's coins, its draw and verdict read them
    (∀ t (ξ ξ' : (t : Fin T) → Ω t), (∀ s, s < t → ξ s = ξ' s) → ∀ b, win b = t → (b ∈ bad ξ ↔ b ∈ bad ξ')) →
    (∀ t (ξ ξ' : (t : Fin T) → Ω t), (∀ s, s ≤ t → ξ s = ξ' s) → ∀ b, win b = t →
      (b ∈ drawn ξ ↔ b ∈ drawn ξ') ∧ (b ∈ detected ξ ↔ b ∈ detected ξ')) →
    0 ≤ q → q ≤ 1 →
    (∀ t (ξ : (t : Fin T) → Ω t),
      prob (PMF.uniformOfFintype (Ω t)) (fun a => ∀ b ∈ bad ξ, win b = t → b ∉ drawn (Function.update ξ t a)) ≤
        (1 - q) ^ ((bad ξ).filter fun b => win b = t).card) →
    (∀ t (ξ : (t : Fin T) → Ω t),
      prob (PMF.uniformOfFintype (Ω t)) (fun a => (∃ b ∈ bad ξ, win b = t ∧ b ∈ drawn (Function.update ξ t a)) ∧
          ∀ b ∈ bad ξ, win b = t → b ∉ detected (Function.update ξ t a)) ≤ ks t) →
    -- the identities and the halt
    (∀ t t', idW t = idW t' → t < t' → τw t < τw t') → (∀ ξ, ∀ b ∈ bad ξ, idW (win b) ∈ I) →
    0 < H → 0 ≤ Δ → Φ * H ≤ C →
    (∀ ξ, Assumptions.TerminalDetection (bad ξ) (detected ξ) (fun b => idW (win b)) (fun b => τw (win b)) Δ) →
    (∀ ξ j s, ∑ b ∈ bad ξ with idW (win b) = j ∧ s ≤ τw (win b) ∧ τw (win b) ≤ s + Δ, w b ≤ Φ * Δ) →
    (∀ b, 0 ≤ w b) → 0 ≤ B → (∀ ξ, ∀ b ∈ bad ξ, w b ≤ B) →
    -- NCI: each declared run as a verdict of compliance of the check of record carries it
    0 < κ → 0 < C → 0 ≤ U → 0 < mfu → mfu ≤ 1 → 0 ≤ α → 0 < b_a → 0 ≤ Λ →
    (∀ i, b_a * (Th i).M.card ≤ α * (Th i).bound (α * Γ) b_w dd e * (d i).D.size (d i).D.computed) →
    (∀ i, (Th i).bound (α * Γ) b_w dd e ≤ Λ) → ∑ i, ((d i).D.size (d i).D.computed : ℝ) ≤ C →
    -- the counted multiplications
    (∀ ξ, 0 < M ξ) → (∀ ξ, M ξ ≤ Mu ξ + Mr ξ + ∑ i, (Th i).M.card + Mx ξ) →
    (∀ ξ, (Mu ξ : ℝ) ≤ mfu * κ * U / 2) →
    (∀ ξ, (Mr ξ : ℝ) ≤ κ * (C - (1 - εW) * W ξ + wrongCapacity c (fun t => wrong t ξ) + ∑ b ∈ bad ξ, w b) / 2) →
    (∀ ξ, (Mx ξ : ℝ) ≤ κ * χ * (C + U) / 2) →
      1 - (p / ε + I.card * (1 - q) ^ (K + 1) + ∑ t, ks t) ≤
        prob (PMF.uniformOfFintype ((t : Fin T) → Ω t)) (fun ξ =>
          efficiency κ (M ξ) C U ≤
              unmonitoredShare C U +
                  (1 - unmonitoredShare C U) *
                    (residualShare (W ξ) C + εW * W ξ / C + ε + lagShare I.card Δ H + I.card * K * B / C) +
                declaredShare κ (rhoT α b_a Λ) C U + χ ∧
            remoteDampener mfu (unmonitoredShare C U)
                ((1 - unmonitoredShare C U) *
                    (residualShare (W ξ) C + εW * W ξ / C + ε + lagShare I.card Δ H + I.card * K * B / C) +
                  declaredShare κ (rhoT α b_a Λ) C U + χ) ≤
              hardwareMultiple mfu κ (M ξ) C U)

/-- **`DampenerRemotePeriod`'s premises hold where its conclusion says something**: an instance meets every premise with
the failure probability `δ = p/ε' + J (1 − q)^(K+1) + ∑_t ks t` at most `10⁻⁶`, and on every run the overhead bound on
the hardware multiple (`remoteDampener`'s value) exceeds 1 while the run's own hardware multiple meets it. The proof's
instance is an honest developer over one window, run by its one identity: its PoUW certificate right, no faulty proof
block, no declared run, its monitored compute all certified work and its counted multiplications on its unmonitored
compute at `MFU₀ = 1/2`, at the remote reference's `p = 3 · 2⁻⁴⁰` (`CircuitCredit`'s `δ_s + δ_bind + δ_ks`) and
`ε' = 2p/δ` at the pooled `δ = 10⁻⁶`, with every proof block proved (`q = 1`, `K = 0`). -/
def DampenerRemotePeriod_honest : Prop :=
  ∃ (T : ℕ) (Ω : Fin T → Type) (_ : ∀ t, Fintype (Ω t)) (_ : ∀ t, Nonempty (Ω t))
    (c : Fin T → ℝ) (wrong : Fin T → ((t : Fin T) → Ω t) → Prop) (p ε : ℝ)
    (β ι : Type) (_ : DecidableEq ι) (bad drawn detected : ((t : Fin T) → Ω t) → Finset β) (win : β → Fin T)
    (I : Finset ι) (idW : Fin T → ι) (τw : Fin T → ℝ) (w : β → ℝ) (ks : Fin T → ℝ) (K : ℕ)
    (Φ Δ H q B εW κ C U χ mfu : ℝ) (W : ((t : Fin T) → Ω t) → ℝ) (M Mu Mr Mx : ((t : Fin T) → Ω t) → ℕ)
    (R : Type) (_ : Fintype R) (r d : R → Declaration) (Th : (i : R) → Threading (r i)) (Γ : ℕ) (α b_w b_a Λ : ℝ)
    (dd e : ℕ),
    -- PoUW, per window
    0 ≤ p ∧ 0 < ε ∧ (∀ t, 0 ≤ c t) ∧ ∑ t, c t ≤ C ∧
    (∀ t (ξ : (t : Fin T) → Ω t),
      prob (PMF.uniformOfFintype (Ω t)) (fun a => wrong t (Function.update ξ t a)) ≤ p) ∧
    -- sampled proofs, per window: faulty blocks fixed before the window's coins, its draw and verdict read them
    (∀ t (ξ ξ' : (t : Fin T) → Ω t), (∀ s, s < t → ξ s = ξ' s) → ∀ b, win b = t → (b ∈ bad ξ ↔ b ∈ bad ξ')) ∧
    (∀ t (ξ ξ' : (t : Fin T) → Ω t), (∀ s, s ≤ t → ξ s = ξ' s) → ∀ b, win b = t →
      (b ∈ drawn ξ ↔ b ∈ drawn ξ') ∧ (b ∈ detected ξ ↔ b ∈ detected ξ')) ∧
    0 ≤ q ∧ q ≤ 1 ∧
    (∀ t (ξ : (t : Fin T) → Ω t),
      prob (PMF.uniformOfFintype (Ω t)) (fun a => ∀ b ∈ bad ξ, win b = t → b ∉ drawn (Function.update ξ t a)) ≤
        (1 - q) ^ ((bad ξ).filter fun b => win b = t).card) ∧
    (∀ t (ξ : (t : Fin T) → Ω t),
      prob (PMF.uniformOfFintype (Ω t)) (fun a => (∃ b ∈ bad ξ, win b = t ∧ b ∈ drawn (Function.update ξ t a)) ∧
          ∀ b ∈ bad ξ, win b = t → b ∉ detected (Function.update ξ t a)) ≤ ks t) ∧
    -- the identities and the halt
    (∀ t t', idW t = idW t' → t < t' → τw t < τw t') ∧ (∀ ξ, ∀ b ∈ bad ξ, idW (win b) ∈ I) ∧
    0 < H ∧ 0 ≤ Δ ∧ Φ * H ≤ C ∧
    (∀ ξ, Assumptions.TerminalDetection (bad ξ) (detected ξ) (fun b => idW (win b)) (fun b => τw (win b)) Δ) ∧
    (∀ ξ j s, ∑ b ∈ bad ξ with idW (win b) = j ∧ s ≤ τw (win b) ∧ τw (win b) ≤ s + Δ, w b ≤ Φ * Δ) ∧
    (∀ b, 0 ≤ w b) ∧ 0 ≤ B ∧ (∀ ξ, ∀ b ∈ bad ξ, w b ≤ B) ∧
    -- NCI: each declared run as a verdict of compliance of the check of record carries it
    0 < κ ∧ 0 < C ∧ 0 ≤ U ∧ 0 < mfu ∧ mfu ≤ 1 ∧ 0 ≤ α ∧ 0 < b_a ∧ 0 ≤ Λ ∧
    (∀ i, b_a * (Th i).M.card ≤ α * (Th i).bound (α * Γ) b_w dd e * (d i).D.size (d i).D.computed) ∧
    (∀ i, (Th i).bound (α * Γ) b_w dd e ≤ Λ) ∧ ∑ i, ((d i).D.size (d i).D.computed : ℝ) ≤ C ∧
    -- the counted multiplications
    (∀ ξ, 0 < M ξ) ∧ (∀ ξ, M ξ ≤ Mu ξ + Mr ξ + ∑ i, (Th i).M.card + Mx ξ) ∧
    (∀ ξ, (Mu ξ : ℝ) ≤ mfu * κ * U / 2) ∧
    (∀ ξ, (Mr ξ : ℝ) ≤ κ * (C - (1 - εW) * W ξ + wrongCapacity c (fun t => wrong t ξ) + ∑ b ∈ bad ξ, w b) / 2) ∧
    (∀ ξ, (Mx ξ : ℝ) ≤ κ * χ * (C + U) / 2) ∧
    p / ε + I.card * (1 - q) ^ (K + 1) + ∑ t, ks t ≤ 1 / 10 ^ 6 ∧
    ∀ ξ, 1 < remoteDampener mfu (unmonitoredShare C U)
        ((1 - unmonitoredShare C U) *
            (residualShare (W ξ) C + εW * W ξ / C + ε + lagShare I.card Δ H + I.card * K * B / C) +
          declaredShare κ (rhoT α b_a Λ) C U + χ) ∧
      remoteDampener mfu (unmonitoredShare C U)
          ((1 - unmonitoredShare C U) *
              (residualShare (W ξ) C + εW * W ξ / C + ε + lagShare I.card Δ H + I.card * K * B / C) +
            declaredShare κ (rhoT α b_a Λ) C U + χ) ≤
        hardwareMultiple mfu κ (M ξ) C U

noncomputable section

/-! **`AcceptAll`, a PoUW verifier that accepts every certificate**: one window, its coins a bit, run by the one
identity at time 0, whose certificate, for all of the monitored compute, is wrong and accepted on every run. It has no
faulty proof block and no declared run, all of the monitored compute is certified work, and the run's one counted
multiplication is in the residual. `DampenerRemotePeriod_nontrivial` is stated at it, its numbers written in place. -/
namespace AcceptAll

/-- The window certifies all of the monitored compute, `C = 1`. -/
abbrev c : Fin 1 → ℝ := fun _ => 1

/-- The window's certificate is wrong on every run. -/
abbrev wrong : Fin 1 → (Fin 1 → Bool) → Prop := fun _ _ => True

/-- No faulty proof block: none drawn, none detected, and the knowledge-soundness bound 0. -/
abbrev bad : (Fin 1 → Bool) → Finset Empty := fun _ => ∅
abbrev drawn : (Fin 1 → Bool) → Finset Empty := fun _ => ∅
abbrev detected : (Fin 1 → Bool) → Finset Empty := fun _ => ∅
abbrev win : Empty → Fin 1 := fun b => b.elim
abbrev w : Empty → ℝ := fun _ => 0
abbrev ks : Fin 1 → ℝ := fun _ => 0

/-- The one identity runs the window at time 0. -/
abbrev idW : Fin 1 → Unit := fun _ => ()
abbrev τw : Fin 1 → ℝ := fun _ => 0

/-- All of the monitored compute is certified work, `W_cert = 1`, and the run's one counted multiplication is in the
residual. -/
abbrev W : (Fin 1 → Bool) → ℝ := fun _ => 1
abbrev M : (Fin 1 → Bool) → ℕ := fun _ => 1
abbrev Mu : (Fin 1 → Bool) → ℕ := fun _ => 0
abbrev Mr : (Fin 1 → Bool) → ℕ := fun _ => 1
abbrev Mx : (Fin 1 → Bool) → ℕ := fun _ => 0

/-- No declared run. -/
abbrev r : Empty → Declaration := fun i => i.elim
abbrev d : Empty → Declaration := fun i => i.elim
abbrev Th : (i : Empty) → Threading (r i) := fun i => i.elim

end AcceptAll

end

/-- **`DampenerRemotePeriod` is false without PoUW's per-window bound**: at `AcceptAll`, with `p = 0`, `ε' = 1/2`,
`C = 1`, `U = 0`, `κ = 2`, `MFU₀ = 1`, `q = 1` and `K = 0`, every other premise holds, PoUW's per-window bound fails
(the certificate is wrong with probability 1, above `p`), and the conclusion fails: at `δ = 0`, the run's compute
efficiency is 1, above the bound `51/100`, and its hardware multiple is 1, below the overhead bound `100/51`, on every
run. -/
def DampenerRemotePeriod_nontrivial : Prop :=
  -- every other premise: PoUW's capacities
  (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 / 2 ∧ (∀ t, 0 ≤ AcceptAll.c t) ∧ ∑ t, AcceptAll.c t ≤ 1 ∧
  -- sampled proofs, per window
  (∀ (t : Fin 1) (ξ ξ' : Fin 1 → Bool), (∀ s, s < t → ξ s = ξ' s) → ∀ b, AcceptAll.win b = t →
    (b ∈ AcceptAll.bad ξ ↔ b ∈ AcceptAll.bad ξ')) ∧
  (∀ (t : Fin 1) (ξ ξ' : Fin 1 → Bool), (∀ s, s ≤ t → ξ s = ξ' s) → ∀ b, AcceptAll.win b = t →
    (b ∈ AcceptAll.drawn ξ ↔ b ∈ AcceptAll.drawn ξ') ∧ (b ∈ AcceptAll.detected ξ ↔ b ∈ AcceptAll.detected ξ')) ∧
  (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 1 ∧
  (∀ (t : Fin 1) (ξ : Fin 1 → Bool), prob (PMF.uniformOfFintype Bool)
      (fun a => ∀ b ∈ AcceptAll.bad ξ, AcceptAll.win b = t → b ∉ AcceptAll.drawn (Function.update ξ t a)) ≤
    (1 - 1) ^ ((AcceptAll.bad ξ).filter fun b => AcceptAll.win b = t).card) ∧
  (∀ (t : Fin 1) (ξ : Fin 1 → Bool), prob (PMF.uniformOfFintype Bool)
      (fun a => (∃ b ∈ AcceptAll.bad ξ, AcceptAll.win b = t ∧ b ∈ AcceptAll.drawn (Function.update ξ t a)) ∧
        ∀ b ∈ AcceptAll.bad ξ, AcceptAll.win b = t → b ∉ AcceptAll.detected (Function.update ξ t a)) ≤
    AcceptAll.ks t) ∧
  -- the identities and the halt
  (∀ t t', AcceptAll.idW t = AcceptAll.idW t' → t < t' → AcceptAll.τw t < AcceptAll.τw t') ∧
  (∀ ξ, ∀ b ∈ AcceptAll.bad ξ, AcceptAll.idW (AcceptAll.win b) ∈ (∅ : Finset Unit)) ∧
  (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) * 1 ≤ 1 ∧
  (∀ ξ, Assumptions.TerminalDetection (AcceptAll.bad ξ) (AcceptAll.detected ξ)
    (fun b => AcceptAll.idW (AcceptAll.win b)) (fun b => AcceptAll.τw (AcceptAll.win b)) 0) ∧
  (∀ ξ j s, ∑ b ∈ AcceptAll.bad ξ with AcceptAll.idW (AcceptAll.win b) = j ∧ s ≤ AcceptAll.τw (AcceptAll.win b) ∧
    AcceptAll.τw (AcceptAll.win b) ≤ s + 0, AcceptAll.w b ≤ 0 * 0) ∧
  (∀ b, 0 ≤ AcceptAll.w b) ∧ (0 : ℝ) ≤ 0 ∧ (∀ ξ, ∀ b ∈ AcceptAll.bad ξ, AcceptAll.w b ≤ 0) ∧
  -- NCI
  (0 : ℝ) < 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 100 ∧ (0 : ℝ) ≤ 1 ∧
  (∀ i, 100 * (AcceptAll.Th i).M.card ≤
    1 * (AcceptAll.Th i).bound (1 * (0 : ℕ)) 0 0 0 * (AcceptAll.d i).D.size (AcceptAll.d i).D.computed) ∧
  (∀ i, (AcceptAll.Th i).bound (1 * (0 : ℕ)) 0 0 0 ≤ 1) ∧
  ∑ i, ((AcceptAll.d i).D.size (AcceptAll.d i).D.computed : ℝ) ≤ 1 ∧
  -- the counted multiplications
  (∀ ξ, 0 < AcceptAll.M ξ) ∧
  (∀ ξ, AcceptAll.M ξ ≤ AcceptAll.Mu ξ + AcceptAll.Mr ξ + ∑ i, (AcceptAll.Th i).M.card + AcceptAll.Mx ξ) ∧
  (∀ ξ, (AcceptAll.Mu ξ : ℝ) ≤ 1 * 2 * 0 / 2) ∧
  (∀ ξ, (AcceptAll.Mr ξ : ℝ) ≤ 2 * (1 - (1 - 0) * AcceptAll.W ξ +
    wrongCapacity AcceptAll.c (fun t => AcceptAll.wrong t ξ) + ∑ b ∈ AcceptAll.bad ξ, AcceptAll.w b) / 2) ∧
  (∀ ξ, (AcceptAll.Mx ξ : ℝ) ≤ 2 * 0 * (1 + 0) / 2) ∧
  -- PoUW's per-window bound fails
  ¬ (∀ (t : Fin 1) (ξ : Fin 1 → Bool),
      prob (PMF.uniformOfFintype Bool) (fun a => AcceptAll.wrong t (Function.update ξ t a)) ≤ 0) ∧
  -- and so does the conclusion
  ¬ (1 - (0 / (1 / 2) + (∅ : Finset Unit).card * (1 - 1) ^ ((0 : ℕ) + 1) + ∑ t, AcceptAll.ks t) ≤
      prob (PMF.uniformOfFintype (Fin 1 → Bool)) (fun ξ =>
        efficiency 2 (AcceptAll.M ξ) 1 0 ≤
            unmonitoredShare 1 0 +
                (1 - unmonitoredShare 1 0) *
                  (residualShare (AcceptAll.W ξ) 1 + 0 * AcceptAll.W ξ / 1 + 1 / 2 +
                    lagShare (∅ : Finset Unit).card 0 1 + (∅ : Finset Unit).card * (0 : ℕ) * 0 / 1) +
              declaredShare 2 (rhoT 1 100 1) 1 0 + 0 ∧
          remoteDampener 1 (unmonitoredShare 1 0)
              ((1 - unmonitoredShare 1 0) *
                  (residualShare (AcceptAll.W ξ) 1 + 0 * AcceptAll.W ξ / 1 + 1 / 2 +
                    lagShare (∅ : Finset Unit).card 0 1 + (∅ : Finset Unit).card * (0 : ℕ) * 0 / 1) +
                declaredShare 2 (rhoT 1 100 1) 1 0 + 0) ≤
            hardwareMultiple 1 2 (AcceptAll.M ξ) 1 0))

end Nci.Guarantees
