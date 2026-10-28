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
  `ThetaRegistered` and `WeightsIncompressible`, which the check does not see. Reading the check itself needs the
  code (package `security_properties`), so the statement takes its verdict's conclusion. `DescThrottles` is not a
  guarantee yet: it rests on `Novel.ReadsExact`, through `DescAgrees`. The cover theorem's factor on the same verdicts,
  `NciCheck.Properties.CheckedCoverTraining`, would replace `ρ_T = b_a/(αΛ)` by its own.

Every other premise is `DampenerRemote`'s: the model's accounting of the four sources of counted multiplications, the
capacities `C` and `U` (`s = U/(C + U)`), and the hardware's rate `Φ` per identity. The developer's identities `I`
are fixed over the period: nothing here models or rests on re-registration.

The statement is not proved: `DampenerRemotePeriodStated` holds its text for the lock.
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
    (R : Type) [Fintype R] (r d : R → RegisteredCircuit) (Th : (i : R) → Threading (r i)) (Γ : ℕ) (α b_w b_a Λ : ℝ)
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

/-- **The remote dampener over a monitoring period is stated**: `DampenerRemotePeriod`, equal to itself, so the lock
records its text and every definition it reads until it is proved. -/
def DampenerRemotePeriodStated : Prop :=
  let s := DampenerRemotePeriod
  s = s

end Nci.Guarantees
