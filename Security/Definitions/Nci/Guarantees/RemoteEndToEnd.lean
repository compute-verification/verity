import Definitions.Nci.Guarantees.RemotePeriod
import Definitions.Nci.Assumptions.RemoteEndToEnd

/-!
# The remote protocol end to end: the statement

The end-to-end overhead guarantee for the remote protocol (note training-20261027T0134Z-draft-e2e-overhead-proof,
"Theorem E2E (remote)"), in the cheap-success form: for every strategy of the developer, a run that is accepted, reaches
the task, and does so with less hardware than the dampener allows has probability at most `δ + η`. Its layers:

* Layer 3 is `DampenerRemotePeriod`: with probability at least `1 − δ` over the period's coins,
  `δ = p/ε' + J (1 − q)^(K+1) + ∑_t ks t`, the hardware multiple the developer needs for its own counted
  multiplications `M` is at least `ρ = max(1, 1/(s + x/MFU₀))`, with
  `x = (1 − s)(r̂ + ε_W W_cert/C + ε' + JΔ/H + J K B/C) + ν_T + χ`.
* Its two protocol legs are the bridges of PR #4653, taken here as the named premises `PouwPerWindow` and
  `SampledPerWindow`, which state those legs, so the statement holds today and each is discharged by its bridge when
  #4653 lands.
* The task model is `Coverage`: a run that reaches the task with fewer than `N` counted multiplications has
  probability at most `η`. The hardware multiple falls as the count grows, so on a run that reaches the task with at
  least `N` the dampener bounds the multiple at the task's `N`, the hardware `C + U` (FLOP at peak) over
  `T_0 = (2/κ) N/MFU₀`, the hardware an unconstrained developer at `MFU₀` needs for the task.

The cheap event is that multiple, `max(1, (C + U)/T_0)`, below `ρ`: for `ρ > 1` it is `C + U < ρ T_0`, a size below
`ρ T_0`, or, at the developer's total peak rate `Φ`, a time below `ρ T_0/Φ`; at `ρ = 1` it is empty, a hardware
multiple being at least 1 by definition. `ρ` reads the run's certified work `W_cert`, so the cheap event is one of the
run, as `ρ` is.

The acceptance event `accept` is any event of the run. The bound does not need it: the dampener counts what rejected
windows and halted identities do inside its residual and lag terms. The hash and signature breaks the protocols'
guarantees end in sit inside the bridges' premises (#4653: `HiddenWindow.CR`, `CircuitCredit`'s binding term), not
in `δ`.
-/

namespace Nci.Guarantees

open Verity Nci.Dampener

/-- **The remote protocol, end to end.** A monitoring period of `T` windows, window `t` with its own uniform coins
`Ω t`, independent of the others'; a run is `ξ`, the period's coins, and everything the developer's strategy does is a
function of it: its wrong certificates `wrong`, faulty proof blocks `bad`, their draws and detections, its certified
work `W`, its counted multiplications `M` and their sources, whether the run is accepted (`accept`) and whether it
reaches the task (`success`). Every premise of `DampenerRemotePeriod` holds, with its PoUW leg and its four
sampled-proofs legs as PR #4653's bridges conclude them (`PouwPerWindow`, `SampledPerWindow`), and the task needs
`N > 0` counted multiplications but with probability `η` (`Coverage`).

Then the run is accepted, reaches the task, and the hardware multiple at the task's `N`,
`max(1, (C + U)/T_0)` with `T_0 = (2/κ) N/MFU₀` (`hardwareMultiple`), is below
`ρ = max(1, 1/(s + ((1 − s)(r̂ + ε_W W_cert/C + ε' + JΔ/H + J K B/C) + ν_T + χ)/MFU₀))` with probability at most
`δ + η`, `δ = p/ε' + J (1 − q)^(K+1) + ∑_t ks t`: for `ρ > 1`, it spends less than `ρ T_0` of hardware, a time below
`ρ T_0/Φ` at a total peak rate `Φ`, with probability at most `δ + η`. Here `r̂ = 1 − W_cert/C`, `s = U/(C + U)`,
`ν_T = (1 − s) · 2αΛ/(κ b_a)`, and every other symbol is `DampenerRemotePeriod`'s. -/
def RemoteEndToEnd : Prop :=
  ∀ (T : ℕ) (Ω : Fin T → Type) [∀ t, Fintype (Ω t)] [∀ t, Nonempty (Ω t)]
    (c : Fin T → ℝ) (wrong : Fin T → ((t : Fin T) → Ω t) → Prop) (p ε : ℝ)
    (β ι : Type) [DecidableEq ι] (bad drawn detected : ((t : Fin T) → Ω t) → Finset β) (win : β → Fin T)
    (I : Finset ι) (idW : Fin T → ι) (τw : Fin T → ℝ) (w : β → ℝ) (ks : Fin T → ℝ) (K : ℕ)
    (Φ Δ H q B εW κ C U χ mfu : ℝ) (W : ((t : Fin T) → Ω t) → ℝ) (M Mu Mr Mx : ((t : Fin T) → Ω t) → ℕ)
    (R : Type) [Fintype R] (r d : R → Declaration) (Th : (i : R) → Threading (r i)) (Γ : ℕ) (α b_w b_a Λ : ℝ)
    (dd e : ℕ) (accept success : ((t : Fin T) → Ω t) → Prop) (N : ℕ) (η : ℝ),
    -- PoUW, per window: #4653's bridge
    0 ≤ p → 0 < ε → (∀ t, 0 ≤ c t) → ∑ t, c t ≤ C → Assumptions.PouwPerWindow Ω wrong p →
    -- sampled proofs, per window: #4653's bridge
    0 ≤ q → q ≤ 1 → Assumptions.SampledPerWindow Ω bad drawn detected win q ks →
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
    -- the task
    0 < N → Assumptions.Coverage (PMF.uniformOfFintype ((t : Fin T) → Ω t)) success M N η →
      prob (PMF.uniformOfFintype ((t : Fin T) → Ω t)) (fun ξ =>
          accept ξ ∧ success ξ ∧
            hardwareMultiple mfu κ N C U <
              remoteDampener mfu (unmonitoredShare C U)
                ((1 - unmonitoredShare C U) *
                    (residualShare (W ξ) C + εW * W ξ / C + ε + lagShare I.card Δ H + I.card * K * B / C) +
                  declaredShare κ (rhoT α b_a Λ) C U + χ)) ≤
        p / ε + I.card * (1 - q) ^ (K + 1) + ∑ t, ks t + η

end Nci.Guarantees
