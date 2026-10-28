import Definitions.Nci.Guarantees.Dampener
import Definitions.Nci.Assumptions.Remote

/-!
# The compute dampener, remote protocol: the guarantees

The remote dampener `1/ρ = s + (r̂ + ε' + JΔ/H + ν)/MFU₀` (`benchmarks/nci/tiers.py` at T1 and T2: `residual_share`,
`fit.residual_with_lag` and `fit.dampener`), with its two terms that `DampenerThreaded` takes inside its residual's
hypothesis derived, and the two that sampled proofs and PoUW's deficit add:

* `CertificateSlack` ([CERTIFICATE-SLACK]): over `N` windows, each certificate wrong with probability at most `p`
  (`PouwSound`), the windows whose certificates are wrong hold more than `ε' C` of the capacity with probability at most
  `p/ε'`, by Markov's inequality, which needs no independence between windows;
* `VerdictLag` ([VERDICT-LAG]): under `TerminalDetection`, the declared work skipped between each identity's first
  detected faulty proof block and its halt `Δ` later is at most `(JΔ/H) C`, so all faulty blocks skip at most that plus
  what the blocks accepted before their identity's first detection (`escaped`) skip;
* `DampenerRemote` composes them with `ThreadedThrottled`'s declared computation, charging the escaped blocks by
  Markov's inequality on their expected work, at most `((1 − q)/q) B` per identity when each faulty block is caught with
  probability at least `q` whatever came before (`SampledDetection`), and PoUW's deficit `ε_W` through the certified
  work: with probability at least `1 − p/ε' − J ((1 − q)/q) B/(θ C)`,
  `η ≤ s + (1 − s)(r̂ + ε_W W_cert/C + ε' + JΔ/H + θ) + ν_T + χ`, and the hardware multiple the developer needs is at
  least `ρ = max(1, 1/(s + ((1 − s)(r̂ + ε_W W_cert/C + ε' + JΔ/H + θ) + ν_T + χ)/MFU₀))`. The factor
  `1 − s = C/(C + U)` is tiers' (`evaluate` counts the residual over `C + U`); it only raises `ρ` over the form above,
  as `0 ≤ s ≤ 1`. With every proof block proved (T1–T4), `q = 1` and the escape term's probability is 0 at any `θ > 0`.
-/

namespace Nci.Guarantees

open Verity Nci.Dampener

/-- **The certificate slack** ([CERTIFICATE-SLACK]). PoUW certifies each of `N` windows, window `i` of capacity
`c_i ≥ 0`, `∑ c_i ≤ C`, and each window's certificate is wrong with probability at most `p ≥ 0` under the outcomes'
distribution `μ` (`PouwSound`). Then the windows whose certificates are wrong hold more than `ε' C` of the capacity with
probability at most `p/ε'`: their expected capacity is at most `p C`, and Markov's inequality, which reads each window's
marginal and nothing of how windows depend on one another. At `p = 3 · 2⁻⁴⁰` and `ε' = 10⁻⁴`, `δ = p/ε' ≈ 2.7 · 10⁻⁸`.
Outside that event the work certified correctly is at least `W_cert − ε' C`. -/
def CertificateSlack : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (μ : PMF Ω) (N : ℕ) (c : Fin N → ℝ) (wrong : Ω → Fin N → Prop) (p ε C : ℝ),
    0 ≤ p → 0 < ε → 0 < C → (∀ i, 0 ≤ c i) → ∑ i, c i ≤ C → Assumptions.PouwSound μ wrong p →
      prob μ (fun ω => ε * C < wrongCapacity c (wrong ω)) ≤ p / ε

/-- **The verdict lag** ([VERDICT-LAG]). Of a developer's faulty proof blocks `bad`, block `b` is run by identity
`ident b`, one of the `J = |I|` it controls, at time `τ b`, and skips `w_b` of declared work; those `detected` fail
their proof. Detection is terminal at a lag `Δ ≥ 0` (`TerminalDetection`), and one identity's blocks within any span
of `Δ` skip at most `Φ Δ`, its hardware's capacity at `Φ` a unit of time, with `Φ H ≤ C` over the horizon `H > 0`.
Then the blocks at or after their identity's first detection skip at most `J Φ Δ`, a share `JΔ/H` of `C`, and the rest
are the blocks accepted before it (`escaped`), none when every faulty block is detected. -/
def VerdictLag : Prop :=
  ∀ (β ι : Type) [DecidableEq ι] (bad detected : Finset β) (I : Finset ι) (ident : β → ι) (τ w : β → ℝ)
    (Φ Δ H C : ℝ),
    0 < H → 0 ≤ Δ → Φ * H ≤ C → (∀ b ∈ bad, ident b ∈ I) → Assumptions.TerminalDetection bad detected ident τ Δ →
    (∀ j t, ∑ b ∈ bad with ident b = j ∧ t ≤ τ b ∧ τ b ≤ t + Δ, w b ≤ Φ * Δ) →
      ∑ b ∈ bad, w b ≤ lagShare I.card Δ H * C + ∑ b ∈ escaped bad detected ident τ, w b

/-- **The compute dampener, remote protocol.** Outcomes `ω` are drawn from `μ` on a finite space. PoUW certifies `N`
windows of capacities `c_i ≥ 0`, `∑ c_i ≤ C`, each certificate wrong (`wrong ω i`) with probability at most `p ≥ 0`
(`PouwSound`). The developer's faulty proof blocks `bad ω`, run by `J = |I|` identities, those `detected ω` failing
their proof, under `TerminalDetection` at a lag `Δ`, skip `w_b ≥ 0` of declared work each, at most `Φ Δ` per identity
within any span of `Δ`, `Φ H ≤ C`, as in `VerdictLag`, and at most `B` each; each faulty block with no earlier detection
of its identity is detected with probability at least `q ∈ (0, 1]` whatever came before (`SampledDetection`), so those
accepted before their identity's first detection skip at most `((1 − q)/q) B` per identity in expectation. Its counted
multiplications `M` are as in `DampenerThreaded`, with no memory cap: unmonitored compute trains like the unconstrained
developer, at model-FLOP utilization `MFU₀ ∈ (0, 1]` (`M_U ≤ MFU₀ κ U/2`); the residual trains at full efficiency on the
monitored compute outside the work certified correctly and run, at least `(1 − ε_W) W_cert` less the wrong windows'
capacity and the skipped work, `ε_W` PoUW's deficit (`M_r ≤ κ (C − (1 − ε_W) W_cert + wrongCapacity + skipped)/2`); the
declared computation is bounded by `ThreadedThrottled`; faulty proof blocks inject at most the cheating share `χ`, 0
with no sampling. Then, at an escape slack `θ > 0`, with probability at least `1 − δ`,
`δ = p/ε' + J ((1 − q)/q) B/(θ C)`, the compute efficiency is at most
`s + (1 − s)(r̂ + ε_W W_cert/C + ε' + JΔ/H + θ) + ν_T + χ`, and the hardware multiple the developer needs is at least
`ρ = max(1, 1/(s + ((1 − s)(r̂ + ε_W W_cert/C + ε' + JΔ/H + θ) + ν_T + χ)/MFU₀))`, with `r̂ = 1 − W_cert/C`. -/
def DampenerRemote : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (μ : PMF Ω) (N : ℕ) (c : Fin N → ℝ) (wrong : Ω → Fin N → Prop) (p ε : ℝ)
    (β ι : Type) [DecidableEq ι] (bad detected : Ω → Finset β) (I : Finset ι) (ident : β → ι) (τ w : β → ℝ)
    (Φ Δ H q B θ εW : ℝ)
    (κ C U W χ mfu : ℝ) (M Mu Mr Mx : Ω → ℕ) (Γ : ℕ) (α b_w b_a : ℝ) (r d : Declaration) (T : Threading r)
    (dd e : ℕ) (Mc : Containment r.D d.D),
    0 ≤ p → 0 < ε → (∀ i, 0 ≤ c i) → ∑ i, c i ≤ C → Assumptions.PouwSound μ wrong p →
    0 < H → 0 ≤ Δ → Φ * H ≤ C → (∀ ω, ∀ b ∈ bad ω, ident b ∈ I) →
    (∀ ω, Assumptions.TerminalDetection (bad ω) (detected ω) ident τ Δ) →
    (∀ ω j t, ∑ b ∈ bad ω with ident b = j ∧ t ≤ τ b ∧ τ b ≤ t + Δ, w b ≤ Φ * Δ) →
    0 < θ → (∀ b, 0 ≤ w b) → 0 < q → q ≤ 1 → 0 ≤ B → (∀ ω, ∀ b ∈ bad ω, w b ≤ B) →
    Assumptions.SampledDetection μ bad detected ident τ q →
    0 < κ → 0 < C → 0 ≤ U → 0 < mfu → mfu ≤ 1 → 0 ≤ α → 0 < b_w → 0 < b_a → 0 < dd →
    Complies Γ α d → Assumptions.BaselineRegistered r d (PartialContainment.ofContainment Mc) →
    T.Threaded Γ (α * Γ) dd e →
    Assumptions.WeightsIncompressible r T Γ (α * Γ) b_w b_a → (d.D.size d.D.computed : ℝ) ≤ C →
    (∀ ω, 0 < M ω) → (∀ ω, M ω ≤ Mu ω + Mr ω + T.M.card + Mx ω) →
    (∀ ω, (Mu ω : ℝ) ≤ mfu * κ * U / 2) →
    (∀ ω, (Mr ω : ℝ) ≤ κ * (C - (1 - εW) * W + wrongCapacity c (wrong ω) + ∑ b ∈ bad ω, w b) / 2) →
    (∀ ω, (Mx ω : ℝ) ≤ κ * χ * (C + U) / 2) →
      1 - (p / ε + I.card * ((1 - q) / q * B) / (θ * C)) ≤ prob μ (fun ω => efficiency κ (M ω) C U ≤
          unmonitoredShare C U +
              (1 - unmonitoredShare C U) * (residualShare W C + εW * W / C + ε + lagShare I.card Δ H + θ) +
            declaredShare κ (rhoT α b_a (T.bound (α * Γ) b_w dd e)) C U + χ ∧
        remoteDampener mfu (unmonitoredShare C U)
            ((1 - unmonitoredShare C U) * (residualShare W C + εW * W / C + ε + lagShare I.card Δ H + θ) +
              declaredShare κ (rhoT α b_a (T.bound (α * Γ) b_w dd e)) C U + χ) ≤
          hardwareMultiple mfu κ (M ω) C U)

end Nci.Guarantees
