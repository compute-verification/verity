import Definitions.Nci.Remote

/-!
# The compute dampener, remote protocol: the named assumptions

The remote dampener's three derived terms each rest on one: the certificate slack on PoUW's soundness per window
(`PouwSound`, [POUW-SOUND]), the verdict lag on detection being terminal (`TerminalDetection`, [TERMINAL-DETECTION]),
and the escape slack on sampled proofs catching each faulty block whatever came before (`SampledDetection`,
[SAMPLED-DETECTION]), from which `Nci.SecurityProofs.sampled_escape` bounds what each identity's escaped blocks skip.
-/

namespace Nci.Assumptions

open Nci.Dampener

/-- **PoUW's certificates are sound** ([POUW-SOUND]; PoUW's `CircuitCredit`): over the outcomes `Ω` of a
deployment, drawn from `μ` (the verifier's coins, the developer's strategy fixed), the certificate of each of the `N`
windows is wrong (`wrong ω i`: it credits more of that window's work than ran, beyond its deficit `ε`) with probability
at most `p`, `3 · 2⁻⁴⁰` per window. It bounds each window's marginal only: it says nothing about how windows depend on
one another.

What backs `p` is `Accounting.Guarantees.Compose.CircuitCredit`: a certificate is accepted with the window's time below
`(1 − ε)·T_cred` with probability at most `δ_s + δ_bind + δ_ks`, `3 · 2⁻⁴⁰` at `2⁻⁴⁰` each, `ε` being 0.744%. Nothing
bounds how likely a certificate is to credit up to `ε` of its window that did not run, so that share is not `wrong`
and belongs in the residual, through the certified work `W`. -/
def PouwSound {Ω : Type} [Fintype Ω] (μ : PMF Ω) {N : ℕ} (wrong : Ω → Fin N → Prop) (p : ℝ) : Prop :=
  ∀ i, prob μ (fun ω => wrong ω i) ≤ p

/-- **Detection is terminal** ([TERMINAL-DETECTION]): a detected cheat ends the identity and takes its hardware out of
use, with the verdict at most `Δ` after the detected block. Of the faulty proof blocks `bad`, block `b` run by identity
`ident b` at time `τ b`, those `detected` (drawn, and failing their proof): no faulty block of a detected block's
identity runs at or after it by more than `Δ`, the identity having halted.

It holds under sampled proofs (T5) too: a faulty block that isn't drawn is accepted and isn't `detected`, so it
constrains nothing, and the faulty blocks an identity runs before its first detection (`escaped`) are charged under
`SampledDetection`. When every proof block is proved (tiers T1–T4) every faulty block fails its own proof, `detected`
holds them all, and it says that any two faulty blocks of one identity run within `Δ` of each other. -/
def TerminalDetection {β ι : Type} (bad detected : Finset β) (ident : β → ι) (τ : β → ℝ) (Δ : ℝ) : Prop :=
  ∀ b ∈ bad, b ∈ detected → ∀ b' ∈ bad, ident b = ident b' → τ b ≤ τ b' → τ b' ≤ τ b + Δ

/-- **Sampled proofs catch each faulty block with probability at least `q`, whatever came before**
([SAMPLED-DETECTION]): over the outcomes `Ω`, drawn from `μ`, a block `b` that is faulty (`bad`) with no other faulty
block of its identity detected at or before it (`noDetectionBefore`) is not detected itself with probability at most
`(1 − q)` times that of the condition: `P(b faulty, none earlier detected, b undetected) ≤ (1 − q) P(b faulty, none
earlier detected)`, for each block `b`.

`q` is the sampled share (tiers' `share/(overhead + replay)`) times one less the luck `p` of a drawn faulty block. The
coins that draw a proof block and check it are live: drawn after the commitment of every block at or before it, so
after whether `b` is faulty and whether the earlier blocks are is fixed, and independently of the coins of the other
blocks. Whatever happened before, a faulty block is then drawn with probability at least the share and, drawn, fails its
proof but with the luck `p`. With every proof block proved the share is 1 and `q = 1 − p`. It is each block's
conditional chance only: it says nothing about how identities depend on one another. -/
def SampledDetection {Ω β ι : Type} [Fintype Ω] (μ : PMF Ω) (bad detected : Ω → Finset β) (ident : β → ι) (τ : β → ℝ)
    (q : ℝ) : Prop :=
  ∀ b, prob μ (fun ω => b ∈ bad ω ∧ b ∉ detected ω ∧ noDetectionBefore (bad ω) (detected ω) ident τ b) ≤
    (1 - q) * prob μ (fun ω => b ∈ bad ω ∧ noDetectionBefore (bad ω) (detected ω) ident τ b)

/-! Over a monitoring period of `T` windows, window `t` with its own uniform coins `Ω t` and a run of the period
`ξ : (t : Fin T) → Ω t`, each protocol's leg is a bound per window at any value of the other windows' coins, over its own
(`ξ[t ↦ a]`, `a` uniform on `Ω t`), as `Proofs.Flock.Soundness.Audit.Periods` states a period's bound. -/

/-- **PoUW's per-window bound** (in place of `PouwSound`): window `t`'s certificate is wrong (`wrong t`) with
probability at most `p` over its own coins, `CircuitCredit`'s conclusion at window `t`'s play. That the deployment's
window is such a play is not stated. -/
def PouwPerWindow {T : ℕ} {Ω : Fin T → Type} [∀ t, Fintype (Ω t)] [∀ t, Nonempty (Ω t)]
    (wrong : Fin T → ((t : Fin T) → Ω t) → Prop) (p : ℝ) : Prop :=
  ∀ t (ξ : (t : Fin T) → Ω t), prob (PMF.uniformOfFintype (Ω t)) (fun a => wrong t (Function.update ξ t a)) ≤ p

/-- **Sampled proofs' per-window bounds** (in place of `SampledDetection`): a window's faulty blocks `bad` (block `b` in
window `win b`) are fixed before its coins, and the units it draws (`drawn`) and the faulty blocks it detects
(`detected`) read only its coins and the earlier windows'; its draw misses its `m` faulty blocks with probability at most
`(1 − q)^m` (`Flock.Guarantees.HiddenWindow.drawMiss_le`, under A3), and it detects none of them while drawing one with
probability at most `ks t` (`HiddenWindow.accepted_drawnWrong_le`). That the deployment's windows are such windows is not
stated. -/
def SampledPerWindow {T : ℕ} {Ω : Fin T → Type} [∀ t, Fintype (Ω t)] [∀ t, Nonempty (Ω t)] {β : Type}
    (bad drawn detected : ((t : Fin T) → Ω t) → Finset β) (win : β → Fin T) (q : ℝ) (ks : Fin T → ℝ) : Prop :=
  (∀ t (ξ ξ' : (t : Fin T) → Ω t), (∀ s, s < t → ξ s = ξ' s) → ∀ b, win b = t → (b ∈ bad ξ ↔ b ∈ bad ξ')) ∧
    (∀ t (ξ ξ' : (t : Fin T) → Ω t), (∀ s, s ≤ t → ξ s = ξ' s) → ∀ b, win b = t →
      (b ∈ drawn ξ ↔ b ∈ drawn ξ') ∧ (b ∈ detected ξ ↔ b ∈ detected ξ')) ∧
    (∀ t (ξ : (t : Fin T) → Ω t),
      prob (PMF.uniformOfFintype (Ω t)) (fun a => ∀ b ∈ bad ξ, win b = t → b ∉ drawn (Function.update ξ t a)) ≤
        (1 - q) ^ ((bad ξ).filter fun b => win b = t).card) ∧
    ∀ t (ξ : (t : Fin T) → Ω t),
      prob (PMF.uniformOfFintype (Ω t)) (fun a => (∃ b ∈ bad ξ, win b = t ∧ b ∈ drawn (Function.update ξ t a)) ∧
          ∀ b ∈ bad ξ, win b = t → b ∉ detected (Function.update ξ t a)) ≤ ks t

/-- **The developer's identities over the period**: window `t` is run by identity `idW t` at time `τw t`, each identity's
windows in their order; every faulty block (`bad`, block `b` in window `win b`) is run by one of the identities `I`;
the hardware's rate `Φ` over the period's length `H` is within the monitored compute `C`; one identity's faulty blocks
within any span of `Δ` skip at most `Φ Δ` of declared work (`w`); and each skips at most `B`. -/
def IdentitySchedule {T : ℕ} {Ξ β ι : Type} [DecidableEq ι] (bad : Ξ → Finset β) (win : β → Fin T) (I : Finset ι)
    (idW : Fin T → ι) (τw : Fin T → ℝ) (w : β → ℝ) (Φ Δ H C B : ℝ) : Prop :=
  (∀ t t', idW t = idW t' → t < t' → τw t < τw t') ∧ (∀ ξ, ∀ b ∈ bad ξ, idW (win b) ∈ I) ∧ Φ * H ≤ C ∧
    (∀ ξ j s, ∑ b ∈ bad ξ with idW (win b) = j ∧ s ≤ τw (win b) ∧ τw (win b) ≤ s + Δ, w b ≤ Φ * Δ) ∧
    ∀ ξ, ∀ b ∈ bad ξ, w b ≤ B

end Nci.Assumptions
