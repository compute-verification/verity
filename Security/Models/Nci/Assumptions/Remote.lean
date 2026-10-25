import Definitions.Nci.Remote

/-!
# The compute dampener, remote protocol: the named assumptions

The remote dampener's three derived terms each rest on one: the certificate slack on PoUW's soundness per window
(`PouwSound`, [POUW-SOUND]), the verdict lag on detection being terminal (`TerminalDetection`, [TERMINAL-DETECTION]),
and the escape slack on sampled proofs catching each faulty block (`SampledEscape`, [SAMPLED-ESCAPE]).
-/

namespace Nci.Assumptions

open Nci.Dampener

/-- **PoUW's certificates are sound** ([POUW-SOUND]; PoUW's `CircuitCredit`, draft #2638): over the outcomes `Ω` of a
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
constrains nothing, and the faulty blocks an identity runs before its first detection (`escaped`) are charged by
`SampledEscape`. When every proof block is proved (tiers T1–T4) every faulty block fails its own proof, `detected` holds
them all, and it says that any two faulty blocks of one identity run within `Δ` of each other. -/
def TerminalDetection {β ι : Type} (bad detected : Finset β) (ident : β → ι) (τ : β → ℝ) (Δ : ℝ) : Prop :=
  ∀ b ∈ bad, b ∈ detected → ∀ b' ∈ bad, ident b = ident b' → τ b ≤ τ b' → τ b' ≤ τ b + Δ

/-- **Sampled proofs catch each faulty block with probability at least `q`** ([SAMPLED-ESCAPE]): over the outcomes `Ω`,
drawn from `μ`, the declared work that identity `j`'s faulty proof blocks accepted before its first detection
(`escaped`) skip, `w_b` each, is at most `((1 − q)/q) B` in expectation, `B` the most one faulty block skips.

The coins of a proof block are live, drawn after its commitment, so whatever the history a faulty block is detected
with probability at least `q`: the sampled share (tiers' `share/(overhead + replay)`) times one less the luck `p` of a
drawn faulty block. An identity then runs `k` or more faulty blocks before its first detection with probability at most
`(1 − q)^k`, and at most `∑_{k ≥ 1} (1 − q)^k = (1 − q)/q` of them in expectation, the stopping-time argument of
`Population.Exfil`'s windowed audit (`ex_cumUpTo`), not yet joined to these outcomes. With every proof block proved,
`q = 1` and none escapes. It bounds each identity's expectation only: it says nothing about how identities depend on one
another. -/
def SampledEscape {Ω β ι : Type} [Fintype Ω] [DecidableEq ι] (μ : PMF Ω) (bad detected : Ω → Finset β) (ident : β → ι)
    (τ w : β → ℝ) (q B : ℝ) : Prop :=
  ∀ j, ∑ ω, (μ ω).toReal * ∑ b ∈ escaped (bad ω) (detected ω) ident τ with ident b = j, w b ≤ (1 - q) / q * B

end Nci.Assumptions
