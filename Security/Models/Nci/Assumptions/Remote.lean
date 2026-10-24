import Definitions.Nci.Remote

/-!
# The compute dampener, remote protocol: the named assumptions

The remote dampener's two derived terms each rest on one: the certificate slack on PoUW's soundness per window
(`PouwSound`, [POUW-SOUND]), and the verdict lag on detection being terminal (`TerminalDetection`,
[TERMINAL-DETECTION]).
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
use, with the verdict at most `Δ` after the identity's first faulty proof block. Of the faulty proof blocks `bad`, block
`b` run by identity `ident b` at time `τ b`, any two of one identity run within `Δ` of each other: none runs before that
identity's first, and none after its halt, `Δ` later.

It holds when every proof block is proved (tiers T1–T4): a faulty block fails its own proof, so the first one is
detected. Under sampled proofs (T5) a faulty block that isn't drawn is accepted, with probability `1 − q` at a sampled
share `q`, so an identity can run one faulty block and another more than `Δ` later, and on that outcome the premise
fails. `Nci.Guarantees.DampenerRemote` reads it on every outcome, so with sampling (`χ > 0`) it covers only strategies
whose faulty blocks lie within `Δ` of each other per identity; the blocks accepted before an identity's first detection
must be charged on their own, as tiers' luck term charges them. -/
def TerminalDetection {β ι : Type} (bad : Finset β) (ident : β → ι) (τ : β → ℝ) (Δ : ℝ) : Prop :=
  ∀ b ∈ bad, ∀ b' ∈ bad, ident b = ident b' → τ b' ≤ τ b + Δ

end Nci.Assumptions
