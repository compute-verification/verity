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
windows is wrong (`wrong ω i`: it credits work that window did not run as declared) with probability at most `p`,
`3 · 2⁻⁴⁰` per window. It bounds each window's marginal only: it says nothing about how windows depend on one another.
-/
def PouwSound {Ω : Type} [Fintype Ω] (μ : PMF Ω) {N : ℕ} (wrong : Ω → Fin N → Prop) (p : ℝ) : Prop :=
  ∀ i, prob μ (fun ω => wrong ω i) ≤ p

/-- **Detection is terminal** ([TERMINAL-DETECTION]): a detected cheat ends the identity and takes its hardware out of
use, with the verdict at most `Δ` after the identity's first faulty proof block. Of the faulty proof blocks `bad`, block
`b` run by identity `ident b` at time `τ b`, any two of one identity run within `Δ` of each other: none runs before that
identity's first, and none after its halt, `Δ` later. -/
def TerminalDetection {β ι : Type} (bad : Finset β) (ident : β → ι) (τ : β → ℝ) (Δ : ℝ) : Prop :=
  ∀ b ∈ bad, ∀ b' ∈ bad, ident b = ident b' → τ b' ≤ τ b + Δ

end Nci.Assumptions
