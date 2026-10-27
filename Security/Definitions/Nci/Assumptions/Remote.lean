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

end Nci.Assumptions
