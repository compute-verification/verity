import Properties.Verity.Soundness
import Definitions.Accounting.JointBlocks
import MemoryChallenger.WipeProtocol

/-!
# The wipe, the sixth subprotocol, in the commit-then-coin game

Onsite is made of six subprotocols: sampled proofs, PoUW, PoUS, the bandwidth audit, the network certifier and the
wipe. `Properties.Verity.Game` states the first five at onsite's compiled components; this module adds the wipe, whose
compiled component is `MemoryChallenger.WipeProtocol.protocol` (`verity/wipe/v0`), in the same form:

* the wipe's verdict at the coins `ξ` (`WipeVocab.run`) is its component's acceptance of its own case in what the
  prover sent (`WipeAccepts`), with the coins its check read equal to the ones the device logged for it;
* its claim is the survivor bound of the wipe that opens the period (`Accounting.BlockWipe.Survives`, the one
  `Accounting.Guarantees.JointHorizonBlocks` reads): at most `f · fill + 2λ` bits of foreign content, plus the state the
  fill leaves out, survive beyond the committed carried state;
* its target (`WipeVocab.WipeSound`, one-10b's, MEM-93) is that survivor bound's per-period guarantee on the wipe's
  own coordinate of the device's coins, in `Accounting.PeriodRun.Sound`'s form.

`Vocab.SoundnessWithWipe` is `Vocab.Soundness` with the wipe's claim among the period's claims and its error in the
bound. The wipe's claim is counted where the wipe's case is accepted: onsite's components don't include
`verity/wipe/v0` yet, so until `Core.Onsite.components` composes it, `WipeAccepts` holds of nothing and the wipe's term
bounds nothing.

The names the wipe defines are `WipeVocab`'s fields: `own` and `coinsRead` (ONE-4), the fill and what survives it
(one-10b), and the fraction and error the product publishes, from the parameters.
-/

namespace Verity

open Accounting Core

/-- The wipe's component accepts its own case in what the prover sent. -/
def WipeAccepts (s : Sent) : Prop := CaseAccepts MemoryChallenger.WipeProtocol.protocol s

/-- **The names the wipe defines.** The coordinate of the device's log the memory challenger's draws are on, and the
coins the wipe's check read, decoded from the device's signed log entries in what the prover sent (ONE-4); the bits
of device memory the fill covers and the bits of foreign content that survive it, in each play (one-10b); and the
fraction its survivor bound is at and its error, from the parameters alone. -/
structure WipeVocab {ι : Type} (d : Device ι) {I : Type} (Ω : I → Type) where
  own : I
  coinsRead : Sent → Option (Ω own)
  fill : Play ((i : I) → Ω i) → ℕ
  fill_le : ∀ τ, fill τ ≤ d.mem
  kept : Play ((i : I) → Ω i) → ((i : I) → Ω i) → ℕ
  fraction : ByteArray → ℝ
  delta : ByteArray → ℝ

namespace WipeVocab

variable {ι : Type} {d : Device ι} {I : Type} {Ω : I → Type} (W : WipeVocab d Ω)

/-- The wipe the play `τ` realizes: its fill and what survives it, and its verdict at the coins `ξ` the wipe
component's acceptance of what the prover sent, on the coins the device logged for it. -/
def run (τ : Play ((i : I) → Ω i)) : BlockWipe d Ω where
  fill := W.fill τ
  fill_le := W.fill_le τ
  accepts := fun ξ => WipeAccepts (τ.sent ξ) ∧ W.coinsRead (τ.sent ξ) = some (ξ W.own)
  kept := W.kept τ

/-- **The wipe's break**: the wipe's component accepts what the prover sent on coins other than those the device
logged for it, a signature, under the device's key, on a message missing from the device's log. -/
def Break (τ : Play ((i : I) → Ω i)) (ξ : (i : I) → Ω i) : Prop :=
  WipeAccepts (τ.sent ξ) ∧ W.coinsRead (τ.sent ξ) ≠ some (ξ W.own)

variable [DecidableEq I] [∀ i, Fintype (Ω i)]

/-- **The wipe's target (one-10b, MEM-93)**: at every strategy and every value of the other coins, the probability over
the wipe's own coins that it accepts, on the device's logged coins, while its survivor bound at the published fraction
fails is at most the published error. -/
def WipeSound : Prop :=
  ∀ (τ : Play ((i : I) → Ω i)) (ξ : (i : I) → Ω i),
    Pouw.pr (fun x : Ω W.own => (W.run τ).accepts (Function.update ξ W.own x) ∧
        ¬ (W.run τ).Survives (W.fraction τ.params) (Function.update ξ W.own x)) ≤ W.delta τ.params

end WipeVocab

namespace Vocab

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] {I : Type} [Fintype I] [DecidableEq I] {Ω : I → Type}
  [∀ i, Fintype (Ω i)]

/-- **Soundness over the six subprotocols, in the commit-then-coin form.** For every strategy there is a transcript
`X`, fixed by the commitment, such that the probability over the device's coins that onsite's compiled verifier
accepts while one of the period's claims fails (one of `Vocab.Soundness`'s five, or the wipe's case is accepted while
its survivor bound fails), with no break, is at most the published errors summed, the wipe's included. -/
def SoundnessWithWipe (V : Vocab d α Ω) (W : WipeVocab d Ω) : Prop :=
  ∀ τ : Play ((i : I) → Ω i), ∃ X : ByteArray → (V.circuitOf τ.params).Transcript,
    Pouw.pr (fun ξ => Accepts (τ.sent ξ) ∧
        ((∃ a, ¬ (V.run τ (X τ.commitment)).Holds (V.certOf τ.params) (V.priceOf τ.params) a ξ) ∨
          (WipeAccepts (τ.sent ξ) ∧ ¬ (W.run τ).Survives (W.fraction τ.params) ξ)) ∧
        ¬ V.Break τ ξ ∧ ¬ W.Break τ ξ)
      ≤ (∑ a, V.delta τ.params a) + W.delta τ.params

end Vocab

end Verity
