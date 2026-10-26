import Properties.Verity.Wipe
import Proofs.Properties.Verity.Soundness
import Proofs.Accounting.PouwWork
import Proofs.Core.UnionBound

/-!
# Soundness over the six subprotocols, from their targets

`Verity.Vocab.soundness` proves `Vocab.SoundnessWithWipe` from the six subprotocols' targets. The composition layer,
`Verity.soundness_of_subprotocols`, bounds the five claims onsite composes today by their errors summed, at a
transcript fixed by the commitment. Outside the wipe's break, an accepted wipe case whose survivor bound fails is the
wipe's own event on the device's logged coins, which `WipeSound` bounds over the wipe's coordinate at every value of the
others, so over every coin (`Pouw.pr_le_of_update`). A union bound adds the two.
-/

namespace Verity.Vocab

open Accounting Core

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] {I : Type} [Fintype I] [DecidableEq I]
  {Ω : I → Type} [∀ i, Fintype (Ω i)] [∀ i, Nonempty (Ω i)] (V : Vocab d α Ω) (W : WipeVocab d Ω)

theorem soundness (hS : V.SampledProofsSound) (hW : V.PouwSound) (hP : V.PousSound) (hB : V.BandwidthAuditSound)
    (hN : V.NetworkCertifierSound) (hWipe : W.WipeSound) : V.SoundnessWithWipe W := by
  intro τ
  obtain ⟨X, hX⟩ := soundness_of_subprotocols ι α I Ω d V hS hW hP hB hN τ
  refine ⟨X, le_trans (Accounting.SecurityProofs.PouwWork.pr_mono ?_)
    (le_trans (Accounting.SecurityProofs.PouwWork.pr_or_le _ _)
      (add_le_add hX (Pouw.pr_le_of_update W.own
        (fun ξ => (W.run τ).accepts ξ ∧ ¬ (W.run τ).Survives (W.fraction τ.params) ξ) (hWipe τ))))⟩
  rintro ξ ⟨hacc, hfail | ⟨hwa, hsurv⟩, hVb, hWb⟩
  · exact Or.inl ⟨hacc, hfail, hVb⟩
  · refine Or.inr ⟨?_, hsurv⟩
    show WipeAccepts (τ.sent ξ) ∧ W.coinsRead (τ.sent ξ) = some (ξ W.own)
    refine ⟨hwa, ?_⟩
    by_contra hne
    exact hWb ⟨hwa, hne⟩

end Verity.Vocab
