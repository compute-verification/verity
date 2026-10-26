import Properties.Verity.Wipe
import Proofs.Properties.Verity.Soundness

/-!
# Soundness over the six subprotocols, from their targets

`Verity.Vocab.soundness` proves `Vocab.SoundnessWithWipe` from the six subprotocols' targets. The composition layer,
`Verity.soundness_of_subprotocols`, bounds the five claims onsite composes today by their errors summed, in the same
game and for the same strategy. Outside the wipe's break, an accepted wipe case whose claim fails at some server is
the wipe's own event at that server, which `WipeSound` bounds. A union bound adds them.
-/

namespace Verity.Vocab

open Accounting Core

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] {Srv : Type} [Fintype Srv]
  (V : Vocab d α) (W : WipeVocab V Srv)

theorem soundness (hS : V.SampledProofsSound) (hW : V.PouwSound) (hP : V.PousSound) (hB : V.BandwidthAuditSound)
    (hN : V.NetworkCertifierSound) (hWipe : W.WipeSound) : V.SoundnessWithWipe W := by
  intro θ F c σ
  refine le_trans (Game.prob_mono ?_ _ σ) (le_trans (Game.prob_or_le _ _ _ σ)
    (add_le_add (soundness_of_subprotocols ι α d V hS hW hP hB hN θ F c σ)
      (le_trans (prob_exists_finset_le Finset.univ _ _ σ)
        (Finset.sum_le_sum fun s _ => hWipe θ F c σ s))))
  rintro o ⟨hacc, hreg, hfail | ⟨hwa, s, hs⟩, hVb, hWb⟩
  · exact Or.inl ⟨hacc, hreg, hfail, hVb⟩
  · refine Or.inr ⟨s, Finset.mem_univ s, ⟨hreg, hwa, fun r hr => ?_⟩, hs, hWb⟩
    by_contra hne
    exact hWb ⟨hwa, r, hr, hne⟩

end Verity.Vocab
