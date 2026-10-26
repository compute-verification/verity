import Properties.Verity.Wipe
import Proofs.Properties.Verity.Soundness

/-!
# Soundness over the six subprotocols, from their targets

`Verity.Vocab.soundness` proves `Vocab.SoundnessWithWipe` from the six subprotocols' targets. The composition layer,
`Verity.soundness_of_subprotocols`, bounds the five claims onsite composes today by their errors summed, in the same
game and for the same strategy. An accepted wipe case whose claim fails at some server is the wipe's own event at that
server (NCI accepted the registration, by `accepts_registered`), which `WipeSound` bounds by its error. A union bound
adds them.

Beside it, its two checks: `soundness_honest`, the six targets hold wherever every claim holds on every play, and
`soundness_nontrivial`, its conclusion fails for a verifier that accepts everything, wherever one of the five claims
fails on every play of a strategy whose files pass the commit phase's check and the errors sum below one.
-/

namespace Verity.Vocab

open Accounting Core
open scoped ENNReal

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] {Srv : Type} [Fintype Srv]
  (V : Vocab d α) (W : WipeVocab V Srv)

theorem soundness (hS : V.SampledProofsSound) (hW : V.PouwSound) (hP : V.PousSound) (hB : V.BandwidthAuditSound)
    (hN : V.NetworkCertifierSound) (hWipe : W.WipeSound) : V.SoundnessWithWipe W := by
  intro θ F c σ
  refine le_trans (Game.prob_mono ?_ _ σ) (le_trans (Game.prob_or_le _ _ _ σ)
    (add_le_add (soundness_of_subprotocols ι α d V hS hW hP hB hN θ F c σ)
      (le_trans (prob_exists_finset_le Finset.univ _ _ σ)
        (Finset.sum_le_sum fun s _ => hWipe θ F c σ s))))
  rintro o ⟨hacc, hreg, hfail | ⟨hwa, s, hs⟩⟩
  · exact Or.inl ⟨hacc, hreg, hfail⟩
  · exact Or.inr ⟨s, Finset.mem_univ s, ⟨hreg, accepts_registered hacc, hwa⟩, hs⟩

/-- **Its hypotheses hold where every claim holds on every play**: the six targets, at any errors. -/
theorem soundness_honest (h : ∀ a θ F c σ o, V.Holds θ F c σ a o) (hw : ∀ θ F c σ s o, W.Holds θ F c σ s o) :
    (V.SampledProofsSound ∧ V.PouwSound ∧ V.PousSound ∧ V.BandwidthAuditSound ∧ V.NetworkCertifierSound) ∧
      W.WipeSound :=
  ⟨soundness_of_subprotocols_honest V h, fun θ F c σ s =>
    (Game.prob_mono (p := fun o => W.Verdict o ∧ ¬ W.Holds θ F c σ s o) (q := fun _ => False)
      (fun o ho => ho.2 (hw θ F c σ s o)) _ σ).trans ((Game.prob_false _ σ).le.trans (zero_le _))⟩

/-- **Its conclusion fails for a verifier that accepts everything**: at files that pass the commit phase's check, a
strategy on whose every play one of the five claims fails, and errors summing below one. -/
theorem soundness_nontrivial (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (hreg : V.Registered θ F)
    (a : Audit) (hfail : ∀ o, ¬ V.Holds θ F c σ a o)
    (hlt : (∑ a, V.error a θ F c σ) + ∑ s, W.error θ F c σ s < 1) :
    ¬ V.SoundnessWithWipeFor W fun _ => True := by
  intro h
  have h1 := prob_play_eq_one (V.rounds θ) θ F c
    (fun o => (fun _ => True) o.sent ∧ V.Registered o.params o.file ∧
      ((∃ a, ¬ V.Holds θ F c σ a o) ∨ (WipeAccepts o.sent ∧ ∃ s, ¬ W.Holds θ F c σ s o)))
    (fun _ _ => ⟨trivial, hreg, Or.inl ⟨a, hfail _⟩⟩) σ
  exact lt_irrefl (1 : ℝ≥0∞) ((h1.symm.trans_le (h θ F c σ)).trans_lt hlt)

end Verity.Vocab
