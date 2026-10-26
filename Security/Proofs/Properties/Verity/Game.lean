import Properties.Verity.Wipe
import Proofs.Properties.Verity.Soundness

/-!
# Soundness over the six subprotocols, from their targets

`Verity.Vocab.soundness` proves `Vocab.SoundnessWithWipe` from the six subprotocols' targets. The composition layer,
`Verity.soundness_of_subprotocols`, bounds the five claims onsite composes today by their errors summed, in the same
game and for the same strategy. An accepted wipe case whose claim fails at some server is the wipe's own event at that
server (NCI accepted the registration, by `accepts_registered`), which `WipeSound` bounds by its error. A union bound
adds them.

The wipe's target follows from memory-accounting's `Pous.Guarantees.StreamedWipeMeetsServer5GB`, the coupling of the
wipe model's game into the causal game, and the published wipes at the 5 GB point (`WipeVocab.wipeSound_of_streamed`).

Beside it, its two checks: `soundness_honest`, at the honest deployment every target's event and the theorem's own have
probability `0`, and `soundness_nontrivial`, its conclusion fails for a verifier that accepts everything, wherever one
of the five claims fails on every play of a strategy whose files pass the commit phase's check and the errors sum below
one.
-/

namespace Verity

open Accounting Core
open scoped ENNReal

namespace WipeVocab

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] {V : Vocab d α} {Srv : Type} (W : WipeVocab V Srv)

/-- **The wipe's target from the streamed wipe's per-server bound** at the 5 GB point, through the coupling. -/
theorem wipeSound_of_streamed (h5 : Pous.Guarantees.StreamedWipeMeetsServer5GB) (hc : W.Couples)
    (hop : W.AtOperatingPoint) : W.WipeSound := by
  intro θ F c σ s
  obtain ⟨hn, hr, hrk, hpt⟩ := hop θ s
  refine (hc θ F c σ s).trans ?_
  unfold WipeVocab.error
  refine add_le_add ?_ le_rfl
  exact h5 (512 * W.n θ s + W.R θ s) (W.n θ s) (W.R θ s) rfl hn (W.r θ s) (W.k θ s) hr hrk hpt
    (40000000000 + W.egressBound θ s) (W.egressBound θ s) (W.κ θ s) (W.prf θ s) (W.prover θ F c σ s) le_rfl

end WipeVocab

namespace Vocab

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

/-- **Its hypotheses hold at the honest deployment**: at registered files, where each subprotocol's honest lemma holds
(`Vocab.HonestHolds`, `WipeVocab.HonestHolds`), the deployment's honest prover meets the six targets with probability
`0` on their events, and the event the theorem bounds has probability `0`. -/
theorem soundness_honest (h : ∀ a, V.HonestHolds a) (hw : W.HonestHolds) (θ F c : ByteArray)
    (hreg : V.Registered θ F) :
    (∀ s, Game.prob (fun o => WipeVocab.Verdict V o ∧ ¬ W.Holds θ F c (V.honest θ F c) s o) (V.game θ F c)
        (V.honest θ F c) ≤ W.error θ F c (V.honest θ F c) s) ∧
      Game.prob (fun o => Accepts o.sent ∧ V.Registered o.params o.file ∧
          ((∃ a, ¬ V.Holds θ F c (V.honest θ F c) a o) ∨
            (WipeAccepts o.sent ∧ ∃ s, ¬ W.Holds θ F c (V.honest θ F c) s o))) (V.game θ F c) (V.honest θ F c) = 0 := by
  have h5 := (soundness_of_subprotocols_honest V h θ F c hreg).2
  have h0 : ∀ s, Game.prob (fun o => ¬ W.Holds θ F c (V.honest θ F c) s o) (V.game θ F c) (V.honest θ F c) = 0 :=
    hw θ F c hreg
  refine ⟨fun s => ?_, le_antisymm ?_ zero_le⟩
  · refine le_trans (Game.prob_mono (p := fun o => WipeVocab.Verdict V o ∧ ¬ W.Holds θ F c (V.honest θ F c) s o)
      (q := fun o => ¬ W.Holds θ F c (V.honest θ F c) s o) (fun o ho => ho.2) _ _) ?_
    rw [h0 s]
    exact zero_le
  · calc Game.prob (fun o => Accepts o.sent ∧ V.Registered o.params o.file ∧
            ((∃ a, ¬ V.Holds θ F c (V.honest θ F c) a o) ∨
              (WipeAccepts o.sent ∧ ∃ s, ¬ W.Holds θ F c (V.honest θ F c) s o))) (V.game θ F c) (V.honest θ F c)
        ≤ Game.prob (fun o => (Accepts o.sent ∧ V.Registered o.params o.file ∧
              ∃ a, ¬ V.Holds θ F c (V.honest θ F c) a o) ∨
            ∃ s ∈ Finset.univ, ¬ W.Holds θ F c (V.honest θ F c) s o) (V.game θ F c) (V.honest θ F c) :=
          Game.prob_mono (fun o ⟨hacc, hreg, hf⟩ => hf.elim (fun ha => Or.inl ⟨hacc, hreg, ha⟩)
            fun ⟨_, s, hs⟩ => Or.inr ⟨s, Finset.mem_univ s, hs⟩) _ _
      _ ≤ _ := Game.prob_or_le _ _ _ _
      _ ≤ 0 + ∑ s, Game.prob (fun o => ¬ W.Holds θ F c (V.honest θ F c) s o) (V.game θ F c) (V.honest θ F c) :=
          add_le_add h5.le (prob_exists_finset_le _ _ _ _)
      _ = 0 := by simp only [h0, Finset.sum_const_zero, add_zero]

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

end Vocab

end Verity
