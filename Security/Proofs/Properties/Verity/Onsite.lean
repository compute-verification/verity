import Properties.Verity.Onsite
import Proofs.Properties.Verity.Soundness

/-!
# Onsite's honest lemma

`Verity.Onsite.soundness_honest`: at onsite's honest instance, each claim onsite decides holds on almost every play of
its honest prover at the honest commitment (`Vocab.HonestHolds G.commit`), sampled proofs' from
`Vocab.holds_sampled_of_opens` at the honest prover's case (`SampledHonest`), and each other claim's from its honest
lemma (`NetworkHonest`, `PouwHonest`, `PousHonest`, `BandwidthHonest`). `soundness_honest_event` is
`soundness_at_honest`'s event at onsite: onsite's compiled verifier accepting registered files while a claim fails has
probability `0` against the honest prover. The model-level instance stays the one `_nontrivial`
(`soundness_at_nontrivial_of … .sampled`).

`Vocab.not_honestHolds_sampled_of_unopened` is why `Vocab.HonestHolds` reads the honest commitment: at a commitment
that opens no gate, sampled proofs' claim fails on every play, whatever the strategy.
-/

namespace Verity.Onsite

open Accounting Core
open scoped ENNReal

variable {ι α : Type} [Fintype α] {d : Device ι}

/-- **Sampled proofs' claim at the honest commitment**, from the honest prover's case. -/
theorem honestHolds_sampled (G : Given d α) (h : SampledHonest G) :
    (vocab G).HonestHolds G.commit .sampled := by
  intro θ F hreg
  obtain ⟨hc, ho, hε⟩ := h θ F hreg
  exact prob_play_eq_zero _ _ θ F _ _ (fun log e b hn => hn ((vocab G).holds_sampled_of_opens θ F (G.commit θ F)
    ((vocab G).honest θ F (G.commit θ F)) ⟨θ, F, G.commit θ F, log, e, b⟩ hc ho hε)) _

/-- **Onsite's honest lemma.** -/
theorem soundness_honest : SoundnessHonest := by
  intro ι α _ d G hs hn hw hp hb a _
  cases a
  · exact honestHolds_sampled G hs
  · exact hw
  · exact hp
  · exact hb
  · exact hn

/-- **`soundness_at_honest`'s event at onsite**: at registered files and the honest commitment, onsite's compiled
verifier accepting while a claim fails has probability `0` against the honest prover. -/
theorem soundness_honest_event (G : Given d α) (hs : SampledHonest G) (hn : NetworkHonest G) (hw : PouwHonest G)
    (hp : PousHonest G) (hb : BandwidthHonest G) (θ F : ByteArray) (hreg : (vocab G).Registered θ F) :
    Game.prob (fun o => AcceptsAt onsite o.sent ∧ (vocab G).Registered o.params o.file ∧
        ∃ a, onsite.Decides a ∧ ¬ (vocab G).Holds θ F (G.commit θ F) ((vocab G).honest θ F (G.commit θ F)) a o)
      ((vocab G).game θ F (G.commit θ F)) ((vocab G).honest θ F (G.commit θ F)) = 0 :=
  soundness_at_honest (vocab G) onsite G.commit (soundness_honest ι α d G hs hn hw hp hb) θ F (G.commit θ F) hreg rfl

end Verity.Onsite

namespace Verity.Vocab

open Accounting

variable {ι α : Type} [Fintype α] {d : Device ι} (V : Vocab d α)

open Classical in
/-- **`HonestHolds .sampled` fails at a commitment function that opens nothing**: at registered files and a commitment
`c = commit θ F` whose positions hold no gate that isn't a registered constant, every unit reading or writing such a gate is unbound
whatever the extractor returns, so where those units are more than an `ε` share, sampled proofs' claim fails on every
play of every strategy, the honest one's included. -/
theorem not_honestHolds_sampled_of_unopened (commit : ByteArray → ByteArray → ByteArray) (θ F c : ByteArray)
    (hc : commit θ F = c) (hreg : V.Registered θ F)
    (hplace : ∀ g, (V.read F).consts g = none → V.place θ F c g = none)
    (hlt : (V.certOf θ).ε * V.units F <
      ((Finset.univ.filter fun u => ∃ g ∈ (V.partsOf F).io u, (V.read F).consts g = none).card : ℝ)) :
    ¬ V.HonestHolds commit .sampled := by
  intro h
  have hfail : ∀ (σ : Game.Strategy (V.game θ F c)) (o : Outcome), ¬ V.Holds θ F c σ .sampled o := by
    intro σ o hh
    have hsub : (Finset.univ.filter fun u => ∃ g ∈ (V.partsOf F).io u, (V.read F).consts g = none) ⊆
        Accounting.unbound (V.partsOf F) (V.Opens θ F c) (V.extract θ F c σ) := by
      intro u hu
      obtain ⟨g, hg, hk⟩ := (Finset.mem_filter.1 hu).2
      refine Finset.mem_filter.2 ⟨Finset.mem_univ u, Or.inr ⟨g, hg, fun ho => ?_⟩⟩
      simp only [Opens, hk] at ho
      obtain ⟨p, hp, -⟩ := ho
      rw [hplace g hk] at hp
      cases hp
    have hcard := Finset.card_le_card hsub
    have hh' : ((Accounting.unbound (V.partsOf F) (V.Opens θ F c) (V.extract θ F c σ)).card : ℝ) ≤
        (V.certOf θ).ε * V.units F := hh
    exact lt_irrefl _ (hlt.trans_le ((Nat.cast_le.2 hcard).trans hh'))
  have h1 := prob_play_eq_one (V.rounds θ) (V.sign θ) θ F c (fun o => ¬ V.Holds θ F c (V.honest θ F c) .sampled o)
    (fun log e b => hfail _ _) (V.honest θ F c)
  subst hc
  have h0 := h θ F hreg
  exact one_ne_zero (h1.symm.trans h0)

end Verity.Vocab
