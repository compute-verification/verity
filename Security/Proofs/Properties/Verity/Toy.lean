import Properties.Verity.Toy
import Proofs.Properties.Verity.Soundness

/-!
# The toy deployment's honest lemma

`Verity.Toy.soundness_honest`: at the honest deployed instance, each claim the toy deployment decides holds on almost
every play of its honest prover, sampled proofs' from `Vocab.holds_sampled_of_opens` at the honest prover's case
(`SampledHonest`), the network certifier's from its honest lemma (`NetworkHonest`) at the honest commitment.
`soundness_honest_event` is `soundness_at_honest`'s event at the toy, at that instance: the toy's compiled verifier
accepting registered files while a claim it decides fails has probability `0`. ONE19's model-level instance stays the
one `_nontrivial` (`soundness_at_nontrivial_of … .sampled`); the toy gets none of its own.

`Vocab.not_honestHolds_sampled_of_unopened` is why the statement is at the honest commitment: `Vocab.HonestHolds
.sampled` holds the honest prover to sampled proofs' claim at every commitment, and at one that opens no gate it fails
on every play, whatever the strategy.
-/

namespace Verity.Toy

open Accounting Core
open scoped ENNReal

variable {ι α : Type} [Fintype α] {d : Device ι}

theorem comp_mem (k : Index) : comp k ∈ Core.Toy.components := by
  cases k <;> simp [comp, Core.Toy.components]

/-- The toy decides exactly sampled proofs' and the network certifier's claims. -/
theorem decides_iff (a : Audit) : deployment.Decides a ↔ a = .sampled ∨ a = .network := by
  cases a <;> simp [Deployment.Decides, deployment, ofAudit]

/-- **Sampled proofs' claim at the honest deployed instance**, from the honest prover's case. -/
theorem honestHoldsAt_sampled (G : Given d α) (h : SampledHonest G) :
    HonestHoldsAt (vocab G) G.commit (audit .sampled) := by
  intro θ F hreg
  obtain ⟨hc, ho, hε⟩ := h θ F hreg
  exact prob_play_eq_zero _ θ F _ _ (fun log e hn => hn ((vocab G).holds_sampled_of_opens θ F (G.commit θ F)
    ((vocab G).honest θ F (G.commit θ F)) ⟨θ, F, G.commit θ F, log, e⟩ hc ho hε)) _

/-- **The network certifier's claim at the honest deployed instance**, from its honest lemma. -/
theorem honestHoldsAt_network (G : Given d α) (h : NetworkHonest G) :
    HonestHoldsAt (vocab G) G.commit (audit .network) :=
  fun θ F hreg => h θ F (G.commit θ F) hreg

/-- **The toy's honest lemma.** -/
theorem soundness_honest : SoundnessHonest := by
  intro ι α _ d G hs hn a ha
  rcases (decides_iff a).1 ha with rfl | rfl
  · exact honestHoldsAt_sampled G hs
  · exact honestHoldsAt_network G hn

/-- **`soundness_at_honest`'s event at the toy**: at registered files and the honest commitment, the toy's compiled
verifier accepting while a claim it decides fails has probability `0` against the honest prover. -/
theorem soundness_honest_event (G : Given d α) (hs : SampledHonest G) (hn : NetworkHonest G) (θ F : ByteArray)
    (hreg : (vocab G).Registered θ F) :
    Game.prob (fun o => AcceptsAt deployment o.sent ∧ (vocab G).Registered o.params o.file ∧
        ∃ a, deployment.Decides a ∧ ¬ (vocab G).Holds θ F (G.commit θ F) ((vocab G).honest θ F (G.commit θ F)) a o)
      ((vocab G).game θ F (G.commit θ F)) ((vocab G).honest θ F (G.commit θ F)) = 0 := by
  classical
  set V := vocab G
  set c := G.commit θ F
  refine le_antisymm ?_ zero_le
  calc Game.prob (fun o => AcceptsAt deployment o.sent ∧ V.Registered o.params o.file ∧
          ∃ a, deployment.Decides a ∧ ¬ V.Holds θ F c (V.honest θ F c) a o) (V.game θ F c) (V.honest θ F c)
      ≤ Game.prob (fun o => ∃ a ∈ Finset.univ.filter deployment.Decides, ¬ V.Holds θ F c (V.honest θ F c) a o)
          (V.game θ F c) (V.honest θ F c) :=
        Game.prob_mono (fun o ⟨_, _, a, ha, hf⟩ => ⟨a, Finset.mem_filter.2 ⟨Finset.mem_univ a, ha⟩, hf⟩) _ _
    _ ≤ ∑ a ∈ Finset.univ.filter deployment.Decides,
          Game.prob (fun o => ¬ V.Holds θ F c (V.honest θ F c) a o) (V.game θ F c) (V.honest θ F c) :=
        prob_exists_finset_le _ _ _ _
    _ = 0 := Finset.sum_eq_zero fun a ha =>
        soundness_honest ι α d G hs hn a (Finset.mem_filter.1 ha).2 θ F hreg

end Verity.Toy

namespace Verity.Vocab

open Accounting

variable {ι α : Type} [Fintype α] {d : Device ι} (V : Vocab d α)

open Classical in
/-- **`HonestHolds .sampled` fails at a commitment that opens nothing**: at registered files and a commitment `c`
whose positions hold no gate that isn't a registered constant, every unit reading or writing such a gate is unbound
whatever the extractor returns, so where those units are more than an `ε` share, sampled proofs' claim fails on every
play of every strategy, the honest one's included. -/
theorem not_honestHolds_sampled_of_unopened (θ F c : ByteArray) (hreg : V.Registered θ F)
    (hplace : ∀ g, (V.read F).consts g = none → V.place θ F c g = none)
    (hlt : (V.certOf θ).ε * V.units F <
      ((Finset.univ.filter fun u => ∃ g ∈ (V.partsOf F).io u, (V.read F).consts g = none).card : ℝ)) :
    ¬ V.HonestHolds .sampled := by
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
  have h1 := prob_play_eq_one (V.rounds θ) θ F c (fun o => ¬ V.Holds θ F c (V.honest θ F c) .sampled o)
    (fun log e => hfail _ _) (V.honest θ F c)
  have h0 := h θ F c hreg
  exact one_ne_zero (h1.symm.trans h0)

end Verity.Vocab
