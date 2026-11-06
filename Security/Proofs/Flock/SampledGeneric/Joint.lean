module

public import Proofs.Flock.SampledGeneric.Compose

/-!
# The textbook form's hypotheses

An extractor that is a game after the judged session (`JointExtractor`) gives the generic theorem's hypotheses but the
transfer:

* `JointExtractor.toExtractor_covers`: acceptance at an incorrect sampled proof block either fails to extract or
  extracts a witness whose openings disagree with the trace (`JointExtractor.Sound`);
* `JointExtractor.toTrials_opens`: what the trials read comes with openings of the registered commitment;
* `JointExtractor.toTrials_wellFormed`: the trials' numbers are in range, with no slack;
* `Trials.plurality_plur`: the plurality is a plurality.
-/

@[expose] public section

namespace FlockSoundness.SampledGeneric

open Game Audit LinkFinder
open scoped ENNReal

variable {n : ℕ} {Reg : Type} {Dh : Type} {Hc : List UInt8 → Dh} {B : Binding Reg Hc} {PS : ProofSystem n Reg}

namespace JointExtractor

variable (J : JointExtractor PS B) {Correct : Reg → B.Trace → Fin n → Prop}

theorem toExtractor_covers (hJ : J.Judged) (hS : J.Sound Correct) : J.toExtractor.Covers Correct := by
  intro S R τ u X hu hX
  rw [← hJ S R τ]
  refine (prob_mono (q := fun o => (o.1 = true ∧ J.Fails S R u o) ∨ (o.1 = true ∧ ∃ w, o.2 u = some w ∧
      J.Rel S R u w ∧ ∃ c ∈ J.reads u, B.val (J.opens S R u w c) ≠ X c)) (fun o h => ?_) _ _).trans
    (prob_or_le _ _ _ _)
  by_cases hf : J.Fails S R u o
  · exact Or.inl ⟨h, hf⟩
  · obtain ⟨w, hw, hR⟩ := not_not.1 hf
    refine Or.inr ⟨h, w, hw, hR, ?_⟩
    by_contra hc
    exact hX ((hS S R u w hu hR).2 X fun c hc' => by_contra fun hne => hc ⟨c, hc', hne⟩)

open Classical in
theorem mem_readsOut {L : Law n} {R : Reg} {c : B.Pos} {e : L.Ω × (Bool × (Fin n → Option J.Wit))} {v : B.Val} :
    v ∈ J.readsOut L R c e ↔ e.2.1 = true ∧ ∃ u ∈ L.draw e.1, c ∈ J.reads u ∧
      ∃ w, e.2.2 u = some w ∧ J.Rel (L.draw e.1) R u w ∧ B.val (J.opens (L.draw e.1) R u w c) = v := by
  unfold readsOut
  exact Finset.mem_filter.trans (and_iff_right (Finset.mem_univ v))

theorem toTrials_opens (hS : J.Sound Correct) (L : Law n) : (J.toTrials L).Opens := by
  intro R τ c e v hv
  obtain ⟨-, u, hu, hc, w, hw, hR, hval⟩ := (J.mem_readsOut (L := L)).1 hv
  obtain ⟨u', hu', hc', w', -, hR', hop, hval'⟩ := Classical.epsilon_spec
    (p := fun o => ∃ u ∈ L.draw e.1, c ∈ J.reads u ∧ ∃ w, e.2.2 u = some w ∧ J.Rel (L.draw e.1) R u w ∧
      o = J.opens (L.draw e.1) R u w c ∧ B.val o = v)
    ⟨J.opens (L.draw e.1) R u w c, u, hu, hc, w, hw, hR, rfl, hval⟩
  have hop' : (J.toTrials L).op R τ c e v = J.opens (L.draw e.1) R u' w' c := hop
  refine ⟨?_, hval'⟩
  rw [hop']
  exact (hS _ R u' w' hu' hR').1 c hc'

theorem toTrials_wellFormed (hc : ∀ S R o, 0 ≤ J.cost S R o) (L : Law n) : (J.toTrials L).WellFormed :=
  ⟨fun R _ e => hc _ R e.2, fun _ => div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _),
    show (0 : ℝ) < 1 from one_pos, show (0 : ℝ) ≤ 0 from le_rfl⟩

end JointExtractor

namespace Trials

variable {L : Law n} (Tr : Trials L PS B)

theorem plurality_plur : Tr.Plurality Tr.plur := by
  intro R τ c h v
  simp only [plur, dite_eq_left h]
  exact (Classical.choose_spec (Finset.exists_max_image Finset.univ
    (pRd (Tr.T R τ c h) (Tr.sT R τ c h) (Tr.rd R τ c)) Finset.univ_nonempty)).2 v (Finset.mem_univ v)

end Trials

end FlockSoundness.SampledGeneric
