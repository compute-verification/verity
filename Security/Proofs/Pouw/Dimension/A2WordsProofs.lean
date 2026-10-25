import Models.Pouw.Assumptions.Dimension.A2Words
import Proofs.Pouw.Dimension.PipeProofs

/-!
# A2 on the checked words: proofs

* `a2ExtraOutputs`: an extra output outside the target set is never a target, so the matching program still works.
* `a2WitnessExtra`: `a2WitnessQuad` with every non-word added; the checked words are finitely many and a program's
  slots are finitely many, while the functions of the noise are infinitely many.
* `emptyQuadPipe`, `a2Separation`: the red team's `red-team-phase12-separation.lean`. The empty program gives
  `extraMachine` a program, at which the unrestricted A2 fails by `a2WitnessExtra`.
-/

namespace Pouw.Dimension.A2Words

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.Dimension Pouw.Dimension.Assumptions

theorem a2ExtraOutputs : A2ExtraOutputs := by
  intro N m n k M Alg W X γ₂ hX hA2 q
  obtain ⟨L, P, hL, hP, hout, hc⟩ := hA2 q
  refine ⟨L, P, hL, hP, fun φ hφ hW => ?_, hc⟩
  rcases hφ with hφ | hφ
  · exact hout φ hφ hW
  · exact absurd hW (Set.disjoint_left.1 hX hφ)

theorem wordSet_finite {N m n k : ℕ} (w : ℕ → ℕ → Noise N m n k → ℚ) (U : Finset ℕ) :
    (wordSet w U).Finite := by
  refine ((U ×ˢ range (nonFinal k m n) : Finset (ℕ × ℕ)).finite_toSet.image
    (fun x => w x.1 x.2)).subset ?_
  rintro φ ⟨u, hu, p, hp, rfl⟩
  exact ⟨(u, p), by simp [hu, hp], rfl⟩

theorem a2WitnessExtra : A2WitnessExtra := by
  intro k m n N R A B U cs
  refine ⟨a2ExtraOutputs N m n k _ _ _ _ 0 disjoint_compl_left (a2WitnessQuad N m n k cs _), ?_⟩
  rintro q ⟨L, P, hall⟩
  have hS : {φ | φ ∈ slots P L}.Finite :=
    (Multiset.finite_toSet (slots P L : Multiset (Noise N m n k → ℚ))).subset fun φ h => Multiset.mem_coe.2 h
  have huniv : (Set.univ : Set (Noise N m n k → ℚ)) ⊆ wordSet (word k m n N R A B) U ∪ {φ | φ ∈ slots P L} := by
    intro φ _
    by_cases hφ : φ ∈ wordSet (word k m n N R A B) U
    · exact Or.inl hφ
    · exact Or.inr (hall φ (Or.inr hφ))
  exact Set.infinite_univ (((wordSet_finite _ U).union hS).subset huniv)

/-- **The empty program is a quadratic closed-pipe program.** -/
theorem emptyQuadPipe : EmptyQuadPipe := fun _ _ _ _ _ =>
  ⟨⟨by simp, by simp, by simp [prodOuts, IsPipe]⟩, by simp [prodOuts, IsQuadPipe]⟩

/-- **The two forms of A2 separate.** -/
theorem a2Separation : A2Separation := by
  intro k m n N R A B U cs
  refine ⟨(a2WitnessExtra k m n N R A B U cs).1, fun γ₂ Alg h => ?_⟩
  let q : (extraMachine N m n k cs (wordSet (word k m n N R A B) U)).Prog :=
    (⟨[], emptyQuadPipe N m n k cs⟩ : {P : List (Instr (Noise N m n k)) // QuadPipe cs P})
  obtain ⟨L, P, -, -, hall, -⟩ := h q
  exact (a2WitnessExtra k m n N R A B U cs).2 q ⟨L, P, fun φ hφ => hall φ hφ trivial⟩

end Pouw.Dimension.A2Words
