import Proofs.Pouw.Fp8Atom.H1TFlipProofs
import Proofs.Pouw.Fp8Atom.H1TGammaProofs
import Models.Pouw.Assumptions.Fp8Atom.H1TRunningAssumptions

/-!
# `DistinctLiveH1T` from the one-flip clauses and `H1TRunningWords`

A credited word reads only its own row's salt (`word_row`). So two equal words of different rows are both constant,
and within a row the pairs split into those `H1TOneFlip` separates on half of the salts, and the `Hard` ones, which
`H1TRunningWords` separates. A word is not constant by `H1TOneFlip` (not `Stuck`) or `H1TRunningWords` (`Stuck`), and
not a raw XOF word by `h1tNotRawWord`. Conversely `DistinctLiveH1T` gives `H1TRunningWords` at once, so the two are
equivalent, and `h1tGamma` holds under `H1TRunningWords`.
-/

namespace Pouw.Fp8Atom.H1T.Proofs

open Pouw.Fp8Atom Pouw.Fp8Atom.H1T

theorem word_row (U : Unit) (c : Fin U.m × Fin U.n × Pos U.T) {g g' : U.Salt} (h : g c.1 = g' c.1) :
    U.word c g = U.word c g' := by
  have : U.rowSalt g c.1 = U.rowSalt g' c.1 := by funext τ; unfold Unit.rowSalt; rw [h]
  unfold Unit.word; rw [this]

theorem exists_of_half {α : Type} [Fintype α] [Nonempty α] (P : α → Prop) [DecidablePred P]
    (h : 2 * (Finset.univ.filter P).card ≤ Fintype.card α) : ∃ a, ¬ P a := by
  by_contra hc
  simp only [not_exists, not_not] at hc
  rw [Finset.filter_true_of_mem (fun a _ => hc a), Finset.card_univ] at h
  have := Fintype.card_pos (α := α)
  omega

theorem h1tDistinctLive (hR : H1TRunningWords) : DistinctLiveH1T := by
  intro U hU
  have hF := h1tOneFlip U hU
  have : Nonempty (Fin U.T → RawSlice) := ⟨fun _ => rawZero⟩
  have hnc : ∀ (i : Fin U.m) (a : Fin U.n × Pos U.T) (q : ℚ), U.word (i, a) ≠ fun _ => q := by
    intro i a q e
    by_cases hr : Stuck a.1.val a.2
    · exact (hR U hU i a a).2 hr q e
    · obtain ⟨s, hs⟩ := exists_of_half _ ((hF i).2 a hr q)
      exact hs (congrFun e _)
  have hsep : ∀ (i : Fin U.m) (a a' : Fin U.n × Pos U.T), a ≠ a' → U.word (i, a) ≠ U.word (i, a') := by
    intro i a a' hne e
    by_cases hH : Hard a.1.val a'.1.val a.2 a'.2
    · exact (hR U hU i a a').1 hne hH e
    · obtain ⟨s, hs⟩ := exists_of_half _ ((hF i).1 a a' hne hH)
      exact hs (congrFun e _)
  refine ⟨fun c c' e => ?_, fun c hc => ?_⟩
  · obtain ⟨i, a⟩ := c
    obtain ⟨i', a'⟩ := c'
    by_cases hi : i = i'
    · subst hi
      by_contra hne
      exact hsep i a a' (fun h => hne (by rw [h])) e
    · exfalso
      let g0 : U.Salt := fun _ _ => rawZero
      obtain ⟨g, hg⟩ : ∃ g, U.word (i, a) g ≠ U.word (i, a) g0 := by
        by_contra hc
        simp only [not_exists, not_not] at hc
        exact hnc i a (U.word (i, a) g0) (funext hc)
      let g1 : U.Salt := Function.update g0 i (g i)
      have h1 : U.word (i, a) g1 = U.word (i, a) g := word_row U (i, a) (by simp [g1])
      have h2 : U.word (i', a') g1 = U.word (i', a') g0 :=
        word_row U (i', a') (by simp [g1, Function.update_of_ne (Ne.symm hi)])
      apply hg
      rw [← h1, e, h2]
  · rcases hc with ⟨q, hq⟩ | ⟨i, τ, ℓ, hr⟩
    · exact hnc c.1 c.2 q hq
    · exact h1tNotRawWord U c i τ ℓ hr

theorem h1tRunningOfDistinct (hD : DistinctLiveH1T) : H1TRunningWords := by
  intro U hU i a a'
  obtain ⟨hinj, hfree⟩ := hD U hU
  exact ⟨fun hne _ e => hne (congrArg Prod.snd (hinj e)), fun _ q e => hfree (i, a) (Or.inl ⟨q, e⟩)⟩

theorem h1tGammaRunning (hR : H1TRunningWords) : H1TGamma := h1tGamma (h1tDistinctLive hR)

end Pouw.Fp8Atom.H1T.Proofs
