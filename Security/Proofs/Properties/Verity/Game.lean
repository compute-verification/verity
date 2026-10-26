import Properties.Verity.Game
import Proofs.Properties.Verity.Soundness

/-!
# Soundness as a commit-then-coin game, from its legs

`Verity.Vocab.soundness` proves `Vocab.Soundness` from the sampled component's leg (`IntegrityLeg`, whose transcript
the commitment fixes) and every other audit's (`Leg`). An accepted play outside `Break` has every component accepting
(`accepts_components`) on the coins the device logged for it, so the event lies inside `JointPeriod`'s for the run at
the committed transcript, as in `soundness_of_legs`.
-/

namespace Verity.Vocab

open Accounting Core

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] {I : Type} [Fintype I] [DecidableEq I]
  {Ω : I → Type} [∀ i, Fintype (Ω i)] [∀ i, Nonempty (Ω i)] (V : Vocab d α Ω)

theorem soundness (hI : V.IntegrityLeg) (hL : ∀ a, a ≠ .sampled → V.Leg a) : V.Soundness := by
  intro τ
  obtain ⟨X, hX⟩ := hI τ
  refine ⟨X, le_trans (Accounting.SecurityProofs.PouwWork.pr_mono ?_)
    (Accounting.SecurityProofs.JointPeriod ι α I Ω d (V.circuitOf τ.params) (V.run τ (X τ.commitment))
      (V.certOf τ.params) (V.priceOf τ.params) (V.delta τ.params) ?_)⟩
  · rintro ξ ⟨hacc, ⟨a, hfail⟩, hbrk⟩
    refine ⟨a, ?_, hfail⟩
    show ComponentAccepts a (τ.sent ξ) ∧ V.coinsRead a (τ.sent ξ) = some (ξ (V.own a))
    refine ⟨accepts_components hacc a, ?_⟩
    by_contra hne
    exact hbrk ⟨a, accepts_components hacc a, hne⟩
  · intro a
    by_cases ha : a = .sampled
    · subst ha
      exact hX
    · exact hL a ha τ (X τ.commitment)

end Verity.Vocab
