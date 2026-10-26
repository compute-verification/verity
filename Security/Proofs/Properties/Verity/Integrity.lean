import Properties.Verity.Game
import Proofs.Accounting.PouwWork
import Proofs.Pouw.PearlC.WholeCircuit

/-!
# The integrity leg as `Properties.Verity.Game` states it holds at every vocabulary

`Vocab.IntegrityLeg` asks, for every strategy, for a transcript `X` at which the sampled audit's claim (at most an `ε`
share of the parameters' circuit's units wrong in `X`) fails with probability at most its error. Nothing ties `X` to
what the commitment binds: it is chosen after the strategy, and the circuit's evaluation on any inputs has no wrong
unit (`Pouw.PearlC.WholeCircuit.correct_eval`). So the leg holds at every vocabulary whose published `ε` and error are
nonnegative (`integrityLeg_of_nonneg`), whatever the component accepts, and says nothing about the transcript the
prover committed.
-/

namespace Verity.Vocab

open Accounting Core

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] {I : Type} {Ω : I → Type} (V : Vocab d α Ω)

/-- The parameters' circuit's evaluation on any inputs has no wrong unit of the parameters' partition. -/
theorem wrong_eval (θ : ByteArray) (a : (V.circuitOf θ).Transcript) :
    (V.partsOf θ).wrong ((V.circuitOf θ).eval a) = ∅ := by
  classical
  exact Finset.filter_eq_empty_iff.2 fun u _ hn => hn (Pouw.PearlC.WholeCircuit.correct_eval _ a u)

/-- **`IntegrityLeg` holds at every vocabulary with a nonnegative `ε` and error**, at the transcript the parameters'
circuit evaluates to on zero inputs, whatever the strategy, the component's acceptance and the coins it reads. -/
theorem integrityLeg_of_nonneg [Fintype I] [DecidableEq I] [∀ i, Fintype (Ω i)]
    (hε : ∀ θ, 0 ≤ (V.certOf θ).ε) (hδ : ∀ θ, 0 ≤ V.delta θ .sampled) : V.IntegrityLeg := by
  intro τ
  refine ⟨fun _ => (V.circuitOf τ.params).eval fun _ => 0, fun ξ => ?_⟩
  refine le_trans (Accounting.SecurityProofs.PouwWork.pr_mono (F := fun _ => False) ?_) ?_
  · rintro x ⟨-, hn⟩
    apply hn
    show (((V.partsOf τ.params).wrong ((V.circuitOf τ.params).eval fun _ => 0)).card : ℝ) ≤
      (V.certOf τ.params).ε * V.units τ.params
    rw [V.wrong_eval]
    simpa using mul_nonneg (hε τ.params) (Nat.cast_nonneg (V.units τ.params))
  · simp [Pouw.pr, hδ]

end Verity.Vocab
