module

public import Proofs.Core.PrivatePartition

/-! # A root opened to another design: equal designs, or a collision of the hash -/

@[expose] public section

namespace Verity.SecurityProofs

open Verity

/-- **A prover who declares one design and opens the root to another holds a collision.** For every function `design`
of the advice, two commitments `r₀` and `r` under one advice root give one design, or `H` collides between a string
`r₀` hashes on its way to the root and a different one `r` hashes there. The premises are the format facts
`CommittedMeets` takes. -/
theorem openedDesign_eq_or_collides {α : Type} (design : List (List Bool) → α) (H : List UInt8 → Digest)
    (F : Advice.Framing) (hF : F.Separated) (A : Advice.Rows) (hA : A.Decodes) (r₀ r : Advice.Committed)
    (h : r.root H F A = r₀.root H F A) :
    design r.a = design r₀.a ∨ ∃ x ∈ r₀.hashInputs H F A, ∃ y ∈ r.hashInputs H F A, x ≠ y ∧ H x = H y := by
  rcases Advice.root_bind hF h.symm with he | hc
  · refine Or.inl (congrArg design (Option.some_inj.mp ?_))
    rw [← hA r.a, ← hA r₀.a, he]
  · exact Or.inr hc

end Verity.SecurityProofs
