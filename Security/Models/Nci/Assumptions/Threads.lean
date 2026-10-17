import Definitions.Nci.Threads

/-!
# NCI over threads: the named assumption

The thread draft's one assumption besides `(d, e)`-threading, which is a property of the reference circuit's wiring
(`Threading.Threaded`), is that weights are incompressible. It is a statement about the reference training circuit `c`,
whose weights are computed after the commitment, not about the declared circuit: a declared circuit reaches it only
through containment.
-/

namespace Nci.Assumptions

open Verity

/-- **Weights are incompressible** (the thread draft, "Lower-bounding training's NCI"): every small subcircuit of the
reference circuit `c` (at most `Γ` computed gates, at most `S` bits of novel reads) that uses `p` distinct weights reads
at least `p · b` bits in addition to the `b` bits per novel thread value it reads:
`b · #(weights R uses) + b · Σ_t #(novel values of thread t that R reads) ≤ width(in(R) \ θ)`.
`b` is the width of a weight and of an activation.

It fails when a small subcircuit can rebuild weights it uses from less than their width: at the first training step,
where they are committed in `θ`; under a low-rank update (LoRA), whose factors are narrower than the weights they
update; or when the subcircuit computes the weights' updates itself from fewer bits of activations and gradients. -/
def WeightsIncompressible (c : Declaration) (T : Threading c) (Γ : ℕ) (S b : ℝ) : Prop :=
  ∀ R, Small c Γ S R →
    b * (T.weightsUsed R).card + b * ∑ t, (T.threadReads R t).card ≤ novelReads c R

end Nci.Assumptions
