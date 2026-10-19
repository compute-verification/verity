import Definitions.Nci.Threads
import Definitions.Nci.Fit

/-!
# NCI over threads: the named assumption

The thread draft's one assumption besides `(d, e)`-threading, which is a property of the reference circuit's wiring
(`Threading.Threaded`), is that weights are incompressible. It is a statement about the reference training circuit `c`,
whose weights are computed after the commitment, not about the declared circuit: a declared circuit reaches it only
through containment. `WeightsIncompressibleSlots` is the same assumption with weights charged per parameter slot
(`Threading.Slots`), once per parameter however many of its optimizer steps a subcircuit spans.
-/

namespace Nci.Assumptions

open Verity

/-- **Weights are incompressible** (the thread draft, "Lower-bounding training's NCI"): every small subcircuit of the
reference circuit `c` (at most `Γ` computed gates, at most `S` bits of novel reads) that uses `p` distinct weights reads
at least `b_w` bits for each of them in addition to `b_a` bits for each novel thread value it reads:
`b_w · #(weights R uses) + b_a · Σ_t #(novel values of thread t that R reads) ≤ width(in(R) \ θ)`.
`b_w` and `b_a` are rates, in bits of novel information per weight used and per activation read: at most the widths of
a weight and an activation, and below them by whatever the updated weights and the activations compress to. The draft
takes both equal to the width `b`.

It fails when a small subcircuit can rebuild weights it uses from less than their width: at the first training step,
where they are committed in `θ`; under a low-rank update (LoRA), whose factors are narrower than the weights they
update; or when the subcircuit computes the weights' updates itself from fewer bits of activations and gradients. -/
def WeightsIncompressible (c : Declaration) (T : Threading c) (Γ : ℕ) (S b_w b_a : ℝ) : Prop :=
  ∀ R, Small c Γ S R →
    b_w * (T.weightsUsed R).card + b_a * ∑ t, (T.threadReads R t).card ≤ novelReads c R

/-- **Weights are incompressible per parameter slot**: every small subcircuit of the reference circuit `c` (at most `Γ`
computed gates, at most `S` bits of novel reads) that uses weights of `p` parameter slots of `σ` reads at least `b_w`
bits for each of those slots in addition to `b_a` bits for each novel thread value it reads:
`b_w · #(slots R uses) + b_a · Σ_t #(novel values of thread t that R reads) ≤ width(in(R) \ θ)`.

It charges each parameter once, however many of its versions `R` uses. Each optimizer step's updated weights are new
gates, the slot map sending each to the parameter it is a version of, so a subcircuit that spans `k` steps on a
parameter uses `k` of its weight gates. It reads the parameter once, in the first version it uses, at most its width
(16 bits at BF16), and computes the later versions itself from that value and gradients, which it pays for as thread
values it reads. `WeightsIncompressible` charges it `k b_w` for that parameter, more than the width once
`k > 16 / 2.5 = 6.4` at `b_w = 2.5`, a difference only the subcircuit's other reads can pay; this charges `b_w`, which
is at most the width whatever `k` is. With one version per slot (`σ.slot` one-to-one on the weight gates) it is
`WeightsIncompressible`. A coarser map charges less, and is a `Threading.Slots` only when each thread's counted
multiplications on all versions of a slot number at most `T.reuse` (`σ.reuse_le`), the reuse the bounds read.

It fails when a small subcircuit rebuilds the first version it uses of a parameter from fewer than `b_w` bits: in the
first steps after a committed checkpoint, whose weights are in `θ` and read for free, until the updates since it depend
on more than a small subcircuit computes; under a low-rank update (LoRA), whose factors are narrower than the weights
they update; or when it computes that version itself from fewer bits of activations and gradients. -/
def WeightsIncompressibleSlots (c : Declaration) (T : Threading c) (σ : T.Slots) (Γ : ℕ) (S b_w b_a : ℝ) : Prop :=
  ∀ R, Small c Γ S R →
    b_w * (σ.slotsUsed R).card + b_a * ∑ t, (T.threadReads R t).card ≤ novelReads c R

end Nci.Assumptions
