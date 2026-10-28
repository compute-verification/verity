import Definitions.Nci.Threads

/-!
# NCI's one assumption about training

The guarantees on a reference training circuit `c` (`Nci.Guarantees.ThreadedThrottled`,
`Nci.Guarantees.ContainedTraining`) charge an IU of `c` for three kinds of values it does not compute itself: the weights
it uses, the activations it uses, and the partial sums it receives. `TrainingIncompressible` states all three as one law
over every division of `c`, in reads: each small IU reads at least what it is charged. It is equivalent to the
conjunction of `WeightsIncompressible`, `ActivationsIncompressible` and `PartialSumsIncompressible` on `c`
(`Nci.SecurityProofs.trainingIncompressible_iff`), and the support-form assumptions imply it
(`Nci.SecurityProofs.trainingIncompressible_of_support`).

It is NCI's one incompressibility premise; the older assumptions stay beside it until the
guarantees that read them move to it. It is about the reference circuit, not a declared one: `ActivationsIncompressible`
and `PartialSumsIncompressible` are false on a declared circuit, whose wiring the prover controls, and that does not
reach it. The reference circuit runs the chain gate for gate, one gate
per product, with ordinary matrix multiplications and no Strassen-like algorithm; that is a stated assumption, and in
Lean it is the `Chain` the guarantees take. The packed forms relax it to at most `k`
products to a gate, as in `Dot16` steps (`PackedChain`, `PackedTrainingIncompressible`).

The chain guarantees restated on it (`Nci.Guarantees.IncompressibleContainedTraining`) conclude what the support forms
do from a weaker premise, since the support forms with `WeightsIncompressible` imply it; the thread guarantees read only
its weights' half. With a partial containment, the restated chain guarantee
(`Nci.Guarantees.IncompressibleContainedMatMulOn`) bounds what the dark gates absorb by their novel reads, where the
support form bounds it by their width.
-/

namespace Nci.Assumptions

open Verity

open Classical in
/-- **Training is incompressible**: in every division `P` of the reference training circuit `c` into IUs, the partial
sums passed between IUs can be charged to the IUs that receive them, `s u` to IU `u`, with an entry of the chain `ch`
whose products lie in `m` IUs charged at least `m − 1` in total, so that every small IU (at most `Γ` computed gates, at
most `K` bits of novel reads) reads at least
1. `b_w` bits for each weight entry of `ch` it uses, and `b_a` for each input entry it uses or partial sum charged to it,
   whichever count is larger, where an input entry linked to an entry whole in the IU is not counted, since the IU
   computes it from its own products; and
2. `b_w` bits for each weight of the threading `T` it uses, in addition to the `b_a` bits for each novel thread value it
   reads.

IUs that are not small are charged nothing, so a single small IU is a division of `c` into it and the rest.

`b_w` and `b_a` are rates: at most the widths of the wires that carry weights, activations, thread values and partial
sums in `c`, and below them by what those values compress to.

Rationale (the draft's): an IU must read each weight entry it uses, or compute it, and computing an updated weight takes
more than `Γ` work of training since the commitment; it must read each input entry it uses, its linked output entry, or a
partial sum of that entry, distinct for distinct entries since the links are injective; and an entry split over `m` IUs
needs at least `m − 1` partial sums read across IUs, one by each IU's first exit on the way to the entry, with each read
paid by its reader. The two charges in (1) are not added, since an IU that completes an entry from partial sums it
receives pays for the linked input entry with the same reads.

It is meant for honest training circuits whose weights, and every input of the chain's first multiplication, depend on
more than `Γ` work of training since the commitment, with no low-rank update (LoRA) whose factors are narrower than the
weights they update. Additions may have any fan-in. It is stated in reads, not support: the gates of an IU take
fewer distinct values than its reads, and in support form the partial-sum charge is false for one honest matrix
multiplication, since the partial sums over a `k × k × k` block are determined by `2k²` entries, not by the `k³`
products an IU must read to form them. -/
def TrainingIncompressible (c : RegisteredCircuit) (ch : Chain c.D) (T : Threading c) (Γ : ℕ) (K b_w b_a : ℝ) : Prop :=
  ∀ (n : ℕ) (P : Partition c.D n), ∃ s : Fin n → ℕ,
    ∑ x : (ℓ : Fin ch.L) × Fin ch.N × Fin (ch.e ℓ), (ch.spread P x.1 x.2.1 x.2.2 - 1) ≤ ∑ u, s u ∧
    ∀ u, Small c Γ K (P.gates u) →
      b_w * (ch.weightsUsed (P.gates u)).card +
          b_a * ((max ((ch.inputsUsed (P.gates u)).filter
            fun x => ¬ ch.LinkedWhole (P.gates u) x.1 x.2.1 x.2.2).card (s u) : ℕ) : ℝ)
        ≤ novelReads c (P.gates u) ∧
      b_w * (T.weightsUsed (P.gates u)).card + b_a * ∑ t, (T.threadReads (P.gates u) t).card
        ≤ novelReads c (P.gates u)

end Nci.Assumptions
