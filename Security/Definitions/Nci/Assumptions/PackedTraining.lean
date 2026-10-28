import Definitions.Nci.Assumptions.Packed
import Definitions.Nci.Threads

/-!
# Training's incompressibility on a chain with several products to a gate

`TrainingIncompressible`, restated for a `PackedChain`, so that it applies to a reference circuit that computes in `Dot16`
steps. At `k = 1` it is `TrainingIncompressible` on the chain itself (`Chain.toPacked`).
-/

namespace Nci.Assumptions

open Verity

open Classical in
/-- **Training is incompressible, at most `k` products to a gate** (`TrainingIncompressible` on a `PackedChain`): in
every division `P` of the reference training circuit `c` into IUs, the partial sums passed between IUs can be charged to
the IUs that receive them, `s u` to IU `u`, with an entry of `ch` whose products' gates lie in `m` IUs charged at least
`m − 1` in total, so that every small IU reads at least `b_w` bits for each weight entry of `ch` it uses and `b_a` for
each input entry it uses or partial sum charged to it, whichever count is larger, and `b_w` bits for each weight of the
threading `T` it uses in addition to the `b_a` bits for each novel thread value it reads. "Uses" means that the IU holds
the gate of a product of the entry.

On an honest `Dot16` circuit the partial sums are the accumulators: the step gates of one entry form a sequence, each
reading the one before, so an entry whose step gates lie in `m` IUs passes its accumulator between IUs at least `m − 1`
times, each read by the IU it enters.

It is adopted, with `PackedActivationsIncompressible`, as the `k`-generalization of the per-product premises
`ActivationsIncompressible` and `TrainingIncompressible`.
It covers a reference training circuit with at most `k` of the chain's products to a gate, such as one in `Dot16` steps,
whose weights and first inputs depend on more than `Γ` work of training since the commitment, as
`TrainingIncompressible` does, and not a declared circuit, whose wiring the prover controls and where
its activations' half is false. -/
def PackedTrainingIncompressible {k : ℕ} (c : RegisteredCircuit) (ch : PackedChain c.D k) (T : Threading c) (Γ : ℕ)
    (K b_w b_a : ℝ) : Prop :=
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
