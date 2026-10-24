import Definitions.NetworkCertifier.SampledAudit

/-!
# The sampled audit's charge: what the faulty units add to the leak

Definitions only. An accepted sampled audit gives `Accepted` by the circuit's Program at `δ`, with the declared advice
holding values at the faulty units' committed gates (`RAcc`, `Definitions.NetworkCertifier.SampledAudit`). Those values
are the prover's to choose, and so is which units are faulty, so the leak bound of a sampled audit charges, beyond the
inputs the circuit reads besides the ingress, the faulty units' committed bits (`faultyBits`) and the choice of faulty set
among the ones `RAcc δ` allows (`faultySets`). The sequences it counts are `AccSeqsSampled`.
-/

namespace NetTiming

open Claims.Sampling

/-- **The faulty units' committed bits**: over the slots `F` of the partition `P`, each replay unit's gates that its
boundary commits (`committed`). -/
def faultyBits (C : Prog) (P : Partition) (F : Finset ℕ) : ℕ :=
  ∑ v ∈ F, committed C (RU.gates (P.rus.getD v []))

open Classical in
/-- **The possible faulty sets at `δ`**: the sets of `P`'s slots that are in `M_δ` and hold no slot drawn `all`, the
faulty sets `RAcc δ` allows. At `δ ≥ miss(1)` none is non-empty. -/
noncomputable def faultySets (d : Design) (P : Partition) (δ : ℝ) : Finset (Finset ℕ) :=
  (Finset.range P.rus.length).powerset.filter fun F => F ∈ d.M P δ ∧ Disjoint F (allSlots d P)

/-- **Accepted with the faulty set `F`**: some declared advice `(a, free)` makes the first `k` steps `Accepted` by the
circuit's Program, with `a`, the inputs other than the ingress, among `As`, and `free` holding values only at the
committed gates of `F`'s units (`FaultyGate`). -/
def AcceptedFaulty {L K S : ℕ} (C : Prog) (P : Partition) (lay : Layout L S) (k : ℕ) (As : Set (ℕ → Bool))
    (F : Finset ℕ) (op : ℕ → List (Obs L K S) → List Bool) (o : List (Obs L K S)) : Prop :=
  ∃ a ∈ As, ∃ free : ℕ → Option Bool, (∀ g b, free g = some b → FaultyGate C P F g) ∧
    Accepted (circuitProgram C lay k) op k o (a, free)

/-- **The sequences a sampled audit accepts at `δ`**: the observation sequences of length `n`, with jitter below `D`,
accepted with some possible faulty set at `δ` (`AcceptedFaulty`). -/
def AccSeqsSampled {L K S : ℕ} (C : Prog) (P : Partition) (d : Design) (lay : Layout L S) (n : ℕ)
    (As : Set (ℕ → Bool)) (δ : ℝ) (op : ℕ → List (Obs L K S) → List Bool) (D : ℕ) : Set (List (Obs L K S)) :=
  {o | o.length = n ∧ (∀ x ∈ o, x.jitter < D) ∧ ∃ F ∈ faultySets d P δ, AcceptedFaulty C P lay n As F op o}

end NetTiming
