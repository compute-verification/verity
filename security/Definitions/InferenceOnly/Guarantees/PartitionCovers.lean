module

public import Definitions.Core.Formalism
public import Definitions.Slowdown.Covering

/-! Appendix A.1: a `(c, k)`-partition as a covering, each proof block completed with its inputs. -/

@[expose] public section

namespace InferenceOnly.Guarantees

open Verity Slowdown.Covering Finset

/-- **Lemma A.1 (boundary completion).** A `(c, k)`-partition into `B` proof blocks gives a
`(c + max_I |in(I)|, k)`-covering by `B` inference blocks, proof block `u`'s completion having the gates `I ∪ in(I)` and
the inputs `in(I)`. -/
def PartitionCovers : Prop :=
  ∀ (d : RegisteredCircuit) (B c k : ℕ) (P : Partition d.D B), P.IsCK d c k →
    ∃ I : Fin B → Subcircuit d.D,
      Covers d (c + univ.sup fun u => (d.D.into (P.gates u)).card) k I ∧
        ∀ u, (I u).gates = P.gates u ∪ d.D.into (P.gates u) ∧ (I u).ins = d.D.into (P.gates u)

end InferenceOnly.Guarantees
