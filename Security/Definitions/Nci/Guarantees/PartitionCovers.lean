import Definitions.Core.Formalism
import Definitions.Nci.Covering

/-!
# Appendix A.1: from a `(c, k)`-partition to a covering

The working draft counts a proof block's operation gates and places its inputs `in(I)` outside it; a covering's
inference block (`Nci.Covering.InferenceBlock`) holds its inputs among its gates. Completing each proof block with its
inputs, `Î = I ∪ in(I)` with inputs `in(I)`, turns a `(c, k)`-partition into a covering at size `c + max_I |in(I)|` and
the same read cap `k`, with one inference block per proof block, each computing exactly the gates of its proof block.
-/

namespace Nci.Guarantees

open Verity Nci.Covering Finset

/-- **Lemma A.1 (boundary completion).** A `(c, k)`-partition into `B` proof blocks gives a
`(c + max_I |in(I)|, k)`-covering by `B` inference blocks, proof block `u`'s completion having the gates `I ∪ in(I)` and
the inputs `in(I)`. -/
def PartitionCovers : Prop :=
  ∀ (d : Declaration) (B c k : ℕ) (P : Partition d.D B), P.IsCK d c k →
    ∃ I : Fin B → Subcircuit d.D,
      Covers d (c + univ.sup fun u => (d.D.into (P.gates u)).card) k I ∧
        ∀ u, (I u).gates = P.gates u ∪ d.D.into (P.gates u) ∧ (I u).ins = d.D.into (P.gates u)

end Nci.Guarantees
