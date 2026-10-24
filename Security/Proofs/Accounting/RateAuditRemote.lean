import Definitions.Accounting.RateAuditRemote

/-! The proof-only part of `Definitions.Accounting.RateAuditRemote`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Accounting.Rate

open Pous (Bits)

namespace GrindChain

variable {P N b h ρ S L q r s G : ℕ}

/-- The chained prover that keeps candidate `c` at every pass. -/
def atCand (A : GrindChain P N b h ρ S L q r s G) (c : Fin G) : Chain P N b h ρ S L q r s where
  device j η := A.device j η c
  root j η E := A.cand j η E c

end GrindChain

end Accounting.Rate
