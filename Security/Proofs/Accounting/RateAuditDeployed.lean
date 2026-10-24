import Definitions.Accounting.RateAuditDeployed

/-! The proof-only part of `Definitions.Accounting.RateAuditDeployed`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Accounting.Rate

open Pous (Bits)

namespace PIDevice

variable {N b h ρ S L q : ℕ}

/-- The bits the device completed in the pass. -/
def completed (A : PIDevice N b h ρ S L q) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) : ℕ :=
  (A.chip (A.state E)).completed (A.world E R O) []

/-- The compute the device's completions cost in the pass, at `cost x` per completion of `x` bits. -/
def work (A : PIDevice N b h ρ S L q) (cost : ℕ → ℕ) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) : ℕ :=
  (A.chip (A.state E)).work cost (A.world E R O) []

end PIDevice

end Accounting.Rate
