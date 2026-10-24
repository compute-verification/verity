import Definitions.Accounting.TS.Play

/-! The proof-only part of `Definitions.Accounting.TS.Play`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Accounting.TS

open Pouw.RelativeSecurity

namespace RoundGame

variable {Ω : Type} [Fintype Ω]

/-- The number of rounds with a positive share along the realized coins. -/
noncomputable def deficientRounds : {n : ℕ} → Pouw.RelativeSecurity.RoundGame Ω n → Paths Ω n → ℕ
  | _, .done, _ => 0
  | _, .round d _ next, x => (by classical exact if 0 < d x.1 then 1 else 0) + deficientRounds (next x.1) x.2

end RoundGame

end Accounting.TS
