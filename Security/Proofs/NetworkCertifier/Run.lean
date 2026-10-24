import Definitions.NetworkCertifier.Run

/-! The proof-only part of `Definitions.NetworkCertifier.Run`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace NetTiming

namespace Interaction

variable {L K S : ℕ} {Z In Act Wire : Type} (I : Interaction L K S Z In Act Wire)

/-- The sender's actions over the run's first `n` steps. -/
def actRun (z : Z) (n : ℕ) : List Act := (I.trace z n).2.2.1

end Interaction

end NetTiming
