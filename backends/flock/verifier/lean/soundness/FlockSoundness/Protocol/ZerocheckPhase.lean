import FlockSoundness.Protocol.LincheckPhase
import FlockSoundness.Protocol.ListSize
import FlockSoundness.Protocol.OpeningPhase

/-!
# The zerocheck phase (PROTOCOL.md §9.2)

For one codeword near the committed table whose witness makes the region claims true, the doomed
start says the block R1CS or the pin fails. The zerocheck then ends with the `a`, `b`, `c` claims
and the pin's extension all true with small probability:

* **R1CS.** A nonzero error bit at skip index `s₀` stays visible after the eq-reduction over the
  other address bits (`m − 13` random coordinates: the pinned coordinates' weights are independent
  over GF(2)). `Pcomb` vanishes on the skip nodes by construction, so either the prover's `P_c` is
  wrong at `z` or its `P_ab` is (degree 127). The residual rounds keep a false claim false
  (`2/|F|` each, given every `r_eq ≠ 1`, one event for all codewords), and a false final claim
  makes the `a` or the `b` claim false.
* **Pin.** A block whose pin bit is 0 makes the pin's extension at `ρ_out` a nonzero polynomial of
  degree `m − k_log` in the rounds' coins.

`zerocheck_phase_of_card` takes the union over a list of at most `L` codewords.
-/

namespace FlockSoundness

open Game Model Accounting Polynomial
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-! ## Averages and coins -/

/-! ## Folding the lowest bit, and the equality weights -/

/-- Fold the lowest bit at `t`: `u'(y) = u(2y) + t·(u(2y+1) + u(2y))`. -/
def foldLo (t : F) (u : ℕ → F) : ℕ → F := fun y => u (2 * y) + t * (u (2 * y + 1) + u (2 * y))

/-! ## The residual rounds (Z6) -/

/-! ## The nodes, and polynomials through them -/

/-! ## The witness by skip index -/

/-! ## The true polynomials of the zerocheck -/

/-! ## The three coin bounds -/

/-! ## The game -/

/-! ## The phase -/

end FlockSoundness
