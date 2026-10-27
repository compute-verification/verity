import Definitions.Core.Boolean

/-!
# Size on the Boolean profile (trusted)

A circuit's or a region's size is its gate count, the one measure of how much it computes (the Glossary's Size, Daniel,
26 Oct). Which gates count is one named parameter, `Measure`, a weight on each Boolean gate by its operation, so a
statement in size holds at every measure and a ruling picks one. Two are named:

* `Measure.gates`: every computed gate counts one and an input gate none, the Glossary's gate count, which
  `Verity.Circuit.size` of a set of computed gates is;
* `Measure.ands`: an AND or a row counts one, and XOR, NOT and an input gate none: the multiplications C-Flock's prover
  pays for (a row is the AND of two XOR sums), PROTO-DESIGN-4's recommendation. Daniel hasn't ruled between them.

`Circuit.sizeAt B μ S` is the size at `μ` of a set `S` of the Boolean circuit `B`'s gates.
-/

namespace Verity.Boolean

/-- **A size measure on the Boolean profile**: each gate's weight in a size, by its operation. -/
structure Measure where
  weight : {N : ℕ} → Op N → ℕ

namespace Measure

/-- **The gate count**: every computed gate counts one, an input gate none. -/
def gates : Measure where
  weight
    | .input => 0
    | _ => 1

/-- **The AND count**: an AND or a row counts one, and XOR, NOT and an input gate none, the multiplications C-Flock's
prover pays for. -/
def ands : Measure where
  weight
    | .and _ _ => 1
    | .row _ _ => 1
    | _ => 0

end Measure

/-- **The size at `μ` of a set `S` of the circuit's gates**: the sum of its gates' weights. -/
def Circuit.sizeAt (B : Circuit) (μ : Measure) (S : Finset (Fin B.N)) : ℕ := ∑ g ∈ S, μ.weight (B.op g)

end Verity.Boolean
