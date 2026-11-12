module

public import Definitions.Nci.Assumptions.Baseline
public import Mathlib.Analysis.SpecialFunctions.Pow.Real
public import Mathlib.Data.FinEnum

/-!
# The inference-only policy's named assumptions

The parameters the guarantees are taken at, named: what a run's coverage and a division's sessions must be for a
guarantee to apply. Each is checkable on the circuit it reads, and none is about the world.

Every bound through a containment, total or partial (`PartialContainment`), also rests on `ThetaRegistered`
(`Definitions/Nci/Assumptions/Baseline.lean`), a premise about what the protocol lets the registered
information `θ` hold rather than about any circuit.
-/

@[expose] public section

namespace Nci.Assumptions

open Verity

/-- **Coverage** (the draft's `κ(T)`): the counted work `x` is at least a fraction `κ` of the reference circuit `r`'s
computed gates, `κ · size(r) ≤ x`. A threading's counted multiplications count `x = 2 |M|`, and the cover theorem's
certified products `x = |P|`. A training run whose counted work is a small share of its gates is slowed in proportion to
that share. -/
def Coverage (κ : ℝ) (r : RegisteredCircuit) (x : ℝ) : Prop :=
  κ * r.D.size r.D.computed ≤ x

/-- **Sessions of exactly `c` gates, each reading its own prompt of at most `k` bits**: each block `u` of the division
`P` of `d` has exactly `c` gates, padding included, and reads from outside itself only registered information and its
prompt `prompt u`, at most `k` bits wide. It is the layout an inference deployment declares, a session per block. -/
def Sessions (d : RegisteredCircuit) {B : ℕ} (P : Partition d.D B) (prompt : Fin B → Finset (Fin d.D.N)) (c k : ℕ) : Prop :=
  (∀ u, (P.gates u).card = c) ∧ (∀ u, d.D.into (P.gates u) ⊆ d.θ ∪ prompt u) ∧ (∀ u, d.D.width (prompt u) ≤ k)

end Nci.Assumptions
