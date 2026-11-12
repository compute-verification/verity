module

public import Definitions.Core.Guarantees.Formalism
public import Definitions.Core.Formalism.Instances

/-!
# The size of a `(c, k)`-partitioned circuit and Theorem 3: honest and nontrivial lemmas

`CKSize` and `Counting` each have two statements here, at the instances of
`Security/Definitions/Core/Formalism/Instances.lean`.
`CKSize_honest`: its premise holds at `Honest`, a `(2, 1)`-partition with a proof block of fewer than `c` gates.
`CKSize_nontrivial`: at `Counter`, one proof block of three gates, the premise fails at `(c, k) = (2, 1)`, and so does
the bound, `|C| = 3 > 2 · 1`.
`Counting_honest`: at `Honest`, counting the operation gates at `weight` with the allowance `q = 2`, its premises hold
and its bound is attained: each proof block weighs `q`, so `(c / q) Σ ω = 4 = c · B`.
`Counting_nontrivial`: the same, at `heavy`, every premise but the allowance holds, the second proof block weighs `3`,
above `q`, and the bound fails, `(c / q) Σ ω = 5 > c · B = 4`.

The proofs are in `Security/Proofs/Core/Formalism/Witnesses.lean`.
-/

@[expose] public section

namespace Verity.Guarantees

/-- **`CKSize`'s premise holds** at `Honest`, a `(2, 1)`-partition whose second proof block has one gate. -/
def CKSize_honest : Prop := Formalism.Honest.IsCK Formalism.d 2 1

/-- **`CKSize` is false without its premise**: at `Counter`, one proof block of three gates, the partition is no
`(2, 1)`-partition, and the circuit's size is above `c · B = 2 · 1`. -/
def CKSize_nontrivial : Prop :=
  ¬ Formalism.Counter.IsCK Formalism.d 2 1 ∧ ¬ Formalism.d.D.size Formalism.d.D.computed ≤ 2 * 1

/-- **`Counting`'s premises hold and its bound is attained** at `Honest`, two proof blocks priced at `c = 2`, counting
the operation gates at `weight` with the allowance `q = 2`: each proof block weighs `2`, so `(c / q) Σ ω = 4 = c · B`,
and the slowdown bound is attained at every `T₀ > 0`. -/
def Counting_honest : Prop :=
  Formalism.counted ⊆ Formalism.d.D.computed ∧ (0 : ℝ) < 2 ∧
    (∀ u, ∑ v ∈ Formalism.counted ∩ Formalism.Honest.gates u, (Formalism.weight v : ℝ) ≤ 2) ∧
    (2 : ℝ) / 2 * ∑ v ∈ Formalism.counted, (Formalism.weight v : ℝ) = (2 * 2 : ℕ) ∧
    ∀ T₀ : ℝ, 0 < T₀ →
      (2 : ℝ) / (T₀ * 2) * ∑ v ∈ Formalism.counted, (Formalism.weight v : ℝ) = slowdown 2 2 T₀

/-- **`Counting` is false without its allowance**: at `Honest`, counting the operation gates at `heavy` with `q = 2`,
the counted gates are operation gates and `q > 0`, but the second proof block weighs `3`, above `q`, and
`(c / q) Σ ω = 5` is above `c · B = 4`. -/
def Counting_nontrivial : Prop :=
  Formalism.counted ⊆ Formalism.d.D.computed ∧ (0 : ℝ) < 2 ∧
    ¬ (∀ u, ∑ v ∈ Formalism.counted ∩ Formalism.Honest.gates u, (Formalism.heavy v : ℝ) ≤ 2) ∧
    ¬ (2 : ℝ) / 2 * ∑ v ∈ Formalism.counted, (Formalism.heavy v : ℝ) ≤ (2 * 2 : ℕ)

end Verity.Guarantees
