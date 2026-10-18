import Models.Nci.Assumptions

/-!
# NCI's partial-sum law on a reference circuit

`UpdatedMatMul` states its incompressibility assumptions about the declared circuit, whose wiring the prover controls,
and they are false there. These guarantees state the same law with the chain and both assumptions on a reference
training circuit `c`, which the declared circuit reaches through containment, as the thread guarantees do
(`Nci.Guarantees.ThreadedThrottled`), and in support form (`ActivationsSupport`, `PartialSumsSupport`), so that they say
what `c`'s values determine rather than how it is wired. The assumptions are then claims about honest training: a small
piece of `c` holds at least `b_w` bits per updated weight entry it uses and `b_a` bits per activation entry or partial
sum it does not compute itself. `b_w` and `b_a` are rates, below the entries' widths by whatever they compress to, so
these guarantees are parameterized by how compressible training is.
-/

namespace Nci.Guarantees

open Verity

/-- **A reference circuit's updated matrix multiplications are throttled**: a declaration `d` that contains `c` and
complies at `(Γ, α)` has at least `2 F` gates per product of a chain `ch` that `c` runs, when `c`'s weights,
activations and partial sums are incompressible at `b_w` and `b_a` bits per entry, with `c`'s baseline at its committed
values `θv`. `dd` is the chain's smallest inner dimension. -/
def ContainedMatMul : Prop :=
  ∀ (Γ : ℕ) (α b_w b_a : ℝ) (c d : Declaration) (θv : c.D.Transcript) (ch : Chain c.D) (dd : ℕ),
    0 < α → 0 < b_w → 0 < b_a → (∀ ℓ, dd ≤ ch.d ℓ) → c.Contains d → Complies Γ α d →
    Assumptions.ActivationsSupport c θv ch (α * Γ + b_w + b_a) b_w b_a →
    Assumptions.PartialSumsSupport c θv ch Γ (α * Γ) b_w b_a →
      F α (α * Γ) b_w b_a dd * (2 * ch.G) ≤ d.D.size d.D.computed

/-- **Training is slowed, by the partial-sum law**: when the chain's products, at 2 flops each, are a fraction `κ` of
the reference circuit's size, every declaration that contains it and complies at `(Γ, α)` is larger by a factor of at
least `κ F`. -/
def ContainedTraining : Prop :=
  ∀ (Γ : ℕ) (α b_w b_a κ : ℝ) (c d : Declaration) (θv : c.D.Transcript) (ch : Chain c.D) (dd : ℕ),
    0 < α → 0 < b_w → 0 < b_a → (∀ ℓ, dd ≤ ch.d ℓ) → c.Contains d → Complies Γ α d →
    Assumptions.ActivationsSupport c θv ch (α * Γ + b_w + b_a) b_w b_a →
    Assumptions.PartialSumsSupport c θv ch Γ (α * Γ) b_w b_a →
    κ * c.D.size c.D.computed ≤ 2 * ch.G →
      κ * F α (α * Γ) b_w b_a dd * c.D.size c.D.computed ≤ d.D.size d.D.computed

end Nci.Guarantees
