module

public import Definitions.Pouw.Guarantees.Dither

/-!
# The no-fold Dither protocol's work-soundness over a monitoring period: lemmas

The statements of `Definitions.Pouw.Guarantees.Dither` that are steps of its guarantees' proofs rather than guarantees.
Each is a `def G : Prop` under the guarantees' namespace, proved by the `SecurityProofs` theorem of the same name, and
audited with the proofs: no record pins it.

* `FreedOfFloor`: the rounds' floor at the period's dual price bounds freed compute by `ε C` (weak duality); the step
  from the floor to `WorkSound`'s conclusion.
-/

@[expose] public section

namespace Pouw.Guarantees.Dither

open Pouw.RelativeSecurity Pouw.Dither Pouw.Assumptions Pouw.MerkleExtract

/-- **Freed compute is at most `ε C` at the period's dual price**. On a period's devices (`Period`), let
`μ ≥ 0` be a dual price for the other work (every job worth at most its use at `μ`) at which an honest strategy
credited `W` with other work `z'` is optimal (its other work worth exactly the capacity it leaves, at `μ`:
complementary slackness). Any strategy credited the same `W`, whose credited part uses `X` and whose other work `z`
fits beside it, and whose credited part meets the work floor at `μ` (`μ · X ≥ (1 − ε) μ · W hon`), frees at most
`ε C` with `C = μ · W hon`, the credited rounds' honest cost at the dual price. The floor on the whole
run would not be enough here: it would count the other work `z` toward the credited part's floor. -/
def FreedOfFloor : Prop :=
  ∀ (G : Accel) (Pd : Period G) (W ε : ℝ) (X μ : G.Res → ℝ) (z z' : Pd.J → ℝ),
    (∀ r, 0 ≤ μ r) → (∀ j r, 0 ≤ Pd.needs j r) → (∀ j, Pd.val j ≤ ∑ r, μ r * Pd.needs j r) → 0 ≤ W → 0 ≤ ε →
    Pd.Fits (fun r => W * Pd.hon r) z' → ∑ j, Pd.val j * z' j = ∑ r, μ r * (Pd.cap r - W * Pd.hon r) →
    Pd.Fits X z → (1 - ε) * ∑ r, μ r * (W * Pd.hon r) ≤ ∑ r, μ r * X r →
    Pd.FreedWithin W z ε (∑ r, μ r * (W * Pd.hon r))

end Pouw.Guarantees.Dither
