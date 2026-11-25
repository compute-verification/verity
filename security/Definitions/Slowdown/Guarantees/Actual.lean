module

public import Definitions.Registration.Physical
public import Definitions.Core.Partition
public import Definitions.InferenceOnly.Reads
public import Definitions.InferenceOnly.Assumptions

/-!
# The actual circuit's runtime statements (trusted)

In the actual-circuit model, the developer runs at `n` locations `X 0, …, X (n − 1)`, pairwise disjoint, and its region
over `[0, t)`, as a circuit whose inputs are what it reads with the baseline `θ_A` (`Placed.declaration`), is the
declaration the inference-only policy's statements are about.

* `ServerBlocksAt` (the capacity form at any cuts): for any cuts `τ` of `[0, t)` and any set `G` of gates, the windows
  `A[X i, [τ j, τ (j + 1))]` divide that declaration's computed gates into `n J` inference blocks, each of at most
  `Φ_X (τ (j + 1) − τ j)` gates and reading at most `stateIn_X(τ j, G) + β_X (τ (j + 1) − τ j)` bits in `G`, baseline
  included (`InferenceOnly.readsIn`). It takes no memory constraint: the state at each cut is in the bound. At the
  cuts `kω` and every gate, a location whose state is within its memory `M_X` reads at most `M_X + β_X ω` a window.

The proofs are in `security/Proofs/ResourceAccounting/Actual/`.
-/

@[expose] public section

namespace ResourceAccounting.Guarantees.Actual

open Verity InferenceOnly Finset

/-- **The windows at any cuts divide the developer's region into small inference blocks** (capacity form at cuts). For every actual
circuit `A` meeting causality, unit-priced compute rate and links at the developer's pairwise disjoint locations `X i`,
every `t`, all cuts `κ` of `[0, t)` and every set `G` of gates: some division `P` of the developer's declaration over
`[0, t)` into `n J` units has as its unit `(i, j)` the window `A[X i, [τ j, τ (j + 1))]`, of at most
`Φ_{X i} (τ (j + 1) − τ j)` gates, whose reads in `G`, baseline included, are at most the part in `G` of `X i`'s state
at `τ j` plus `β_{X i} (τ (j + 1) − τ j)`. -/
def ServerBlocksAt : Prop :=
  ∀ {L E : Type*} [Fintype E] [DecidableEq L] (A : Placed L) (ℓ : L → L → ℕ) (Φ : Finset L → ℕ)
    (ends : E → L × L) (β : E → ℕ) (link : Fin A.N → Fin A.N → E) (n : ℕ) (X : Fin n → Finset L)
    (θ : Finset (Fin A.N)) (hθ : ∀ a ∈ θ, A.isInput a = true) (t : ℕ) (κ : Cuts t) (G : Finset (Fin A.N)),
    (∀ i j, i ≠ j → Disjoint (X i) (X j)) →
    A.Causal ℓ → A.ComputeRate (Set.range X) (fun _ => 1) Φ → A.Links ends β link →
    ∃ P : Partition (A.declaration (univ.biUnion X) t θ hθ).D (n * κ.J), ∀ (i : Fin n) (j : Fin κ.J),
      P.gates (finProdFinEquiv (i, j)) = A.region (X i) (Set.Ico (κ.τ j.castSucc) (κ.τ j.succ)) ∧
      (A.declaration (univ.biUnion X) t θ hθ).D.size (P.gates (finProdFinEquiv (i, j))) ≤
        Φ (X i) * (κ.τ j.succ - κ.τ j.castSucc) ∧
      readsIn (A.declaration (univ.biUnion X) t θ hθ) (P.gates (finProdFinEquiv (i, j))) G ≤
        A.stateIn (X i) (κ.τ j.castSucc) G + linkCapacity ends β (X i) * (κ.τ j.succ - κ.τ j.castSucc)

end ResourceAccounting.Guarantees.Actual
