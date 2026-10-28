import Definitions.Accounting.Physical
import Definitions.Core.Partition
import Definitions.Nci.Reads
import Definitions.Nci.Assumptions

/-!
# The actual circuit's runtime statements (trusted)

In the actual-circuit model, the developer runs at `n` locations `X 0, …, X (n − 1)`, pairwise disjoint, and its region
over `[0, t)`, as a circuit whose inputs are what it reads with the baseline `θ_A` (`Placed.declaration`), is the
declaration NCI's statements are about.

* `WindowsAt` (the capacity form at any cuts): for any cuts `τ` of `[0, t)` and any set `G` of gates, the windows
  `A[X i, [τ j, τ (j + 1))]` divide that declaration's computed gates into `n J` IUs, each of at most
  `Φ_X (τ (j + 1) − τ j)` gates and reading at most `stateIn_X(τ j, G) + β_X (τ (j + 1) − τ j)` bits in `G`, baseline
  included (`Nci.readsIn`). It takes no memory constraint: the state at each cut is in the bound.
* `Windows` (the capacity form): the windows `A[X i, [kω, (k + 1)ω)]`, cut at `t`, divide that declaration's computed
  gates into `n ⌈t/ω⌉` IUs, each of at most `Φ_X ω` gates (work priced at one per gate) and reading at most
  `M_X + β_X ω` novel bits. It takes the four physical constraints it uses (`Causal`, `ComputeRate`, `Memory`, `Links`).
* `Runtime` (the corollary): when, in a run, the developer's region before every `t` by which it holds `P`'s outputs
  at positions fixed across the values of `P`'s inputs (the runtime reads the strategy: `Actual.runtime`) runs a
  chain of at least `G` products, under NCI's two incompressibility assumptions at `(Γ_A, K_A)`, the runtime is at least
  `ω (G f(K_A) / (n K_A) − 1)`, where `f = Nci.readFactor` (`F = f(K)/(2α)`) and `Γ_A ≥ Φ_X ω`, `K_A ≥ M_X + β_X ω` at
  every location.

The proofs are in `Security/Proofs/Accounting/Actual/`.
-/

namespace Accounting.Guarantees.Actual

open Verity Nci Finset

/-- **The windows at any cuts divide the developer's region into small IUs** (capacity form at cuts). For every actual
circuit `A` meeting causality, unit-priced compute rate and links at the developer's pairwise disjoint locations `X i`,
every `t`, all cuts `κ` of `[0, t)` and every set `G` of gates: some division `P` of the developer's declaration over
`[0, t)` into `n J` units has as its unit `(i, j)` the window `A[X i, [τ j, τ (j + 1))]`, of at most
`Φ_{X i} (τ (j + 1) − τ j)` gates, whose reads in `G`, baseline included, are at most the part in `G` of `X i`'s state
at `τ j` plus `β_{X i} (τ (j + 1) − τ j)`. -/
def WindowsAt : Prop :=
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

/-- **The windows divide the developer's region into small IUs that read little** (capacity form). For every actual
circuit `A` meeting causality, unit-priced compute rate, memory and links at the developer's pairwise disjoint locations
`X i`, and every window `ω > 0` and `t`: some division `P` of the developer's declaration over `[0, t)` into
`n ⌈t/ω⌉` units has as its unit `(i, k)` the window `A[X i, [kω, (k + 1)ω) ∩ [0, t)]`, of at most `Φ_{X i} ω` gates,
which reads at most `M_{X i} + β_{X i} ω` bits of novel information. -/
def Windows : Prop :=
  ∀ {L E : Type*} [Fintype E] [DecidableEq L] (A : Placed L) (ℓ : L → L → ℕ) (Φ M : Finset L → ℕ)
    (ends : E → L × L) (β : E → ℕ) (link : Fin A.N → Fin A.N → E) (n : ℕ) (X : Fin n → Finset L)
    (θ : Finset (Fin A.N)) (hθ : ∀ a ∈ θ, A.isInput a = true) (t ω : ℕ),
    0 < ω → (∀ i j, i ≠ j → Disjoint (X i) (X j)) →
    A.Causal ℓ → A.ComputeRate (Set.range X) (fun _ => 1) Φ → A.Memory (Set.range X) M → A.Links ends β link →
    ∃ P : Partition (A.declaration (univ.biUnion X) t θ hθ).D (n * ((t + ω - 1) / ω)),
      ∀ (i : Fin n) (k : Fin ((t + ω - 1) / ω)),
        P.gates (finProdFinEquiv (i, k)) = A.region (X i) (Set.Ico (k * ω) ((k + 1) * ω) ∩ Set.Iio t) ∧
        (A.declaration (univ.biUnion X) t θ hθ).D.size (P.gates (finProdFinEquiv (i, k))) ≤ Φ (X i) * ω ∧
        novelReads (A.declaration (univ.biUnion X) t θ hθ) (P.gates (finProdFinEquiv (i, k))) ≤
          M (X i) + linkCapacity ends β (X i) * ω

/-- **A prohibited computation takes long on the actual circuit** (the runtime corollary). For a circuit `P`, the runs
`S v` of a strategy with each value `v` of `P`'s inputs in place (`Actual.runtime`), one of them, `S v`, whose actual
circuit `A` meets the constraints of `Windows` at the locations and `ω` as there, caps `Γ ≥ Φ_{X i} ω` and
`K ≥ M_{X i} + β_{X i} ω` at every location, and such that at every `t` by which positions of the developer's before
`t`, the same in every run, hold `P`'s outputs (`Actual.Computes`), `A[X, [0, t))` runs a chain of at least `G`
products, of inner dimensions at least `d`, under NCI's two incompressibility assumptions at `(Γ, K)`: every `t` at or
past the runtime `t_A(P)` is at least `ω (G f(K) / (n K) − 1)`, `f = Nci.readFactor`, so `t_A(P)` is. -/
def Runtime : Prop :=
  ∀ {L E : Type*} [Fintype E] [DecidableEq L] (P : Circuit) (S : P.Transcript → Actual.Run L) (v : P.Transcript)
    (ℓ : L → L → ℕ) (Φ M : Finset L → ℕ) (ends : E → L × L) (β : E → ℕ)
    (link : Fin (S v).A.N → Fin (S v).A.N → E) (n : ℕ) (X : Fin n → Finset L) (θ : Finset (Fin (S v).A.N))
    (hθ : ∀ a ∈ θ, (S v).A.isInput a = true) (ω Γ : ℕ) (K b_w b_a : ℝ) (G d : ℕ),
    0 < ω → (∀ i j, i ≠ j → Disjoint (X i) (X j)) →
    (S v).A.Causal ℓ → (S v).A.ComputeRate (Set.range X) (fun _ => 1) Φ → (S v).A.Memory (Set.range X) M →
    (S v).A.Links ends β link →
    0 < b_w → 0 < b_a →
    (∀ i, Φ (X i) * ω ≤ Γ ∧ ((M (X i) + linkCapacity ends β (X i) * ω : ℕ) : ℝ) ≤ K) →
    (∀ t, Actual.Computes S (univ.biUnion X) t →
      ∃ c : Chain ((S v).A.declaration (univ.biUnion X) t θ hθ).D, G ≤ c.G ∧ (∀ j, d ≤ c.d j) ∧
        Assumptions.ActivationsIncompressible ((S v).A.declaration (univ.biUnion X) t θ hθ) c Γ K b_w b_a ∧
        Assumptions.PartialSumsIncompressible ((S v).A.declaration (univ.biUnion X) t θ hθ) c Γ K b_w b_a) →
    ∀ t : ℕ, Actual.runtime S (univ.biUnion X) ≤ t →
      (ω : ℝ) * (G * readFactor K b_w b_a d / (n * K) - 1) ≤ t

end Accounting.Guarantees.Actual
