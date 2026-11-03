import Definitions.Nci.Assumptions.Formalism

/-!
# The training formalism v2: its theorems

The theorems of Daniel's training formalism v2 (`Definitions.Nci.Formalism`), §5, proved in
`Nci.Formalism.SecurityProofs`. Its two lemmas, reconstruction (§1) and the three kinds of a block's inputs (§2), are
`Nci.Formalism.reconstruction` and `Nci.Formalism.reads_split` in `Security/Proofs/Nci/Formalism/Reconstruction.lean`.

A lower bound on `ρ` is stated, as each is proved, for every partition `𝓘` of `C` into inference blocks:
`X ≤ |𝓘| c / |C|` for each is `X ≤ ρ` (`Nci.Formalism.le_slowdown_iff`). Every `≈` and `≳` of v2 is an exact bound
here, its factors explicit, and the claim reads the upper bounds A1 and A1′ give, never best-known values:

* **The counting lemma** (`Counting`, no assumptions): for any set of gates `S`,
  `ρ ≥ (|S| / |C|) / (max_I |I ∩ S| / c)`. **Slowdown** (`Slowdown`) is it at `S = A \ R`;
* **the token form** (`TokenForm`), identical token blocks: `ρ ≥ σ (1 - x₀) s / s*`, `x₀` any bound on the reachable
  shares. Identical is what the bound reads of them, the same `α > 0` affected gates each. Varying token blocks
  (`TokenFormVarying`): `ρ ≥ (1 - x₀) |A ∩ ⋃ 𝒯| c / (|C| α_max s*)`, the largest affected count `α_max` in place of a
  common one; at identical disjoint token blocks `|A ∩ ⋃ 𝒯| / α_max` is their number, and `σ s ≤ |𝒯| c / |C|`;
* **from A1** (`CapacityBound`): `a s* ≤ M` for any `M ≥ max_x x (k - u(x))`, v2's `s* ≤ max_x x (k - u(x)) / a` with
  no division by `a = 0`;
* **from A1′** (`ReachableShareBound`): every reachable share is at most `D_T(ξ)`, for any `ξ ≥ max {x : u_in(x) ≤ k}`,
  `D_T` the token's dependency profile. v2's `x₀ ≤ max {x : u_in(x) ≤ k}` is its case `D_T(ξ) ≤ ξ`, which holds when a
  token's affected gates are a chain, and fails for a token of two independent halves: a block computes each half from
  its inputs, and `R` holds the first share `ξ` of both;
* **the claim** (`SlowdownClaim`), under A1 and A1′, identical token blocks: `ρ ≥ σ (1 - x₀) s a / M`, for any
  `x₀ ≥ D_T(ξ)`. Evenly spread, `u = u_in = x u(1)` with `u(1) > 0` (`SlowdownEven`):
  `ρ ≥ 4 σ (1 - x₀) s a u(1) / k²`, for any `x₀ ≥ D_T(k / u(1))`, v2's `4 σ s a u(1) / k²` with the factor `1 - x₀`,
  near `1` when `u(1) ≫ k`. Frozen layers above changed ones (`SlowdownFrozen`), `u` above `k` beyond the frozen share
  `t`: `ρ ≥ σ (1 - x₀) s a / (t k)`, v2's `σ s a / (t k)` with the factor `1 - x₀`, near `1` when `u_in` jumps at once;
* **Fit**: a partition whose every block holds at least `c - d` gates and reads within `k` gives `ρ ≤ c / (c - d)`.
  Running `θ'` (`FitServing`), session blocks with `Δ + s τ ≤ k`; honest serving, `Δ = 0` and no decoder, has exactly
  `ρ = 1`, one block per session (`FitHonestServing`). Training (`FitTraining`), compute blocks with `Δ + s τ ≤ k`,
  reduction blocks with `2 Δ_grad ≤ k` and update application blocks with `Δ + Δ_grad ≤ k`. Fit counts the blocks'
  gates in `C`: `c / (c - d)` is `ρ` against a circuit whose blocks each leave `d` of their `c` gates to a decoder;
* **Security** (`Security`), under A1, A1′ and A2: a run of the methods considered that reaches `ℓ*` through a partition
  into inference blocks that fit in `N` periods of `n` blocks has `N n c ≥ ρ̲ p H(ℓ*)`, `ρ̲` the claim's bound for it
  and `p` A2's floor: reaching `ℓ*` covertly costs at least `ρ̲ p H(ℓ*)`, with at most `n c` available per period.

The bounds on `ρ`, the claim's cases, Fit's bounds and Security each have an honest and a nontrivial lemma
(`Definitions.Nci.Guarantees.Formalism.Witnesses`).
-/

namespace Nci.Formalism.Guarantees

open Finset
open Verity (Declaration Partition)
open Nci.Formalism Nci.Formalism.Assumptions
open scoped NNReal ENNReal

/-- **The counting lemma**: for any set of gates `S`, every partition `𝓘` into inference blocks has
`|𝓘| c / |C| ≥ (|S| / |C|) / (max_I |I ∩ S| / c)`. -/
def Counting : Prop :=
  ∀ (d : Declaration) (c k : ℕ) (S : Finset (Fin d.D.N)) (B : ℕ) (P : Partition d.D B), GateSet d.D S →
    P.IsCK d c k →
    ((S.card : ℝ≥0∞) / (d.D.computed.card : ℝ≥0∞)) / ((maxInBlock d c k S : ℝ≥0∞) / (c : ℝ≥0∞)) ≤
      partitionSlowdown d B c

/-- **Slowdown**: the counting lemma at `S = A \ R`, the affected gates no inference block reaches from its inputs. -/
def Slowdown : Prop :=
  ∀ (d : Declaration) (L : Type) (A : Annotation d L) (c k B : ℕ) (P : Partition d.D B), P.IsCK d c k →
    (((A.affected \ A.reachable c k).card : ℝ≥0∞) / (d.D.computed.card : ℝ≥0∞)) /
      ((maxInBlock d c k (A.affected \ A.reachable c k) : ℝ≥0∞) / (c : ℝ≥0∞)) ≤ partitionSlowdown d B c

/-- **The token form**, identical token blocks: `ρ ≥ σ (1 - x₀) s / s*`, for any `x₀` at least every reachable share. -/
def TokenForm : Prop :=
  ∀ (d : Declaration) (L : Type) [Fintype L] (A : Annotation d L) (c k α : ℕ) (x₀ : ℝ≥0) (B : ℕ)
    (P : Partition d.D B), A.Identical α → A.ReachableAtMost c k x₀ → P.IsCK d c k →
    ((A.tokenShare * (1 - x₀) * A.tokensPerBlock c : ℝ≥0) : ℝ≥0∞) / (A.unreachableCapacity c k : ℝ≥0∞) ≤
      partitionSlowdown d B c

/-- **The token form, varying token blocks**: `ρ ≥ (1 - x₀) |A ∩ ⋃ 𝒯| c / (|C| α_max s*)`, `α_max = max_T |T ∩ A|`. -/
def TokenFormVarying : Prop :=
  ∀ (d : Declaration) (L : Type) [Fintype L] (A : Annotation d L) (c k : ℕ) (x₀ : ℝ≥0) (B : ℕ) (P : Partition d.D B),
    A.ReachableAtMost c k x₀ → P.IsCK d c k →
    (((1 - x₀) * (A.affected ∩ A.tokenGates).card * c : ℝ≥0) : ℝ≥0∞) /
      ((d.D.computed.card * A.maxAffected * A.unreachableCapacity c k : ℝ≥0) : ℝ≥0∞) ≤ partitionSlowdown d B c

/-- **`s*` from A1**: `a s* ≤ M`, for any `M ≥ max_x x (k - u(x))`. -/
def CapacityBound : Prop :=
  ∀ (d : Declaration) (L : Type) [Fintype L] (A : Annotation d L) (c k : ℕ) (u : ℝ≥0 → ℝ≥0) (a M : ℝ≥0),
    CutCost d A c k u a → PeakBound u k M → a * A.unreachableCapacity c k ≤ M

/-- **The reachable share from A1′**: every token's reachable share is at most its dependency profile at `ξ`, for any
`ξ ≥ max {x : u_in(x) ≤ k}`. -/
def ReachableShareBound : Prop :=
  ∀ (d : Declaration) (L : Type) (A : Annotation d L) (c k : ℕ) (u_in : ℝ≥0 → ℝ≥0) (ξ : ℝ≥0),
    A.Valid → InputCost d A c k u_in → ReachLimit u_in k ξ → ∀ T, A.reachableShare c k T ≤ A.dependencyProfile T ξ

/-- **The claim**: under A1 and A1′, identical token blocks, `ρ ≥ σ (1 - x₀) s a / M`, for any
`M ≥ max_x x (k - u(x))`, `ξ ≥ max {x : u_in(x) ≤ k}` and `x₀` at least every token's dependency profile at `ξ`. -/
def SlowdownClaim : Prop :=
  ∀ (d : Declaration) (L : Type) [Fintype L] (A : Annotation d L) (c k α : ℕ) (u u_in : ℝ≥0 → ℝ≥0) (a M ξ x₀ : ℝ≥0)
    (B : ℕ) (P : Partition d.D B),
    A.Valid → A.Identical α → CutCost d A c k u a → PeakBound u k M → InputCost d A c k u_in → ReachLimit u_in k ξ →
    A.ProfileAtMost ξ x₀ → P.IsCK d c k →
    ((A.tokenShare * (1 - x₀) * A.tokensPerBlock c * a : ℝ≥0) : ℝ≥0∞) / (M : ℝ≥0∞) ≤ partitionSlowdown d B c

/-- **The claim, evenly spread update** `u = u_in = x u(1)`, `u(1) > 0`: `ρ ≥ 4 σ (1 - x₀) s a u(1) / k²`, for any `x₀`
at least every token's dependency profile at `k / u(1)`. -/
def SlowdownEven : Prop :=
  ∀ (d : Declaration) (L : Type) [Fintype L] (A : Annotation d L) (c k α : ℕ) (u₁ a x₀ : ℝ≥0) (B : ℕ)
    (P : Partition d.D B),
    A.Valid → A.Identical α → CutCost d A c k (fun x => x * u₁) a → InputCost d A c k (fun x => x * u₁) → 0 < u₁ →
    A.ProfileAtMost ((k : ℝ≥0) / u₁) x₀ → P.IsCK d c k →
    ((4 * A.tokenShare * (1 - x₀) * A.tokensPerBlock c * a * u₁ : ℝ≥0) : ℝ≥0∞) / (((k : ℝ≥0) ^ 2 : ℝ≥0) : ℝ≥0∞) ≤
      partitionSlowdown d B c

/-- **The claim, frozen layers above changed ones**, `u` above `k` beyond the frozen share `t`:
`ρ ≥ σ (1 - x₀) s a / (t k)`. -/
def SlowdownFrozen : Prop :=
  ∀ (d : Declaration) (L : Type) [Fintype L] (A : Annotation d L) (c k α : ℕ) (u u_in : ℝ≥0 → ℝ≥0) (a t ξ x₀ : ℝ≥0)
    (B : ℕ) (P : Partition d.D B),
    A.Valid → A.Identical α → CutCost d A c k u a → FrozenShare u k t → InputCost d A c k u_in →
    ReachLimit u_in k ξ → A.ProfileAtMost ξ x₀ → P.IsCK d c k →
    ((A.tokenShare * (1 - x₀) * A.tokensPerBlock c * a : ℝ≥0) : ℝ≥0∞) / ((t * k : ℝ≥0) : ℝ≥0∞) ≤
      partitionSlowdown d B c

/-- **Fit, running `θ'`**: session blocks of at least `c - d` gates that read within `k`, `Δ + s τ ≤ k`, give
`ρ ≤ c / (c - d)`. -/
def FitServing : Prop :=
  ∀ (d : Declaration) (B : ℕ) (P : Partition d.D B) (c k dg Δ s τ : ℕ), d.D.Boolean → SessionBlocks d P c dg Δ s τ →
    ServingFits Δ s τ k → slowdown d c k ≤ (c : ℝ≥0∞) / ((c - dg : ℕ) : ℝ≥0∞)

/-- **Fit, honest serving**: with `Δ = 0` and no decoder, session blocks that read `s τ ≤ k` give exactly `ρ = 1`, one
block per session. -/
def FitHonestServing : Prop :=
  ∀ (d : Declaration) (B : ℕ) (P : Partition d.D B) (c k s τ : ℕ), d.D.Boolean → SessionBlocks d P c 0 0 s τ →
    ServingFits 0 s τ k → 0 < B → 0 < c → slowdown d c k = 1

/-- **Fit, training**: compute, reduction and update application blocks of at least `c - d` gates that each read
within `k` give `ρ ≤ c / (c - d)`. -/
def FitTraining : Prop :=
  ∀ (d : Declaration) (B : ℕ) (P : Partition d.D B) (c k dg Δ Δgrad s τ : ℕ), d.D.Boolean →
    TrainingBlocks d P c dg Δ Δgrad s τ → TrainingFits Δ Δgrad s τ k →
    slowdown d c k ≤ (c : ℝ≥0∞) / ((c - dg : ℕ) : ℝ≥0∞)

/-- **Security**: under A1, A1′ and A2, a run of `methods` whose update stays within `D` bits and that reaches `ℓ*`
through a partition into inference blocks that fit in `N` periods of `n` blocks has `N n c ≥ ρ̲ p H(ℓ*)`, `ρ̲` the
claim's bound. -/
def Security : Prop :=
  ∀ (methods : Set Run) (dec : Verity.Decoder (List Bool)) (ℓ : List Bool → ℝ) (ℓstar : ℝ) (D : ℕ∞) (p : ℝ) (r : Run)
    (c k n N α : ℕ) (u u_in : ℝ≥0 → ℝ≥0) (a M ξ x₀ : ℝ≥0) (B : ℕ) (P : Partition r.d.D B),
    EfficiencyFloor methods dec c ℓ ℓstar D p → r.Among methods → r.A.Valid → r.Within dec c D → r.Reaches ℓ ℓstar →
    r.A.Identical α → CutCost r.d r.A c k u a → PeakBound u k M → InputCost r.d r.A c k u_in → ReachLimit u_in k ξ →
    r.A.ProfileAtMost ξ x₀ → P.IsCK r.d c k → InPeriods B N n →
    ((r.A.tokenShare * (1 - x₀) * r.A.tokensPerBlock c * a : ℝ≥0) : ℝ≥0∞) / (M : ℝ≥0∞) * ENNReal.ofReal p *
      honestCost methods ℓ ℓstar ≤ ((N * n * c : ℕ) : ℝ≥0∞)

end Nci.Formalism.Guarantees
