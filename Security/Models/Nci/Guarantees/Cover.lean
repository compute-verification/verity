import Models.Nci.Assumptions.Cover

/-!
# The cover theorem: the guarantees

A run that declares its computation as inference blocks (subcircuits of `s` gates, each handed at most `k` bits besides
the registered parameters `θ`) and covers a training circuit with them pays for it in blocks. The argument has three
steps, and nothing in it assumes its conclusion:

1. Assume `UpdateHardness`: a block that computes whole products over `w` distinct updated parameters is handed `b w`
   bits.
2. From a structural property of the training computation, the upstream certificate (`MulBlocks.Upstream`), prove that
   activations require communication: a block that computes whole a counted product of use `j` is handed a gate of `j`'s
   activation region (`ActivationCharge`, through `Entry` and `BackwardClosure`).
3. Combine them: a block finishes at most `k² / b + k` counted products (`Intensity`), a cover needs
   `#𝓘 ≥ b #P / (k² + b k)` blocks (`CoverBound`), and the declared compute `s #𝓘` is at least `α₀ s b / (k² + b k)`
   times a baseline in which the counted products are a fraction `α₀` (`Slowdown`).

Under stronger named premises the factor grows, each variant its own guarantee: `ConditionalUpdateHardness` gives
`4 b #P ≤ #𝓘 (k² + 4 b k)` (`AdditiveCoverBound`), and with `Width f` as well, `4 b f #P ≤ #𝓘 (k² + 4 b f k)`
(`WidthCoverBound`).

Kimi K2, at `k = 2·10⁷` bits, `s = 4.52·10¹⁶` gates, `α₀ = 0.147` and `b = 2.5`: the area factor `α₀ s b / k²` is about
41, the additive one about 166, and the width one at `f = 4` about 660. The certificate holds with `D` about one layer's
active weights, `4·10⁸`, so `b D ≈ 10⁹` exceeds `k` fifty times.

Scope. First-layer products, whose activation is a function of the token id, have no certificate and are left out of
`P`. `UpdateHardness` needs the run to have drifted from `θ`, and fails under compressed updates such as LoRA.
-/

namespace Nci.Cover.Guarantees

/-- **Entry** (lemma 1): a path that ends in a subcircuit `I` and starts at a gate that `I` does not compute passes
through an input of `I`. A value reaches a block's computation only through the gates the block is handed. -/
def Entry : Prop :=
  ∀ (C : Circuit) (I : Finset (Fin C.N)) (π : List (Fin C.N)) (x y : Fin C.N),
    C.Subcircuit I → C.IsPath π x y → y ∈ I → x ∉ C.computed I → ∃ g ∈ π, g ∈ C.inputs I

/-- **Backward closure** (lemma 2): a subcircuit `I` that holds a gate `a` and is handed no gate of a region `A`
computes every gate that reaches `a` inside `A`, `a` itself among them. A block that uses an activation without being
handed any gate of its region has computed all of the activation's upstream computation. -/
def BackwardClosure : Prop :=
  ∀ (C : Circuit) (I A : Finset (Fin C.N)) (a x : Fin C.N),
    C.Subcircuit I → a ∈ I → C.inputs I ∩ A = ∅ → C.ReachesIn A x a → x ∈ C.computed I

/-- **Activation charge** (lemma 3): under `UpdateHardness` and the upstream certificate at depth `D` with `k < b D`, an
inference block computes whole counted products of at most `#(inputs I \ θ)` uses.

A block that computes whole a counted product of use `j` uses its activation. Were it handed no gate of `A j`, it would
compute all of `j`'s upstream products (`BackwardClosure`), over at least `D` updated parameters, which `UpdateHardness`
prices at `b D > k` bits. So it is handed a gate of `A j`, which is not in `θ`, and the regions are disjoint, so
distinct uses cost distinct gates. In ML terms: a block computes a token's products at a layer whole only if it is
handed part of that token's activation region, since computing the activation itself means rebuilding a whole layer's
updated weights (Kimi K2: `b D ≈ 10⁹`, fifty times `k`). -/
def ActivationCharge : Prop :=
  ∀ (C : Circuit) {J L : Type} [DecidableEq J] [DecidableEq L] (M : MulBlocks C J L) (A : J → Finset (Fin C.N))
    (s k D : ℕ) (b : ℝ),
    0 < b → Assumptions.UpdateHardness M s b → M.Upstream A D → (k : ℝ) < b * D →
    ∀ I, C.InferenceBlock s k I → ((M.whole I).image Prod.fst).card ≤ (C.inputs I \ C.θ).card

/-- **Intensity** (area): under `UpdateHardness` and the upstream certificate with `k < b D`, an inference block
finishes at most `k · k / b + k` counted products: `b · #(finished) ≤ k² + b k`.

The counted products it computes whole lie in `J × L` with at most `k` uses (`ActivationCharge`) and at most `k / b`
updated parameters (`UpdateHardness`), each pair once. Each of the others, whose output gate it computes but not all of
whose gates, enters the block through an input among its own gates (`Entry`): not in `θ`, and distinct since products
are disjoint, so at most `k`. -/
def Intensity : Prop :=
  ∀ (C : Circuit) {J L : Type} [DecidableEq J] [DecidableEq L] (M : MulBlocks C J L) (A : J → Finset (Fin C.N))
    (s k D : ℕ) (b : ℝ),
    0 < b → Assumptions.UpdateHardness M s b → M.Upstream A D → (k : ℝ) < b * D →
    ∀ I, C.InferenceBlock s k I → b * (M.finished I).card ≤ k * k + b * k

/-- **Cover bound** (area): under `UpdateHardness` and the upstream certificate with `k < b D`, a cover of `C` by
inference blocks has at least `b #P / (k² + b k)` blocks: `b · #P ≤ #𝓘 · (k² + b k)`. Every counted product's output
gate is computed in some block (`Covers`), and each block finishes at most `(k² + b k) / b` of them (`Intensity`). -/
def CoverBound : Prop :=
  ∀ (C : Circuit) {J L : Type} [DecidableEq J] [DecidableEq L] (M : MulBlocks C J L) (A : J → Finset (Fin C.N))
    (s k D : ℕ) (b : ℝ) (𝓘 : Finset (Finset (Fin C.N))),
    0 < b → Assumptions.UpdateHardness M s b → M.Upstream A D → (k : ℝ) < b * D →
    (∀ I ∈ 𝓘, C.InferenceBlock s k I) → C.Covers 𝓘 →
    b * M.P.card ≤ 𝓘.card * (k * k + b * k)

/-- **Slowdown** (area): when the counted products are at least a fraction `α₀` of a baseline of `n₀` gates (the
training computation run as it is), the declared compute `s · #𝓘` of a cover by inference blocks is at least
`α₀ s b / (k² + b k)` times the baseline: `α₀ n₀ s b / (k² + b k) ≤ s · #𝓘`. It is `CoverBound`, multiplied out.

Kimi K2 (`k = 2·10⁷` bits, `s = 4.52·10¹⁶` gates, `α₀ = 0.147`, `b = 2.5`): about 41. It holds for the counted products
only: first-layer products are not in `P`, and `UpdateHardness` needs the run to have drifted from `θ` and fails under
compressed updates such as LoRA. -/
def Slowdown : Prop :=
  ∀ (C : Circuit) {J L : Type} [DecidableEq J] [DecidableEq L] (M : MulBlocks C J L) (A : J → Finset (Fin C.N))
    (s k D n₀ : ℕ) (b α₀ : ℝ) (𝓘 : Finset (Finset (Fin C.N))),
    0 < b → Assumptions.UpdateHardness M s b → M.Upstream A D → (k : ℝ) < b * D →
    (∀ I ∈ 𝓘, C.InferenceBlock s k I) → C.Covers 𝓘 → α₀ * n₀ ≤ M.P.card →
    α₀ * n₀ * s * b / (k * k + b * k) ≤ s * 𝓘.card

/-- **Intensity, additive**: under `ConditionalUpdateHardness` (which implies `UpdateHardness`) and the upstream
certificate with `k < b D`, an inference block finishes at most `k² / (4 b) + k` counted products:
`4 b · #(finished) ≤ k² + 4 b k`.

The uses of the products it computes whole are paid inside the activation regions and their updated parameters outside
them, so `#J + b #L ≤ k`, and `4 b #J #L ≤ (#J + b #L)² ≤ k²` (AM–GM). The rest are at most `k`, as in `Intensity`. -/
def AdditiveIntensity : Prop :=
  ∀ (C : Circuit) {J L : Type} [DecidableEq J] [DecidableEq L] (M : MulBlocks C J L) (A : J → Finset (Fin C.N))
    (s k D : ℕ) (b : ℝ),
    0 < b → Assumptions.ConditionalUpdateHardness M A s b → M.Upstream A D → (k : ℝ) < b * D →
    ∀ I, C.InferenceBlock s k I → 4 * b * (M.finished I).card ≤ k * k + 4 * b * k

/-- **Cover bound, additive**: under `ConditionalUpdateHardness` and the upstream certificate with `k < b D`, a cover of
`C` by inference blocks has `4 b · #P ≤ #𝓘 · (k² + 4 b k)`: four times `CoverBound`'s factor for `k ≫ b` (Kimi K2: about
166). -/
def AdditiveCoverBound : Prop :=
  ∀ (C : Circuit) {J L : Type} [DecidableEq J] [DecidableEq L] (M : MulBlocks C J L) (A : J → Finset (Fin C.N))
    (s k D : ℕ) (b : ℝ) (𝓘 : Finset (Finset (Fin C.N))),
    0 < b → Assumptions.ConditionalUpdateHardness M A s b → M.Upstream A D → (k : ℝ) < b * D →
    (∀ I ∈ 𝓘, C.InferenceBlock s k I) → C.Covers 𝓘 →
    4 * b * M.P.card ≤ 𝓘.card * (k * k + 4 * b * k)

/-- **Intensity, width `f`**: under `ConditionalUpdateHardness`, `Width f` and the upstream certificate with `k < b D`,
an inference block finishes at most `k² / (4 b f) + k` counted products: `4 b f · #(finished) ≤ k² + 4 b f k`.

Each use of the products it computes whole now costs `f` gates of its region, so `f #J + b #L ≤ k`, and AM–GM gives
`4 b f #J #L ≤ k²`. The rest are at most `k`, as in `Intensity`. -/
def WidthIntensity : Prop :=
  ∀ (C : Circuit) {J L : Type} [DecidableEq J] [DecidableEq L] (M : MulBlocks C J L) (A : J → Finset (Fin C.N))
    (s k D f : ℕ) (b : ℝ),
    0 < b → Assumptions.ConditionalUpdateHardness M A s b → Assumptions.Width M A s f → M.Upstream A D →
    (k : ℝ) < b * D →
    ∀ I, C.InferenceBlock s k I → 4 * b * f * (M.finished I).card ≤ k * k + 4 * b * f * k

/-- **Cover bound, width `f`**: under `ConditionalUpdateHardness`, `Width f` and the upstream certificate with
`k < b D`, a cover of `C` by inference blocks has `4 b f · #P ≤ #𝓘 · (k² + 4 b f k)` (Kimi K2 at `f = 4`: about 660).
`Width` is a named premise until it is derived from `f` disjoint paths into each activation. -/
def WidthCoverBound : Prop :=
  ∀ (C : Circuit) {J L : Type} [DecidableEq J] [DecidableEq L] (M : MulBlocks C J L) (A : J → Finset (Fin C.N))
    (s k D f : ℕ) (b : ℝ) (𝓘 : Finset (Finset (Fin C.N))),
    0 < b → Assumptions.ConditionalUpdateHardness M A s b → Assumptions.Width M A s f → M.Upstream A D →
    (k : ℝ) < b * D →
    (∀ I ∈ 𝓘, C.InferenceBlock s k I) → C.Covers 𝓘 →
    4 * b * f * M.P.card ≤ 𝓘.card * (k * k + 4 * b * f * k)

end Nci.Cover.Guarantees
