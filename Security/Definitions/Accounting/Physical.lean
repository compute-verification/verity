import Definitions.Accounting.Actual

/-!
# Physical assumptions (trusted)

The model of `note:console/20261012T0645Z-draft-actual-circuit-model`, section 2. A physical assumption is a constraint
on the actual circuit `A` of one run, a `Prop` over `Placed` taken as a hypothesis by what needs it, never a global
assumption. The *feasible set* is the actual circuits that meet all of them (`Feasible`). Each that fails frees a
different covert resource:

| Constraint | On `A` | What its failure frees |
| --- | --- | --- |
| `Causal ℓ` | a gate reads `a` at least `ℓ (x a) (x g)` ticks after `a` | answers before questions |
| `ComputeRate 𝒳 π Π` | `π(A[X, [a, b))) ≤ Π_X (b − a)` | compute faster than declared |
| `SequentialSpeed root τ_root` | a path that computes a P2 root spans at least `τ_root` ticks | rebuilding a block in time |
| `Memory 𝒳 M` | `mem_X(t) ≤ M_X` | hidden storage |
| `Links 𝒳 B` | `in_X([a, b))`, `out_X([a, b)) ≤ B_X (b − a)` | hidden bandwidth |
| `Isolation 𝒳 cert σ` | wires across `X`'s boundary that bypass every certifier carry `≤ σ_X (b − a)` bits | side channels |
| `TrustedUnits 𝒱 cert …` | gates in `𝒱` are the protocol's; a certifier reads at most `log₂ D` bits a bucket beyond its record | forged records, timing channels |

The units `X` range over the tree's units `𝒳` (a set of `Finset L`); the capacities `Π`, `M`, `B`, `σ` are numbers per unit,
in gates, bits and ticks (`compute` is `Π`). Three constraints read what the circuit doesn't carry, taken as parameters: which paths of `A`
compute a P2 root (`root`, a predicate on paths), the certifiers' leaves (`cert`) and records (`record`), the trusted
units' leaves (`𝒱`) with what the protocol says each of their gates does (`Spec`), and the buckets of ticks over which
a certifier's jitter is counted. `Hardware` bundles the parameters that don't depend on `A`, and `Annotation A` those
that name its gates.
-/

namespace Accounting

open Verity Finset

namespace Placed

variable {L : Type*} (A : Placed L)

/-- **A path of `A`** from `p 0` to `p k`: each gate is read by the next. -/
def IsPath {k : ℕ} (p : Fin (k + 1) → Fin A.N) : Prop :=
  ∀ i : Fin k, p i.castSucc ∈ A.reads (p i.succ)

/-- **Causality** at latency `ℓ`: a gate placed at `y` reads a gate placed at `x` at least `ℓ x y` ticks after it. -/
def Causal (ℓ : L → L → ℕ) : Prop :=
  ∀ g, ∀ a ∈ A.reads g, A.t a + ℓ (A.x a) (A.x g) ≤ A.t g

/-- **Compute rate**: each unit `X` of `𝒳` does at most `compute X` (`Π_X`) work a tick, at the pricing `π`. -/
def ComputeRate (𝒳 : Set (Finset L)) (π : Fin A.N → ℕ) (compute : Finset L → ℕ) : Prop :=
  ∀ X ∈ 𝒳, ∀ a b, A.work π (A.region X (Set.Ico a b)) ≤ compute X * (b - a)

/-- **Sequential speed**: a path of `A` that computes a P2 root (`root k p`, the data of which paths do) spans at least
`τroot` ticks. -/
def SequentialSpeed (root : (k : ℕ) → (Fin (k + 1) → Fin A.N) → Prop) (τroot : ℕ) : Prop :=
  ∀ k (p : Fin (k + 1) → Fin A.N), A.IsPath p → root k p → A.t (p 0) + τroot ≤ A.t (p (Fin.last k))

/-- **Memory**: each unit `X` of `𝒳` carries at most `M X` bits across every tick. -/
def Memory (𝒳 : Set (Finset L)) (M : Finset L → ℕ) : Prop :=
  ∀ X ∈ 𝒳, ∀ t, A.state X t ≤ M X

/-- **Links**: at most `B X` bits a tick cross each unit `X` of `𝒳`'s boundary, in and out. -/
def Links (𝒳 : Set (Finset L)) (B : Finset L → ℕ) : Prop :=
  ∀ X ∈ 𝒳, ∀ a b, A.ingress X (Set.Ico a b) ≤ B X * (b - a) ∧ A.egress X (Set.Ico a b) ≤ B X * (b - a)

open Classical in
/-- The bits that cross `X`'s boundary during `I` on wires that bypass every certifier, whose ends are both placed at no
leaf of `cert`: the gates placed outside `X` that such a wire carries into `A[X, I]`, and the gates of `A[X, I]` that such
a wire carries out of `X`. -/
noncomputable def bypass (cert X : Finset L) (I : Set ℕ) : ℕ :=
  A.width ((A.into (A.region X I)).filter fun a =>
      A.x a ∉ X ∧ A.x a ∉ cert ∧ ∃ g ∈ A.region X I, A.x g ∉ cert ∧ a ∈ A.reads g) +
    A.width ((A.region X I).filter fun g => A.x g ∉ cert ∧ ∃ h, A.x h ∉ X ∧ A.x h ∉ cert ∧ g ∈ A.reads h)

/-- **Isolation**: the wires across each unit `X` of `𝒳`'s boundary that bypass every certifier (placed at `cert`) carry
at most `σ X` bits a tick. Perfect isolation is `σ = 0`. -/
def Isolation (𝒳 : Set (Finset L)) (cert : Finset L) (σ : Finset L → ℕ) : Prop :=
  ∀ X ∈ 𝒳, ∀ a b, A.bypass cert X (Set.Ico a b) ≤ σ X * (b - a)

/-- What a protocol says the gates of its trusted units do: the gates each reads and how it computes its value. -/
structure Spec where
  reads : Fin A.N → Finset (Fin A.N)
  op : (g : Fin A.N) → ((a : Fin A.N) → BitVec (A.w a)) → BitVec (A.w g)

open Classical in
/-- **Trusted units**: every computed gate placed in `𝒱` reads what the protocol `spec` says and computes as it says;
the certifiers' leaves `cert` lie in `𝒱`; and in each bucket `k` of ticks (`bucket`), the gates that a certifier's gates
read from outside the certifiers, beyond its record `record`, are at most `jitter` bits wide (`log₂ D`, W2 and J). A
certifier's gates being the protocol's, what it emits is then a function of its record and of those bits. -/
def TrustedUnits (𝒱 cert : Finset L) (spec : A.Spec) (record : Finset (Fin A.N)) (bucket : ℕ → ℕ) (jitter : ℕ) :
    Prop :=
  cert ⊆ 𝒱 ∧
  (∀ g, A.x g ∈ 𝒱 → A.isInput g = false → A.reads g = spec.reads g ∧ A.op g = spec.op g) ∧
  ∀ k, A.width (univ.filter fun a => A.x a ∉ cert ∧ a ∉ record ∧
    ∃ e, A.x e ∈ cert ∧ bucket (A.t e) = k ∧ a ∈ A.reads e) ≤ jitter

end Placed

/-- The hardware a run's physical assumptions are about: the tree's units `𝒳`, the latencies, each unit's compute rate
`compute` (`Π`), memory `M`, link capacity `B` and bypass rate `σ` (gates, bits and bits a tick), the span `τroot` of a P2 root's
computation, the certifiers' leaves, the trusted units' leaves `𝒱`, the buckets of ticks and the jitter bits a bucket. -/
structure Hardware (L : Type*) where
  units : Set (Finset L)
  ℓ : L → L → ℕ
  compute : Finset L → ℕ
  M : Finset L → ℕ
  B : Finset L → ℕ
  σ : Finset L → ℕ
  τroot : ℕ
  cert : Finset L
  trusted : Finset L
  bucket : ℕ → ℕ
  jitter : ℕ

/-- What the physical assumptions read about `A` that its gates don't carry: the pricing `π`, which paths compute a P2
root, what the protocol says the trusted gates do, and the certifiers' record. -/
structure Annotation {L : Type*} (A : Placed L) where
  π : Fin A.N → ℕ
  root : (k : ℕ) → (Fin (k + 1) → Fin A.N) → Prop
  spec : A.Spec
  record : Finset (Fin A.N)

/-- **The feasible set**: `A` meets every physical assumption on the hardware `H`, read with `ann`. -/
def Feasible {L : Type*} (H : Hardware L) (A : Placed L) (ann : Annotation A) : Prop :=
  A.Causal H.ℓ ∧ A.ComputeRate H.units ann.π H.compute ∧ A.SequentialSpeed ann.root H.τroot ∧ A.Memory H.units H.M ∧
    A.Links H.units H.B ∧ A.Isolation H.units H.cert H.σ ∧
    A.TrustedUnits H.trusted H.cert ann.spec ann.record H.bucket H.jitter

end Accounting
