import Definitions.Accounting.Actual

/-!
# Physical assumptions (trusted)

The model of `note:console/20261012T0645Z-draft-actual-circuit-model` (v2), sections 1 and 3. A physical assumption is
a constraint on the actual circuit `A` of one run, a `Prop` over `Placed` taken as a hypothesis by what needs it, never
a global assumption. The *feasible set* is the actual circuits that meet all of them (`Feasible`); each guarantee names
the ones it uses. Each that fails frees a different covert resource:

| Constraint | On `A` | What its failure frees |
| --- | --- | --- |
| `Causal ℓ` | a gate reads `a` at least `ℓ (x a) (x g)` ticks after `a` | answers before questions |
| `ComputeRate 𝒳 π Φ` | `π(A[X, [a, b))) ≤ Φ_X (b − a)` | compute faster than declared |
| `SequentialSpeed root t_root` | a path that computes a P2 root spans at least `t_root` ticks | rebuilding a block in time |
| `Memory 𝒳 M` | `mem_X(t) ≤ M_X` | hidden storage |
| `Links ends β link` | every wire between two leaves runs on a declared link joining them; link `e` carries at most `β_e (b − a)` bits read during `[a, b)`, and as many sent | side channels, hidden bandwidth |
| `TrustedUnits 𝒱 cert …` | gates in `𝒱` are the protocol's; a certifier reads at most `log₂ D` bits a bucket beyond its record | forged records, timing channels |

The hardware is a tree of *locations* `𝒳` (a set of `Finset L`) and *links* `e : E` (a finite type), each joining two
leaves `ends e` with capacity `β_e` bits a tick. A link is *certified* when a certifier sits on it, at one of its ends
(`cert`). `β_X` (`linkCapacity`) is the total capacity of the links across `X`'s boundary (`boundary`), `β°_X`
(`uncertifiedCapacity`) that of the uncertified ones, and a *spatial unit* is a location with `β°_X = 0`
(`Hardware.SpatialUnit`). `in_X(I), out_X(I) ≤ β_X |I|` is then a lemma (`Accounting.SecurityProofs.Actual.ingress_le`,
`egress_le`). Compute `Φ_X` (`compute`) and memory `M_X` are numbers per location, in work and bits.

Four constraints read what the circuit doesn't carry, taken as parameters: which link each wire runs on (`link`), which
paths of `A` compute a P2 root (`root`, a predicate on paths), the certifiers' leaves (`cert`) and records (`record`),
the trusted devices' leaves (`𝒱`) with what the protocol says each of their gates does (`Spec`), and the buckets of
ticks over which a certifier's jitter is counted. `Hardware` bundles the parameters that don't depend on `A`, and
`Annotation A` those that name its gates.
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

/-- **Compute rate**: each location `X` of `𝒳` does at most `compute X` (`Φ_X`) work a tick, at the pricing `π`. -/
def ComputeRate (𝒳 : Set (Finset L)) (π : Fin A.N → ℕ) (compute : Finset L → ℕ) : Prop :=
  ∀ X ∈ 𝒳, ∀ a b, A.work π (A.region X (Set.Ico a b)) ≤ compute X * (b - a)

/-- **Sequential speed**: a path of `A` that computes a P2 root (`root k p`, the data of which paths do) spans at least
`tRoot` (`t_root`) ticks. -/
def SequentialSpeed (root : (k : ℕ) → (Fin (k + 1) → Fin A.N) → Prop) (tRoot : ℕ) : Prop :=
  ∀ k (p : Fin (k + 1) → Fin A.N), A.IsPath p → root k p → A.t (p 0) + tRoot ≤ A.t (p (Fin.last k))

/-- **Memory**: each location `X` of `𝒳` carries at most `M X` bits across every tick. -/
def Memory (𝒳 : Set (Finset L)) (M : Finset L → ℕ) : Prop :=
  ∀ X ∈ 𝒳, ∀ t, A.state X t ≤ M X

open Classical in
/-- The bits link `e` carries read during `I`: the gates whose wire to a reader placed at another leaf during `I` runs on
`e`, which `link` names for each wire. -/
noncomputable def readOn {E : Type*} (link : Fin A.N → Fin A.N → E) (e : E) (I : Set ℕ) : Finset (Fin A.N) :=
  univ.filter fun a => ∃ g, a ∈ A.reads g ∧ A.x a ≠ A.x g ∧ link g a = e ∧ A.t g ∈ I

open Classical in
/-- The bits link `e` carries sent during `I`: the gates placed during `I` whose wire to a reader placed at another leaf
runs on `e`. -/
noncomputable def sentOn {E : Type*} (link : Fin A.N → Fin A.N → E) (e : E) (I : Set ℕ) : Finset (Fin A.N) :=
  univ.filter fun a => A.t a ∈ I ∧ ∃ g, a ∈ A.reads g ∧ A.x a ≠ A.x g ∧ link g a = e

/-- **Links**: every wire between two leaves runs on a declared link joining them (`link g a`, for `g` reading `a`, with
ends `ends`), and each link `e` carries at most `β e` bits a tick, counted at the readers' ticks and at the senders'. -/
def Links {E : Type*} (ends : E → L × L) (β : E → ℕ) (link : Fin A.N → Fin A.N → E) : Prop :=
  (∀ g, ∀ a ∈ A.reads g, A.x a ≠ A.x g →
    ends (link g a) = (A.x a, A.x g) ∨ ends (link g a) = (A.x g, A.x a)) ∧
  ∀ e a b, A.width (A.readOn link e (Set.Ico a b)) ≤ β e * (b - a) ∧
    A.width (A.sentOn link e (Set.Ico a b)) ≤ β e * (b - a)

/-- What a protocol says the gates of its trusted devices do: the gates each reads and how it computes its value. -/
structure Spec where
  reads : Fin A.N → Finset (Fin A.N)
  op : (g : Fin A.N) → ((a : Fin A.N) → BitVec (A.w a)) → BitVec (A.w g)

open Classical in
/-- **Trusted devices**: every computed gate placed in `𝒱` reads what the protocol `spec` says and computes as it says;
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

section Links

variable {L E : Type*} [Fintype E]

open Classical in
/-- The links across `X`'s boundary: one end in `X`, the other outside. -/
noncomputable def boundary (ends : E → L × L) (X : Finset L) : Finset E :=
  univ.filter fun e => ((ends e).1 ∈ X ↔ (ends e).2 ∉ X)

/-- `β_X`: the total capacity of the links across `X`'s boundary. -/
noncomputable def linkCapacity (ends : E → L × L) (β : E → ℕ) (X : Finset L) : ℕ :=
  ∑ e ∈ boundary ends X, β e

open Classical in
/-- `β°_X`: the total capacity of the uncertified links across `X`'s boundary, those with no end at a certifier's leaf
(`cert`). -/
noncomputable def uncertifiedCapacity (ends : E → L × L) (β : E → ℕ) (cert X : Finset L) : ℕ :=
  ∑ e ∈ (boundary ends X).filter (fun e => (ends e).1 ∉ cert ∧ (ends e).2 ∉ cert), β e

end Links

/-- The hardware a run's physical assumptions are about: the tree's locations `units`, the latencies, each location's
compute `compute` (`Φ`) and memory `M` (work a tick and bits), the links `E` with their ends and capacities `β` (bits a
tick), the span `tRoot` (`t_root`) of a P2 root's computation, the certifiers' leaves, the trusted devices' leaves, the
buckets of ticks and the jitter bits a bucket. -/
structure Hardware (L : Type*) (E : Type*) where
  units : Set (Finset L)
  ℓ : L → L → ℕ
  compute : Finset L → ℕ
  M : Finset L → ℕ
  ends : E → L × L
  β : E → ℕ
  tRoot : ℕ
  cert : Finset L
  trusted : Finset L
  bucket : ℕ → ℕ
  jitter : ℕ

/-- **A spatial unit**: a location all of whose boundary links are certified, `β°_X = 0`. -/
def Hardware.SpatialUnit {L E : Type*} [Fintype E] (H : Hardware L E) (X : Finset L) : Prop :=
  X ∈ H.units ∧ uncertifiedCapacity H.ends H.β H.cert X = 0

/-- What the physical assumptions read about `A` that its gates don't carry: the pricing `π`, which link each wire runs
on, which paths compute a P2 root, what the protocol says the trusted gates do, and the certifiers' record. -/
structure Annotation {L : Type*} (A : Placed L) (E : Type*) where
  π : Fin A.N → ℕ
  link : Fin A.N → Fin A.N → E
  root : (k : ℕ) → (Fin (k + 1) → Fin A.N) → Prop
  spec : A.Spec
  record : Finset (Fin A.N)

/-- **The feasible set**: `A` meets every physical assumption on the hardware `H`, read with `ann`. -/
def Feasible {L E : Type*} (H : Hardware L E) (A : Placed L) (ann : Annotation A E) : Prop :=
  A.Causal H.ℓ ∧ A.ComputeRate H.units ann.π H.compute ∧ A.SequentialSpeed ann.root H.tRoot ∧
    A.Memory H.units H.M ∧ A.Links H.ends H.β ann.link ∧
    A.TrustedUnits H.trusted H.cert ann.spec ann.record H.bucket H.jitter

end Accounting
