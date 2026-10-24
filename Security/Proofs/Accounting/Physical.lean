import Definitions.Accounting.Physical

/-! The proof-only part of `Definitions.Accounting.Physical`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Accounting

open Verity Finset

namespace Placed

variable {L : Type*} (A : Placed L)

/-- **A path of `A`** from `p 0` to `p k`: each gate is read by the next. -/
def IsPath {k : ℕ} (p : Fin (k + 1) → Fin A.N) : Prop :=
  ∀ i : Fin k, p i.castSucc ∈ A.reads (p i.succ)

open Classical in
/-- **Sequential speed** (v2.7): a path of `A` through more than `D` multiplications modulo `p` spans more than `span`
(`Δ + RTT`) ticks. `ring` is the location of the ring's multiplications modulo `p`, so the count of a path's computed
gates at it is v2.7's count of multiplications: a location is the operation class, not a pricing's. The ring's `add`
and `one` sit at a location of their own and no other gate computes modulo `p` (B2's oracle locations, the ring
model's idealization). -/
def SequentialSpeed (ring : Finset L) (D span : ℕ) : Prop :=
  ∀ k (p : Fin (k + 1) → Fin A.N), A.IsPath p →
    D < (univ.filter fun i => A.isInput (p i) = false ∧ A.x (p i) ∈ ring).card →
    A.t (p 0) + span < A.t (p (Fin.last k))

/-- **Memory at the ticks `T`**: `Memory` read only at the ticks of `T`; `Memory` is it at every tick. -/
def MemoryAt (𝒳 : Set (Finset L)) (M : Finset L → ℕ) (T : Set ℕ) : Prop :=
  ∀ X ∈ 𝒳, ∀ t ∈ T, A.state X t ≤ M X

end Placed

/-- The hardware a run's physical assumptions are about: the tree's locations `units`, the latencies, each location's
compute `compute` (`Φ`) and memory `M` (work a tick and bits), the links `E` with their ends and capacities `β` (bits a
tick), the location `ring` of the ring's multiplications modulo `p` (its `add` and `one` sit elsewhere) with the
sequential depth `D` and the span `Δ + RTT` (`span`) of `SequentialSpeed` (v2.7), the certifiers' leaves, the trusted
devices' leaves, the buckets of ticks, each certifier's jitter bits a bucket, and the record cells `record y` each
certifier's leaf `y` declares (v2.6: its record is its gates in them, `Actual.Run.record`). -/
structure Hardware (L : Type*) (E : Type*) where
  units : Set (Finset L)
  ℓ : L → L → ℕ
  compute : Finset L → ℕ
  M : Finset L → ℕ
  ends : E → L × L
  β : E → ℕ
  ring : Finset L
  D : ℕ
  span : ℕ
  cert : Finset L
  trusted : Finset L
  bucket : ℕ → ℕ
  jitter : ℕ
  record : L → Set ℕ

/-- **A spatial unit**: a location all of whose boundary links are certified, `β°_X = 0`. -/
def Hardware.SpatialUnit {L E : Type*} [Fintype E] (H : Hardware L E) (X : Finset L) : Prop :=
  X ∈ H.units ∧ uncertifiedCapacity H.ends H.β H.cert X = 0

/-- What the physical assumptions on one run read about `A` that its gates don't carry: the pricing `π` and which link
each wire runs on. -/
structure Annotation {L : Type*} (A : Placed L) (E : Type*) where
  π : Fin A.N → ℕ
  link : Fin A.N → Fin A.N → E

/-- **The feasible set**: `A` meets every physical assumption on one run on the hardware `H`, read with `ann`.
`TrustedUnits`, on a strategy, is taken beside it. -/
def Feasible {L E : Type*} (H : Hardware L E) (A : Placed L) (ann : Annotation A E) : Prop :=
  A.Causal H.ℓ ∧ A.ComputeRate H.units ann.π H.compute ∧ A.SequentialSpeed H.ring H.D H.span ∧
    A.Memory H.units H.M ∧ A.Links H.ends H.β ann.link

end Accounting
