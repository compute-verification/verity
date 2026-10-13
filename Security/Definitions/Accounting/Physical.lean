import Definitions.Accounting.Confidence

/-!
# Physical assumptions (trusted)

The model of `note:console/20261012T0645Z-draft-actual-circuit-model` (v2.1, `TrustedUnits` v2.8, `SequentialSpeed`
and `Links` v2.7), sections 1 and 3.
A physical assumption is a constraint on the actual circuit `A` of one run, a `Prop` over `Placed` taken as a hypothesis
by what needs it, never a global assumption; `TrustedUnits` is one on a strategy, a map from coins to runs, since what a
certifier sends is a function of its record across runs. The *feasible set* is the actual circuits that meet the
constraints on one run (`Feasible`); each guarantee names the ones it uses. Each that fails frees a different covert
resource:

| Constraint | What it says | What its failure frees |
| --- | --- | --- |
| `Causal ℓ` | a gate reads `a` at least `ℓ (x a) (x g)` ticks after `a` | answers before questions |
| `ComputeRate 𝒳 π Φ` | `π(A[X, [a, b))) ≤ Φ_X (b − a)` | compute faster than declared |
| `SequentialSpeed ring D span` | a path through more than `D` computed gates at the location of the ring's multiplications spans more than `span` (`Δ + RTT`) ticks | answering faster than `D` sequential multiplications modulo `p` |
| `Memory 𝒳 M` | `mem_X(t) ≤ M_X` | hidden storage |
| `Links ends β link` | every wire between two leaves has a computed source and runs on a declared link joining them; each way, link `e` delivers at most `β_e (b − a)` bits read during `[a, b)` (v2.7) | side channels, hidden bandwidth |
| `TrustedUnits S 𝒱 cert …` | a strategy `S`: gates in `𝒱`, the auditor's devices on machines the prover doesn't control, are the protocol's; what each certifier sends outside itself during `[a, b)` is a function of its own record before `b` and of its own jitter word, one of `χ` classes a bucket (v2.8) | forged records, timing channels |

The hardware is a tree of *locations* `𝒳` (a set of `Finset L`) and *links* `e : E` (a finite type), each joining two
leaves `ends e` with capacity `β_e` bits a tick each way. A link is *certified* when a certifier sits on it, at one of its ends
(`cert`). `β_X` (`linkCapacity`) is the total capacity of the links across `X`'s boundary (`boundary`), `β°_X`
(`uncertifiedCapacity`) that of the uncertified ones, and a *spatial unit* is a location with `β°_X = 0`
(`Hardware.SpatialUnit`). `in_X(I), out_X(I) ≤ β_X |I|` is then a lemma (`Accounting.SecurityProofs.Actual.ingress_le`,
`egress_le`). Compute `Φ_X` (`compute`) and memory `M_X` are numbers per location, in work and bits.

Four constraints read what the circuit doesn't carry, taken as parameters: which link each wire runs on (`link`), the
location of the ring's multiplications modulo `p` (`ring`), the certifiers' leaves (`cert`) and record cells
(`record`, v2.6: a certifier's record is its gates in them, `Actual.Run.record`), the trusted devices' leaves (`𝒱`) with
what the protocol says each of their gates does (`Spec`), and the buckets of ticks over which a certifier's jitter is
counted. `Hardware` bundles the parameters that don't depend on `A`, and `Annotation A` those of the constraints on one
run that name its gates; `TrustedUnits` takes each run's `Spec`.
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

/-- **Memory**: each location `X` of `𝒳` carries at most `M X` bits across every tick. -/
def Memory (𝒳 : Set (Finset L)) (M : Finset L → ℕ) : Prop :=
  ∀ X ∈ 𝒳, ∀ t, A.state X t ≤ M X

open Classical in
/-- The bits link `e` carries to the leaf `y`, read during `I`: the gates whose wire to a reader placed at `y` during `I`,
from another leaf, runs on `e`, which `link` names for each wire. -/
noncomputable def readOn {E : Type*} (link : Fin A.N → Fin A.N → E) (e : E) (y : L) (I : Set ℕ) :
    Finset (Fin A.N) :=
  univ.filter fun a => ∃ g, a ∈ A.reads g ∧ A.x a ≠ A.x g ∧ link g a = e ∧ A.x g = y ∧ A.t g ∈ I

/-- **Links**: every wire between two leaves has a computed source and runs on a declared link joining them (`link g a`,
for `g` reading `a`, with ends `ends`), and each way (to each of its ends `y`) each link `e` delivers at most `β e` bits
a tick, counted at the readers' ticks. -/
def Links {E : Type*} (ends : E → L × L) (β : E → ℕ) (link : Fin A.N → Fin A.N → E) : Prop :=
  (∀ g, ∀ a ∈ A.reads g, A.x a ≠ A.x g →
    A.isInput a = false ∧ (ends (link g a) = (A.x a, A.x g) ∨ ends (link g a) = (A.x g, A.x a))) ∧
  ∀ e y a b, A.width (A.readOn link e y (Set.Ico a b)) ≤ β e * (b - a)

/-- What a protocol says the gates of its trusted devices do: the gates each reads and how it computes its value. -/
structure Spec where
  reads : Fin A.N → Finset (Fin A.N)
  op : (g : Fin A.N) → ((a : Fin A.N) → BitVec (A.w a)) → BitVec (A.w g)

end Placed

open Classical in
/-- **The certifiers' record** in a run (v2.6): the gates at the certifiers' leaves `cert` in the record cells each
declares (`cells y` at the leaf `y`, `Hardware.record`). A certifier's record is what it holds, not a set chosen per
run, so the record placed before `t` is part of what the certifiers can tell at `t` (`history`). -/
noncomputable def Actual.Run.record {L : Type*} (R : Actual.Run L) (cert : Finset L) (cells : L → Set ℕ) :
    Finset (Fin R.A.N) :=
  univ.filter fun g => R.A.x g ∈ cert ∧ R.A.cell g ∈ cells (R.A.x g)

/-- **Trusted devices** (v2.8), across a strategy `S`: in every run, every computed gate placed in `𝒱` reads what the
protocol (`spec r`) says and computes as it says, and the certifiers' leaves `cert` lie in `𝒱` (W1); and for some jitter
words `J`, one per certifier, each one of `χ = 2^jitter` classes a bucket of ticks (`bucket`) in each run, what each
certifier `y` sends outside itself from the gates it places during `[a, b)`, values and ticks (`sent`), is a function of
its own record placed before `b` (its gates in its record cells `record y`, `Actual.Run.record`, as seen, so whatever
timing it logs is counted there; a send during `[a, b)` can't read a later record) and of its own jitter word on the
buckets of `[a, b)` (W2 and J). What a certifier reads is unconstrained.

Each trusted device, every certifier among them, is the auditor's hardware and runs on a machine the prover doesn't
control: the auditor's own VM or box, under the auditor's account, never a process on the prover's node. A device on a
node the prover controls, such as the bandwidth audit's challenger on the GPU's own node, is the prover's hardware,
and its leaves aren't in `𝒱`. -/
def TrustedUnits {Ω L : Type*} (S : Ω → Actual.Run L) (𝒱 cert : Finset L) (spec : ∀ r, (S r).A.Spec)
    (record : L → Set ℕ) (bucket : ℕ → ℕ) (jitter : ℕ) : Prop :=
  cert ⊆ 𝒱 ∧
  (∀ r g, (S r).A.x g ∈ 𝒱 → (S r).A.isInput g = false →
    (S r).A.reads g = (spec r).reads g ∧ (S r).A.op g = (spec r).op g) ∧
  ∃ J : Ω → L → ℕ → Fin (2 ^ jitter), ∀ y ∈ cert, ∀ a b r r',
    (S r).seen '' ↑(((S r).record {y} record).filter fun g => (S r).A.t g < b) =
      (S r').seen '' ↑(((S r').record {y} record).filter fun g => (S r').A.t g < b) →
    (∀ t ∈ Set.Ico a b, J r y (bucket t) = J r' y (bucket t)) →
    (S r).sent {y} (Set.Ico a b) = (S r').sent {y} (Set.Ico a b)

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
