module

public import Definitions.Pouw.Confidence

/-! The actual-circuit model's physical assumptions, each a `Prop` on one run (`TrustedUnits` on a strategy) taken as a hypothesis by what needs it, never a global assumption. -/

@[expose] public section

namespace ResourceAccounting

open Verity Finset

namespace Placed

variable {L : Type*} (A : Placed L)

def IsPath {k : ℕ} (p : Fin (k + 1) → Fin A.N) : Prop :=
  ∀ i : Fin k, p i.castSucc ∈ A.reads (p i.succ)

/-- **Causality** at latency `ℓ`: a gate placed at `y` reads a gate placed at `x` at least `ℓ x y` ticks after it. -/
def Causal (ℓ : L → L → ℕ) : Prop :=
  ∀ g, ∀ a ∈ A.reads g, A.t a + ℓ (A.x a) (A.x g) ≤ A.t g

/-- **A latency ceiling** (`latency-ceiling`). `Causal` bounds latency from below only, so without a ceiling a value
sent before a tick can be read at any later one. -/
def LatencyAtMost (X Y : Finset L) (ℓmax : ℕ) : Prop :=
  ∀ g, ∀ a ∈ A.reads g, A.x a ∈ X → A.x g ∈ Y → A.t g ≤ A.t a + ℓmax

/-- **Compute rate**: each location `X` of `𝒳` does at most `compute X` (`Φ_X`) work a tick, at the pricing `π`. -/
def ComputeRate (𝒳 : Set (Finset L)) (π : Fin A.N → ℕ) (compute : Finset L → ℕ) : Prop :=
  ∀ X ∈ 𝒳, ∀ a b, A.work π (A.region X (Set.Ico a b)) ≤ compute X * (b - a)

open Classical in
/-- **Sequential speed**. The ring's `add` and `one` sit at a location of their own and no other gate computes modulo `p`
(the ring model's idealization). -/
def SequentialSpeed (ring : Finset L) (D span : ℕ) : Prop :=
  ∀ k (p : Fin (k + 1) → Fin A.N), A.IsPath p →
    D < (univ.filter fun i => A.isInput (p i) = false ∧ A.x (p i) ∈ ring).card →
    A.t (p 0) + span < A.t (p (Fin.last k))

/-- **Memory**: each location `X` of `𝒳` carries at most `M X` bits across every tick. -/
def Memory (𝒳 : Set (Finset L)) (M : Finset L → ℕ) : Prop :=
  ∀ X ∈ 𝒳, ∀ t, A.state X t ≤ M X

def MemoryAt (𝒳 : Set (Finset L)) (M : Finset L → ℕ) (T : Set ℕ) : Prop :=
  ∀ X ∈ 𝒳, ∀ t ∈ T, A.state X t ≤ M X

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

/-- **A causal strategy**: causal from every tick (`CausalFrom`, v2.4: what each location does during an interval is a
function of its state's content at its start and of what it reads from outside meanwhile), and each run causal at the
latencies `ℓ` (`Placed.Causal`, v2.1). One physical premise, that answers never precede their questions, across the
coins and in every run. -/
def CausalStrategy {Ω L : Type*} [LinearOrder L] (S : Ω → Actual.Run L) (ℓ : L → L → ℕ) : Prop :=
  CausalFrom S ∧ ∀ r, (S r).A.Causal ℓ

/-- **The links as declared**: across the strategy each link delivers one of at most `2^{β e}` signals a tick each way
(`Alphabet`, v2.2), and in every run each wire between two leaves has a computed source and runs on a declared link
joining them, within its capacity (`Placed.Links`, v2.7). One physical premise, that nothing crosses between locations
but the declared links, at their capacities. -/
def DeclaredLinks {Ω L E : Type*} (S : Ω → Actual.Run L) (ends : E → L × L) (β : E → ℕ)
    (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → E) : Prop :=
  Alphabet S β link ∧ ∀ r, (S r).A.Links ends β (link r)

/-- **The strategy varies only `X`'s secrets** (v2.8): `X` is a location apart from the certifiers `cert`, and what the
certifiers and the outside `(X ∪ cert)ᶜ` can tell at `a` (`history`: their inputs, their gates placed before `a` and
what they read from outside before `a`) is fixed across the coins; the certifiers' record placed before `a` is part of
the former. One physical premise, that the coins reach the window only through `X`'s secret inputs. -/
def Secrets {Ω L : Type*} [LinearOrder L] [Fintype L] (S : Ω → Actual.Run L) (cert X : Finset L) (a : ℕ) : Prop :=
  Disjoint X cert ∧ ∀ r r', (S r).history cert a = (S r').history cert a ∧
    (S r).history (X ∪ cert)ᶜ a = (S r').history (X ∪ cert)ᶜ a

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
noncomputable def uncertifiedLinkCapacity (ends : E → L × L) (β : E → ℕ) (cert X : Finset L) : ℕ :=
  ∑ e ∈ (boundary ends X).filter (fun e => (ends e).1 ∉ cert ∧ (ends e).2 ∉ cert), β e

/-- **Coverage** (the listed claim `certifier-coverage`): every boundary link of `X` is certified, `β°_X = 0`
(`uncertifiedLinkCapacity`), so nothing leaves `X` but through a certifier. One physical premise, the auditor's placement
in the onsite protocol; `Hardware.SpatialUnit` is it at a `Hardware`'s parameters. -/
def Covered (ends : E → L × L) (β : E → ℕ) (cert X : Finset L) : Prop :=
  uncertifiedLinkCapacity ends β cert X = 0

end Links

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

def Hardware.SpatialUnit {L E : Type*} [Fintype E] (H : Hardware L E) (X : Finset L) : Prop :=
  X ∈ H.units ∧ uncertifiedLinkCapacity H.ends H.β H.cert X = 0

structure Usage {L : Type*} (A : Placed L) (E : Type*) where
  π : Fin A.N → ℕ
  link : Fin A.N → Fin A.N → E

def Feasible {L E : Type*} (H : Hardware L E) (A : Placed L) (usage : Usage A E) : Prop :=
  A.Causal H.ℓ ∧ A.ComputeRate H.units usage.π H.compute ∧ A.SequentialSpeed H.ring H.D H.span ∧
    A.Memory H.units H.M ∧ A.Links H.ends H.β usage.link

end ResourceAccounting
