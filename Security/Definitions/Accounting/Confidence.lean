import Definitions.Accounting.Actual
import Mathlib.Basic.Real.Basic
import Mathlib.Data.Set.Card

/-!
# Runs, strategies and the confidence set (trusted)

The model of `note:console/20261012T0645Z-draft-actual-circuit-model` (v2.2), sections 2 ("Runs"), 4, 5 and 6. A *run*
(`Actual.Run`) is an actual circuit with its gates' values, each computed gate's value its operation on the values it
reads (faithfulness). A gate is seen as its tick, leaf, width and value (`Actual.Run.seen`), so runs of different
circuits compare. The verifier at leaves `V` sees its gates (`view`). A *strategy* maps coins `r : Ω` to a run.

What crosses from one set of leaves to another, read during `I`, is `crossing` (arrival tick, reader's leaf, source as
seen). What a location `X` does during `I` (`acts`) is its gates placed during `I` and what it sends from them (`sent`,
keyed by the source's tick: the sender chooses what crosses, to whom and when). What `X` can tell at `t` (`history`) is
its input gates (`inputs`), its gates placed before `t` (`before`) and what it read from outside before `t`; runs alike
in it are, for `X`, each other's continuations at `t`. The *state over continuations* (`stateOver`) is what `X` must
hold at `t` for every continuation. A strategy is *causal from every tick* (`CausalFrom`) when what `X` does during
`[t, u)` is a function of that state and of what `X` reads from outside during `[t, u)`, and *admissible*
(`Admissible`) when it is a restriction of one that is (fixing coins keeps admissibility, not `CausalFrom`). Memory over
continuations (`MemoryOver`) and links as alphabets (`Alphabet`, by `signal`) are further hypotheses on a strategy.

A guarantee `E` is *sound at `δ`* against the feasible runs `𝓕` (`Sound`) when every admissible strategy with runs in
`𝓕` whose prover's inputs (every input gate off `V`) don't depend on the coins accepts with `¬E` on at most a `δ`
fraction of the coins, uniform over a finite `Ω`. The *confidence set* `𝓒(v)` (`confidence`) is the feasible runs with
view `v` meeting every guarantee. For a quantity `μ` of a run: `req μ 𝓕 P` is the least `μ` of a feasible run that
implements `P`, `Additive` says every run of a set that implements `P` has `μ ≥ cert + req`, and the residual is
`min(cap, obs) − cert`.
-/

namespace Accounting

open Finset

/-- **A run**: an actual circuit and its gates' values, each computed gate's its operation on what it reads. -/
structure Actual.Run (L : Type*) where
  A : Placed L
  val : A.Transcript
  faithful : ∀ g, A.isInput g = false → val g = A.op g val

namespace Actual.Run

variable {L : Type*} (R : Actual.Run L)

/-- A gate as seen from outside the circuit's numbering: its tick, leaf, width and value. -/
def seen (g : Fin R.A.N) : ℕ × L × ℕ × ℕ := (R.A.t g, R.A.x g, R.A.w g, (R.val g).toNat)

/-- **The view** of the verifier at leaves `V`: its gates, their ticks and values. -/
def view (V : Finset L) : Set (ℕ × L × ℕ × ℕ) := {p | ∃ g, R.A.x g ∈ V ∧ p = R.seen g}

/-- The location `X`'s gates placed before `t`, when and with what values. -/
def before (X : Finset L) (t : ℕ) : Set (ℕ × L × ℕ × ℕ) := {p | ∃ g, R.A.x g ∈ X ∧ R.A.t g < t ∧ p = R.seen g}

/-- The input gates placed at leaves of `X`. -/
def inputs (X : Set L) : Set (ℕ × L × ℕ × ℕ) := {p | ∃ g, R.A.isInput g = true ∧ R.A.x g ∈ X ∧ p = R.seen g}

/-- What crosses from the leaves `X` to the leaves `Y`, read during `I`: each wire as its arrival tick, its reader's
leaf and its source as seen. -/
def crossing (X Y : Set L) (I : Set ℕ) : Set (ℕ × L × ℕ × L × ℕ × ℕ) :=
  {p | ∃ g a, a ∈ R.A.reads g ∧ R.A.x a ∈ X ∧ R.A.x g ∈ Y ∧ R.A.t g ∈ I ∧ p = (R.A.t g, R.A.x g, R.seen a)}

/-- What the location `X` sends outside itself from the gates it places during `I`: each wire from such a gate to a
reader outside `X`, as arrival tick, reader's leaf and source as seen. A wire lives in its reader's `reads`, and the
source's tick keys it: the sender chooses what crosses, to whom and when (faithfulness), and a value sent before `t`
that arrives after `t` is not something `X` does from `t`. -/
def sent (X : Set L) (I : Set ℕ) : Set (ℕ × L × ℕ × L × ℕ × ℕ) :=
  {p | ∃ g a, a ∈ R.A.reads g ∧ R.A.x a ∈ X ∧ R.A.x g ∉ X ∧ R.A.t a ∈ I ∧ p = (R.A.t g, R.A.x g, R.seen a)}

/-- What the location `X` does during `I`: its gates placed during `I`, and what it sends from them. -/
def acts (X : Set L) (I : Set ℕ) : Set (ℕ × L × ℕ × ℕ) × Set (ℕ × L × ℕ × L × ℕ × ℕ) :=
  ({p | ∃ g, R.A.x g ∈ X ∧ R.A.t g ∈ I ∧ p = R.seen g}, R.sent X I)

/-- What the location `X` can tell at `t`: its input gates, its gates placed before `t`, and what it read from outside
before `t`, with arrival ticks. Runs alike in it are, for `X`, each other's continuations at `t`. -/
def history (X : Finset L) (t : ℕ) :
    Set (ℕ × L × ℕ × ℕ) × Set (ℕ × L × ℕ × ℕ) × Set (ℕ × L × ℕ × L × ℕ × ℕ) :=
  (R.inputs ↑X, R.before X t, R.crossing (↑X)ᶜ ↑X (Set.Iio t))

/-- The signal link `e` delivers to the leaf `y` at tick `u` (`link` names each wire's link): the sources, as seen, of
the wires on `e` read at `y` at `u`. -/
def signal {E : Type*} (link : Fin R.A.N → Fin R.A.N → E) (e : E) (y : L) (u : ℕ) : Set (ℕ × L × ℕ × ℕ) :=
  {p | ∃ g a, a ∈ R.A.reads g ∧ R.A.x a ≠ R.A.x g ∧ link g a = e ∧ R.A.x g = y ∧ R.A.t g = u ∧ p = R.seen a}

end Actual.Run

open Classical in
/-- **The state over continuations** of `X` at `t` in the run `S r` (v2.2): its gates outside `A[X, [t, ∞)]` (its
inputs and its gates placed before `t`) that `A[X, [t, ∞)]` reads in some continuation of `S r` for `X` at `t` (a run
of `S` with the same `history X t`), matched by how they are seen: what `X` must hold at `t`, not knowing which of it
it will read. With `r' = r` it holds the per-run state. -/
noncomputable def stateOver {Ω L : Type*} (S : Ω → Actual.Run L) (X : Finset L) (t : ℕ) (r : Ω) :
    Finset (Fin (S r).A.N) :=
  univ.filter fun a => (S r).A.x a ∈ X ∧ a ∉ (S r).A.region X (Set.Ici t) ∧
    ∃ r', (S r').history X t = (S r).history X t ∧
      ∃ a' ∈ (S r').A.into ((S r').A.region X (Set.Ici t)), (S r').seen a' = (S r).seen a

/-- **Causality from every tick** (v2.2), in prefix form: in a strategy `S`, what each location `X` does during
`[t, u)` is a function of its state over continuations at `t` and of what it reads from outside during `[t, u)`, with
arrival ticks. -/
def CausalFrom {Ω L : Type*} (S : Ω → Actual.Run L) : Prop :=
  ∀ (X : Finset L) t u r r', (S r).seen '' ↑(stateOver S X t r) = (S r').seen '' ↑(stateOver S X t r') →
    (S r).crossing (↑X)ᶜ ↑X (Set.Ico t u) = (S r').crossing (↑X)ᶜ ↑X (Set.Ico t u) →
    (S r).acts ↑X (Set.Ico t u) = (S r').acts ↑X (Set.Ico t u)

/-- **An admissible strategy** (v2.2): a restriction `S' ∘ ι` of a strategy `S'` causal from every tick, its states
over continuations computed in `S'`. Fixing some of an admissible strategy's coins leaves it admissible; `CausalFrom`
alone doesn't survive that, since fixing coins shrinks the runs a location can't tell apart. -/
def Admissible {Ω L : Type*} (S : Ω → Actual.Run L) : Prop :=
  ∃ (Ω' : Type) (S' : Ω' → Actual.Run L) (ι : Ω → Ω'), CausalFrom S' ∧ ∀ r, S r = S' (ι r)

/-- **Memory over continuations** (v2.2): in every run of `S`, each `X ∈ 𝒳` holds at most `M X` bits of state over
continuations at every tick. -/
def MemoryOver {Ω L : Type*} (S : Ω → Actual.Run L) (𝒳 : Set (Finset L)) (M : Finset L → ℕ) : Prop :=
  ∀ r, ∀ X ∈ 𝒳, ∀ t, (S r).A.width (stateOver S X t r) ≤ M X

/-- **A link is an alphabet** (v2.2), across the runs of `S`: each way, at each tick, link `e` delivers one of at most
`2^{β e}` signals (delivering nothing is one of them). -/
def Alphabet {Ω L E : Type*} (S : Ω → Actual.Run L) (β : E → ℕ) (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → E) :
    Prop :=
  ∀ e y u, (Set.range fun r => (S r).signal (link r) e y u).encard ≤ ((2 ^ β e : ℕ) : ℕ∞)

open Classical in
/-- **`E` is sound at `δ`** against the feasible runs `𝓕`, for the verifier at `V` accepting on `acc` of its view with
coins uniform over `Ω`: every admissible strategy with runs in `𝓕` whose input gates off `V` don't depend on the coins
has `Pr_r[acc ∧ ¬E] ≤ δ`. -/
def Sound {L : Type*} (Ω : Type) [Fintype Ω] (𝓕 : Set (Actual.Run L)) (V : Finset L) (acc : Set (ℕ × L × ℕ × ℕ) → Prop)
    (E : Actual.Run L → Prop) (δ : ℝ) : Prop :=
  ∀ S : Ω → Actual.Run L, (∀ r, S r ∈ 𝓕) → Admissible S → (∀ r r', (S r).inputs (↑V)ᶜ = (S r').inputs (↑V)ᶜ) →
    ((univ.filter fun r => acc ((S r).view V) ∧ ¬E (S r)).card : ℝ) ≤ δ * Fintype.card Ω

/-- **The confidence set** `𝓒(v)`: the feasible runs with the view `v` at `V` that meet every guarantee `E i`. -/
def confidence {L ι : Type*} (𝓕 : Set (Actual.Run L)) (V : Finset L) (E : ι → Actual.Run L → Prop)
    (v : Set (ℕ × L × ℕ × ℕ)) : Set (Actual.Run L) :=
  {R | R ∈ 𝓕 ∧ R.view V = v ∧ ∀ i, E i R}

/-- `req_μ(P)`: the least `μ` of a run of `𝓕` that implements `P` (`impl`), `⊤` when none does. -/
noncomputable def req {L : Type*} (μ : Actual.Run L → ℕ) (𝓕 : Set (Actual.Run L)) (impl : Actual.Run L → Prop) : ℕ∞ :=
  ⨅ (R : Actual.Run L) (_ : R ∈ 𝓕 ∧ impl R), (μ R : ℕ∞)

/-- **`μ` adds** over the runs `𝓒` for `P` above `cert`: each of them that implements `P` has `μ ≥ cert + req_μ(P)`. -/
def Additive {L : Type*} (𝓒 : Set (Actual.Run L)) (μ : Actual.Run L → ℕ) (cert : ℕ) (r : ℕ∞)
    (impl : Actual.Run L → Prop) : Prop :=
  ∀ R ∈ 𝓒, impl R → (cert : ℕ∞) + r ≤ μ R

end Accounting
