import Definitions.Accounting.Actual
import Mathlib.Data.ENat.Lattice
import Mathlib.Basic.Real.Basic
import Mathlib.Data.Set.Card
import Mathlib.Data.Finset.Sort

/-!
# Runs, strategies and the confidence set (trusted)

The model of `note:console/20261012T0645Z-draft-actual-circuit-model` (v2.6), sections 2 ("Runs"), 4, 5 and 6. A *run*
(`Actual.Run`) is an actual circuit with its gates' values, each computed gate's value its operation on the values it
reads (faithfulness), and what each leaf holds at the start of each tick (`held`, constrained by `Placed.Holding`). A
gate is seen as its tick, leaf, cell, width and value (`Actual.Run.seen`), so runs of different circuits compare. The
verifier at leaves `V` sees its gates (`view`). A location's *state* at `t` (`Run.state`, of width `Run.mem`) is what
its leaves hold and what is in flight between them; `Run.Memory` bounds it per run. Its *content* (`Run.content`) is
its values as a bit string, in a fixed order of position (the leaves ordered by `L`'s order, then cell, then tick)
with no position in it.
A *strategy* maps coins `r : Ω` to a run.

What crosses from one set of leaves to another, read during `I`, is `crossing` (arrival tick, reader's leaf, source as
seen). What a location `X` does during `I` (`acts`) is its gates placed during `I`, what it sends from them (`sent`,
keyed by the source's tick: the sender chooses what crosses, to whom and when) and its state at the end of each tick of
`I` (v2.6: what its leaves hold and what is in flight between them). A strategy is *causal from every tick*
(`CausalFrom`) when what `X` does during `[t, u)` is a function of its state's content at `t` and of what it reads from
outside during `[t, u)`; the state is the run's own, so fixing coins keeps it. What `X` can tell at `t` (`history`:
its inputs, its gates before `t` and what it read from outside before `t`) and links as alphabets (`Alphabet`, by
`signal`) are further hypotheses on a strategy.

The *runtime* `t_A(P)` (v2.8, section 7; `Actual.runtime`) reads the strategy too: the least `t` for which some
positions before `t` (leaf of the developer's, cell, tick), the same for every value of `P`'s inputs put in place of
`A`'s, hold `P`'s outputs in the run with that value (`Actual.Computes`).

A guarantee `E` reads a strategy and a coin (a property of the run, or of how the strategy varies with the coins). It
is *sound at `δ`* against the feasible runs `𝓕` (`Sound`) when every strategy causal from every tick with runs in `𝓕`
whose prover's inputs (every input gate off `V`) don't depend on the coins accepts with `¬E` on at most a `δ` fraction
of the coins, uniform over a finite `Ω`. For a quantity `μ` of a run: `req μ 𝓕 P` is the least `μ` of a feasible run
that implements `P`, `Additive` says every run of a set that implements `P` has `μ ≥ cert + req`, and the residual is
`min(cap, obs) − cert`.
-/

namespace Accounting

open Finset

/-- **Holding** (v2.3): `held y t` is the gates the leaf `y` holds at the start of tick `t`. A held gate is `y`'s own
and an input or placed before `t`; `y`'s inputs are held at tick 0; and a gate of `y` that `y` reads at a later tick is
held at `y` at the start of every tick from the one after its placement (from 0 for an input) to the reading one. -/
def Placed.Holding {L : Type*} (A : Placed L) (held : L → ℕ → Finset (Fin A.N)) : Prop :=
  (∀ y t, ∀ a ∈ held y t, A.x a = y ∧ (A.isInput a = true ∨ A.t a < t)) ∧
  (∀ a, A.isInput a = true → a ∈ held (A.x a) 0) ∧
  ∀ g, ∀ a ∈ A.reads g, A.x a = A.x g → ∀ s, (A.isInput a = true ∨ A.t a < s) → s ≤ A.t g → a ∈ held (A.x g) s

/-- **A run**: an actual circuit and its gates' values, each computed gate's its operation on what it reads, and what
each leaf holds at the start of each tick. -/
structure Actual.Run (L : Type*) where
  A : Placed L
  val : A.Transcript
  faithful : ∀ g, A.isInput g = false → val g = A.op g val
  held : L → ℕ → Finset (Fin A.N)
  holding : A.Holding held

/-- A value's `w` bits, least significant first. -/
def Actual.bits (w v : ℕ) : List Bool := List.ofFn fun i : Fin w => v.testBit i

/-- A gate as seen, keyed by position: leaf, then cell, then tick (v2.5), its width and value carried after them. In
one run no two gates share a leaf, cell and tick (`Placed.cell_inj`), so the width and value never decide the order. -/
def Actual.position {L : Type*} (p : ℕ × L × ℕ × ℕ × ℕ) : L ×ₗ ℕ ×ₗ ℕ ×ₗ ℕ ×ₗ ℕ :=
  toLex (p.2.1, toLex (p.2.2.1, toLex (p.1, toLex (p.2.2.2.1, p.2.2.2.2))))

/-- The bits of a gate keyed by position: its value at its width. -/
def Actual.chunk {L : Type*} (p : L ×ₗ ℕ ×ₗ ℕ ×ₗ ℕ ×ₗ ℕ) : List Bool :=
  Actual.bits (ofLex (ofLex (ofLex (ofLex p).2).2).2).1 (ofLex (ofLex (ofLex (ofLex p).2).2).2).2

namespace Actual.Run

variable {L : Type*} (R : Actual.Run L)

/-- A gate as seen from outside the circuit's numbering: its tick, leaf, cell, width and value. -/
def seen (g : Fin R.A.N) : ℕ × L × ℕ × ℕ × ℕ := (R.A.t g, R.A.x g, R.A.cell g, R.A.w g, (R.val g).toNat)

/-- **The view** of the verifier at leaves `V`: its gates, their ticks and values. -/
def view (V : Finset L) : Set (ℕ × L × ℕ × ℕ × ℕ) := {p | ∃ g, R.A.x g ∈ V ∧ p = R.seen g}

/-- The location `X`'s gates placed before `t`, when and with what values. -/
def before (X : Finset L) (t : ℕ) : Set (ℕ × L × ℕ × ℕ × ℕ) := {p | ∃ g, R.A.x g ∈ X ∧ R.A.t g < t ∧ p = R.seen g}

/-- The input gates placed at leaves of `X`. -/
def inputs (X : Set L) : Set (ℕ × L × ℕ × ℕ × ℕ) := {p | ∃ g, R.A.isInput g = true ∧ R.A.x g ∈ X ∧ p = R.seen g}

open Classical in
/-- **The state** of `X` at `t` (v2.3): what its leaves hold at the start of `t`, and what is in flight between them,
the gates of `X` outside `A[X, [t, ∞)]` that `A[X, [t, ∞)]` reads. -/
noncomputable def state (X : Finset L) (t : ℕ) : Finset (Fin R.A.N) :=
  univ.filter fun a => (∃ y ∈ X, a ∈ R.held y t) ∨ (a ∈ R.A.into (R.A.region X (Set.Ici t)) ∧ R.A.x a ∈ X)

/-- **The content** of `X`'s state at `t` (v2.5): its values as a bit string, the state's gates as seen in order of
position (leaf, then cell, then tick; a gate in flight at its source's) and their values concatenated at their widths.
It says what `X` holds and not where: one bit held at any of `2^k` places is the same one-bit string. A coin hidden in
which cell holds a bit is excluded unless every cell that matters is held, and then it is counted in `mem`. -/
noncomputable def content [LinearOrder L] (X : Finset L) (t : ℕ) : List Bool :=
  ((((R.state X t).image R.seen).image Actual.position).sort).flatMap Actual.chunk

/-- `mem X t`: the width of `X`'s state at `t`. -/
noncomputable def mem (X : Finset L) (t : ℕ) : ℕ := R.A.width (R.state X t)

/-- **Memory** (v2.3), per run: each `X ∈ 𝒳` has a state of at most `M X` bits at every tick. -/
def Memory (𝒳 : Set (Finset L)) (M : Finset L → ℕ) : Prop := ∀ X ∈ 𝒳, ∀ t, R.mem X t ≤ M X

/-- What crosses from the leaves `X` to the leaves `Y`, read during `I`: each wire as its arrival tick, its reader's
leaf and its source as seen. -/
def crossing (X Y : Set L) (I : Set ℕ) : Set (ℕ × L × ℕ × L × ℕ × ℕ × ℕ) :=
  {p | ∃ g a, a ∈ R.A.reads g ∧ R.A.x a ∈ X ∧ R.A.x g ∈ Y ∧ R.A.t g ∈ I ∧ p = (R.A.t g, R.A.x g, R.seen a)}

/-- What the location `X` sends outside itself from the gates it places during `I`: each wire from such a gate to a
reader outside `X`, as arrival tick, reader's leaf and source as seen. A wire lives in its reader's `reads`, and the
source's tick keys it: the sender chooses what crosses, to whom and when (faithfulness), and a value sent before `t`
that arrives after `t` is not something `X` does from `t`. -/
def sent (X : Set L) (I : Set ℕ) : Set (ℕ × L × ℕ × L × ℕ × ℕ × ℕ) :=
  {p | ∃ g a, a ∈ R.A.reads g ∧ R.A.x a ∈ X ∧ R.A.x g ∉ X ∧ R.A.t a ∈ I ∧ p = (R.A.t g, R.A.x g, R.seen a)}

/-- What the location `X` does during `I`: its gates placed during `I`, what it sends from them, and its state at the
end of each tick `v` of `I` (at the start of `v + 1`; v2.6), what its leaves hold and what is in flight between them. A
value in flight is in the state only while a later read of `X` awaits it, so keeping it is something `X` does. -/
def acts (X : Finset L) (I : Set ℕ) :
    Set (ℕ × L × ℕ × ℕ × ℕ) × Set (ℕ × L × ℕ × L × ℕ × ℕ × ℕ) × Set (ℕ × ℕ × L × ℕ × ℕ × ℕ) :=
  ({p | ∃ g, R.A.x g ∈ X ∧ R.A.t g ∈ I ∧ p = R.seen g}, R.sent ↑X I,
    {p | ∃ v a, v ∈ I ∧ a ∈ R.state X (v + 1) ∧ p = (v, R.seen a)})

/-- What the location `X` can tell at `t`: its input gates, its gates placed before `t`, and what it read from outside
before `t`, with arrival ticks. -/
def history (X : Finset L) (t : ℕ) :
    Set (ℕ × L × ℕ × ℕ × ℕ) × Set (ℕ × L × ℕ × ℕ × ℕ) × Set (ℕ × L × ℕ × L × ℕ × ℕ × ℕ) :=
  (R.inputs ↑X, R.before X t, R.crossing (↑X)ᶜ ↑X (Set.Iio t))

/-- The signal link `e` delivers to the leaf `y` at tick `u` (`link` names each wire's link): the sources, as seen, of
the wires on `e` read at `y` at `u`. -/
def signal {E : Type*} (link : Fin R.A.N → Fin R.A.N → E) (e : E) (y : L) (u : ℕ) : Set (ℕ × L × ℕ × ℕ × ℕ) :=
  {p | ∃ g a, a ∈ R.A.reads g ∧ R.A.x a ≠ R.A.x g ∧ link g a = e ∧ R.A.x g = y ∧ R.A.t g = u ∧ p = R.seen a}

end Actual.Run

/-- **Causality from every tick** (v2.4), in prefix form: in a strategy `S`, what each location `X` does during
`[t, u)` is a function of its state's content at `t` (its values in order of position, not where they are held) and of
what it reads from outside during `[t, u)`, with arrival ticks. The state is the run's own, so a restriction of a
strategy causal from every tick is too. -/
def CausalFrom {Ω L : Type*} [LinearOrder L] (S : Ω → Actual.Run L) : Prop :=
  ∀ (X : Finset L) t u r r', (S r).content X t = (S r').content X t →
    (S r).crossing (↑X)ᶜ ↑X (Set.Ico t u) = (S r').crossing (↑X)ᶜ ↑X (Set.Ico t u) →
    (S r).acts X (Set.Ico t u) = (S r').acts X (Set.Ico t u)

/-- **The developer at `X` holds `P`'s outputs at fixed positions before `t`** (v2.8), across the runs `S v`, one for
each value `v` of `P`'s inputs (`P.eval v` reads only them): `S v` is the strategy's run with `v` in place of `A`'s
values at `P`'s inputs, the other inputs and the coins the same, which this takes as given. Each output `o` of `P` has a
position, a leaf `y` of `X`, a cell `c` and a tick `u < t`, chosen before `v`, at which every run `S v` places a gate
with `o`'s width and its value at `v`. With the position chosen after `v`, a one-bit output is among almost any run's
values (v2.7's form, a runtime of about 0). -/
def Actual.Computes {L : Type*} {P : Verity.Circuit} (S : P.Transcript → Actual.Run L) (X : Finset L) (t : ℕ) :
    Prop :=
  ∀ o ∈ P.outputs, ∃ y ∈ X, ∃ c u, u < t ∧ ∀ v, ∃ g, (S v).A.x g = y ∧ (S v).A.cell g = c ∧ (S v).A.t g = u ∧
    (S v).A.w g = P.w o ∧ ((S v).val g).toNat = (P.eval v o).toNat

/-- **The runtime** `t_A(P)` (v2.8) of a circuit `P` at the developer's leaves `X`, read off the strategy: the least `t`
by which some positions before `t` in `X`, the same for every value of `P`'s inputs, hold `P`'s outputs in every run
`S v` (`Actual.Computes`); `⊤` when no `t` does. One run holds only the branches its data took, so no one run says
this. -/
noncomputable def Actual.runtime {L : Type*} {P : Verity.Circuit} (S : P.Transcript → Actual.Run L) (X : Finset L) :
    ℕ∞ :=
  ⨅ (t : ℕ) (_ : Actual.Computes S X t), (t : ℕ∞)

/-- **A link is an alphabet** (v2.2), across the runs of `S`: each way, at each tick, link `e` delivers one of at most
`2^{β e}` signals (delivering nothing is one of them). -/
def Alphabet {Ω L E : Type*} (S : Ω → Actual.Run L) (β : E → ℕ) (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → E) :
    Prop :=
  ∀ e y u, (Set.range fun r => (S r).signal (link r) e y u).encard ≤ ((2 ^ β e : ℕ) : ℕ∞)

open Classical in
/-- **`E` is sound at `δ`** against the feasible runs `𝓕`, for the verifier at `V` accepting on `acc` of its view with
coins uniform over `Ω`: every strategy causal from every tick with runs in `𝓕` whose input gates off `V` don't depend on
the coins has `Pr_r[acc ∧ ¬E] ≤ δ`. `E` reads the strategy and the coin. -/
def Sound {L : Type*} [LinearOrder L] (Ω : Type) [Fintype Ω] (𝓕 : Set (Actual.Run L)) (V : Finset L)
    (acc : Set (ℕ × L × ℕ × ℕ × ℕ) → Prop) (E : (Ω → Actual.Run L) → Ω → Prop) (δ : ℝ) : Prop :=
  ∀ S : Ω → Actual.Run L, (∀ r, S r ∈ 𝓕) → CausalFrom S → (∀ r r', (S r).inputs (↑V)ᶜ = (S r').inputs (↑V)ᶜ) →
    ((univ.filter fun r => acc ((S r).view V) ∧ ¬E S r).card : ℝ) ≤ δ * Fintype.card Ω

/-- `req_μ(P)`: the least `μ` of a run of `𝓕` that implements `P` (`impl`), `⊤` when none does. -/
noncomputable def req {L : Type*} (μ : Actual.Run L → ℕ) (𝓕 : Set (Actual.Run L)) (impl : Actual.Run L → Prop) : ℕ∞ :=
  ⨅ (R : Actual.Run L) (_ : R ∈ 𝓕 ∧ impl R), (μ R : ℕ∞)

/-- **`μ` adds** over the runs `𝓒` for `P` above `cert`: each of them that implements `P` has `μ ≥ cert + req_μ(P)`. -/
def Additive {L : Type*} (𝓒 : Set (Actual.Run L)) (μ : Actual.Run L → ℕ) (cert : ℕ) (r : ℕ∞)
    (impl : Actual.Run L → Prop) : Prop :=
  ∀ R ∈ 𝓒, impl R → (cert : ℕ∞) + r ≤ μ R

end Accounting
