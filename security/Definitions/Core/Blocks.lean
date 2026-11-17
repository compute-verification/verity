module

public import Definitions.Core.Circuit
public import Definitions.Registration.Device
public import Definitions.Core.Game
public import Mathlib.Data.Fintype.Pi
public import Mathlib.Data.Fintype.Prod
import Mathlib.Algebra.Order.Module.Field
import Mathlib.Basic.ENNReal.Real

/-! Server-time blocks, each opened by a certified memory wipe, what crosses into each and what the wipe bounds; it
states no guarantee. -/

@[expose] public section

namespace Verity

open Finset
open scoped ENNReal

variable {ι : Type}

/-- **The device's state**: its on-chip state, `chip` bits (S_chip), and its store, `mem` bits (M), the GPU's HBM. The
model has no host store. It is written `Device.State d`: `d` is an `ResourceAccounting.Device`, so `d.State` doesn't resolve. -/
structure Device.State (d : ResourceAccounting.Device ι) where
  /-- the on-chip state: registers, shared memory, caches -/
  chip : Fin d.chip → Bool
  /-- the store: the device memory, the GPU's HBM -/
  store : Fin d.mem → Bool

/-- **A server-time block** of the circuit `C` on the device: the window of ticks it runs after the wipe that opens it,
and the gates of `C` it computes there. -/
structure Block (C : Circuit) where
  win : ResourceAccounting.Window
  gates : Finset (Fin C.N)

/-- **A schedule of `B` server-time blocks** of the circuit `C` on the device: a block ends before every later block
starts, the ticks between them being the wipes', and no gate is computed in two blocks. -/
structure Blocks (C : Circuit) (B : ℕ) where
  block : Fin B → Block C
  ordered : ∀ b k, b < k → (block b).win.start + (block b).win.len ≤ (block k).win.start
  disjoint : ∀ b k, b ≠ k → Disjoint (block b).gates (block k).gates

namespace Blocks

variable {C : Circuit} {B : ℕ} (s : Blocks C B)

/-- The gates computed before block `b`. -/
def before (b : Fin B) : Finset (Fin C.N) := (univ.filter (· < b)).biUnion fun k => (s.block k).gates

/-- The gates computed from block `b` on. -/
def since (b : Fin B) : Finset (Fin C.N) := (univ.filter (b ≤ ·)).biUnion fun k => (s.block k).gates

/-- **The carried state at boundary `b`**, the wipe that opens block `b`: the gates computed before block `b` that a
gate computed from block `b` on reads. A gate computed before block `b` and read only after block `b` crosses each
boundary in between. `C.width (s.carried b)` is its size in bits. -/
def carried (b : Fin B) : Finset (Fin C.N) := (s.before b).filter fun a => ∃ g ∈ s.since b, a ∈ C.reads g

/-- A value of the carried state at boundary `b`: one for each of its gates. -/
abbrev Carried (b : Fin B) : Type := (g : s.carried b) → BitVec (C.w g)

def carriedValue (X : C.Trace) (b : Fin B) : s.Carried b := fun g => X g

end Blocks

/-- The state block `b` holds: the carried value re-read after the wipe, and the device's state. That the re-read
value is the committed one is the commitment's binding, which this type doesn't state. -/
structure Blocks.State {C : Circuit} {B : ℕ} (s : Blocks C B) (d : ResourceAccounting.Device ι) (b : Fin B) where
  carried : s.Carried b
  device : Device.State d

/-- **The wipe at boundary `b`** of the schedule `s` on the device `d`, the streamed-label wipe, with the seed sampled
from `Ω` and the spot reads from `Ch`, each uniform once the device's state at the boundary is fixed: the device fills
`fill` bits of its store with labels derived from the seed, then the memory challenger reveals the spot reads one at a
time and reads the labels back under a deadline.

* `carry σ`: the carried state's value as the device in state `σ` holds it at the boundary. The device commits it before
  the wipe and re-reads it after. The check of what it re-reads against that commitment isn't in `accepts`.
* `filled σ ω`: the device's state when the fill ends, and the egress: the `egress` bits that left the device during the
  wipe, as the network attester counts them. Nothing enters the device during the fill but the seed.
* `accepts σ ω c`: the memory challenger's verdict.
* `after σ' ξ c`: the device's state when block `b` starts, from its state `σ'` when the fill ends, the egress `ξ` and
  the spot reads `c`. During the spot reads nothing enters the device but the spot reads and what the outside sends
  back, a function of the egress and the spot reads. -/
structure Wipe (d : ResourceAccounting.Device ι) {C : Circuit} {B : ℕ} (s : Blocks C B) (b : Fin B) (Ω Ch : Type) where
  /-- the bits of the device's store the fill covers -/
  fill : ℕ
  fill_le : fill ≤ d.mem
  /-- the bits that leave the device during the wipe (`E`) -/
  egress : ℕ
  carry : Device.State d → s.Carried b
  /-- the device's state when the fill ends, and the egress -/
  filled : Device.State d → Ω → Device.State d × (Fin egress → Bool)
  accepts : Device.State d → Ω → Ch → Prop
  /-- the device's state when block `b` starts -/
  after : Device.State d → (Fin egress → Bool) → Ch → Device.State d

namespace Wipe

variable {d : ResourceAccounting.Device ι} {C : Circuit} {B : ℕ} {s : Blocks C B} {b : Fin B} {Ω Ch : Type}
  (w : Wipe d s b Ω Ch)

def start (σ : Device.State d) (ω : Ω) (c : Ch) : s.State d b :=
  ⟨w.carry σ, w.after (w.filled σ ω).1 (w.filled σ ω).2 c⟩

/-- The bits of the device's state the fill leaves out: its on-chip state and the store the fill doesn't reach. -/
def unfilled : ℕ := d.chip + (d.mem - w.fill)

/-- What the device does from block `b` on depends on its boundary state and the seed only through `start`. What
reaches it later (`y`), content the prover kept off the device among it, is not bounded here. -/
def Separates {Y Z : Type} (next : Device.State d → Ω → Ch → Y → Z) : Prop :=
  ∃ g : s.State d b → Ch → Y → Z, ∀ σ ω c y, next σ ω c y = g (w.start σ ω c) c y

/-- **The wipe's survivor bound** at `foreignBits` and `ε`, in the streamed-label wipe's game (`Pous.StreamedWipe`):
for every `S_φ` of at least `foreignBits` bits plus the state the fill leaves out (`unfilled`: the on-chip state and the
store the fill doesn't reach), every way `hold` that the device's state at the boundary holds foreign content `z` of
`S_φ` bits, and every way `keep` of reading it back from the device's state when the fill ends, the egress and the spot
reads, the probability over a uniform `z`, a uniform seed and uniform spot reads that the memory challenger accepts and
`keep` gives back `z` is at most `ε S_φ hold keep`. The probability is the value of a sample with no prover move
(`Game.value` on `Game.draw`). At `foreignBits ≤ f · fill + 2λ` it bounds, with what the block keeps read as the bits
it gives back, the wipe's acceptance while more than `f · fill + 2λ` bits plus the unfilled state survive
(`ResourceAccounting.SecurityProofs.Live.survives_of_survivorBound`).

`z` is foreign content as in PoUS's games: uniform, and independent of the wipe's coins. `keep` reads all that the state
block `b` starts from is a function of, besides the carried value, which, once bound to the committed trace, is
fixed before `z` is sampled. The on-chip state is counted, not assumed wiped: chip bits may survive any wipe.

**The pair is the protocol's.** `foreignBits` and `ε` are what the wipe's protocol publishes with an accepted verdict
(the streamed wipe's verdict publishes `foreign_bits = ⌈fill/19⌉ + E`, `E` the egress), not constants of this
definition. `ε` is a function of the adversary's moves (`S_φ`, `hold`, `keep`), so it can cover the wipe's own error,
the labels' PRF advantage against the distinguisher those moves make (the standard-model term, as
`Pous.Guarantees.StreamedWipe` adds `Pous.StreamedWipe.prfAdvantage`), and `δ_net` once the network attester's count of
the egress enters. A constant `ε` reads the labels as ideal. `Pous.Guarantees.StreamedWipe` is the streamed game's
proved bound. The bridge from an accepted verdict to this bound is memory-accounting's, in a package that sees both. -/
def SurvivorBound [Fintype Ω] [Nonempty Ω] [Fintype Ch] [Nonempty Ch] (foreignBits : ℕ)
    (ε : (S_φ : ℕ) → ((Fin S_φ → Bool) → Device.State d) →
      (Device.State d → (Fin w.egress → Bool) → Ch → Fin S_φ → Bool) → ℝ≥0∞) : Prop :=
  ∀ S_φ : ℕ, foreignBits + w.unfilled ≤ S_φ →
    ∀ (hold : (Fin S_φ → Bool) → Device.State d)
      (keep : Device.State d → (Fin w.egress → Bool) → Ch → Fin S_φ → Bool),
      Game.value
        (fun x : (Fin S_φ → Bool) × Ω × Ch =>
          w.accepts (hold x.1) x.2.1 x.2.2 ∧
            keep (w.filled (hold x.1) x.2.1).1 (w.filled (hold x.1) x.2.1).2 x.2.2 = x.1)
        (Game.draw _) ≤ ε S_φ hold keep

/-- **A run of the game `G` at the wipe's boundary, as the wipe's game**: a view of the run's outcome as the wipe's
game's, the foreign content of `S_φ` bits the device holds at the boundary, the seed and the spot reads, with the moves
of the run's strategy there: `hold`, the device's state at the boundary from the foreign content it holds, and `keep`,
how block `b` reads that content back from the device's state when the fill ends, the egress and the spot reads. On the
run, the wipe's verdict is `OfRun.accepts` and the event its survivor bound bounds is `OfRun.Kept`: each is the
preimage, under the view, of the wipe's game's. That the view is distributed as the wipe's game samples it is
`OfRun.Preserves`, which a statement over a run takes as a premise. -/
structure OfRun {O : Type} (G : Game O) where
  /-- the bits of foreign content the device holds at the boundary (S_φ) -/
  S_φ : ℕ
  hold : (Fin S_φ → Bool) → Device.State d
  /-- how block `b` reads the foreign content back: from the device's state when the fill ends, the egress and the
  spot reads -/
  keep : Device.State d → (Fin w.egress → Bool) → Ch → Fin S_φ → Bool
  view : O → (Fin S_φ → Bool) × Ω × Ch

namespace OfRun

variable {w} {O : Type} {G : Game O} (R : w.OfRun G)

/-- **The wipe's verdict on the run's outcome `o`**: the memory challenger's, at the device's state at the boundary,
the seed and the spot reads the run gives. -/
def accepts (o : O) : Prop := w.accepts (R.hold (R.view o).1) (R.view o).2.1 (R.view o).2.2

/-- **The run keeps foreign content past the survivor threshold** on the outcome `o`, at `foreignBits`: the foreign
content the device holds at the boundary is at least `foreignBits` bits plus the state the fill leaves out
(`Wipe.unfilled`), and block `b` reads it back from the device's state when the fill ends, the egress and the spot
reads. -/
def Kept (foreignBits : ℕ) (o : O) : Prop :=
  foreignBits + w.unfilled ≤ R.S_φ ∧
    R.keep (w.filled (R.hold (R.view o).1) (R.view o).2.1).1 (w.filled (R.hold (R.view o).1) (R.view o).2.1).2
      (R.view o).2.2 = (R.view o).1

/-- **The run's coins at the boundary are the wipe's** at the strategy `t`: on the run, the foreign content the device
holds at the boundary, the seed and the spot reads are uniform and independent, as the wipe's game samples them, so each
event of them has the same probability on the run as in the wipe's game. A run whose foreign content is the prover's
choice, a function of what came before the boundary rather than uniform, doesn't meet it. -/
def Preserves [Fintype Ω] [Nonempty Ω] [Fintype Ch] [Nonempty Ch] (t : Game.Strategy G) : Prop :=
  ∀ P : (Fin R.S_φ → Bool) × Ω × Ch → Prop,
    Game.prob (fun o => P (R.view o)) G t = Game.value P (Game.draw _)

end OfRun

end Wipe

end Verity
